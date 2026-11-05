# Copyright 2025 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
Integral-direct density-fitted exchange on the occupied space, without storing the DF tensor.

    K(D) C = sum_PQ X_P (J^-1)_PQ (X_Q^T C),   X_P = (P|mu nu) C,   D = C C^T

is evaluated in two streamed passes over the raw 3c integrals (aux batches):
  pass 1: T_P = C^T X_P = (P|ij)  and  rho_P = sum_i T_P,ii
  metric: Z = W (W^T T), c = tr_i Z, with W = c_aux L^-T the half metric of the DF build (decontracted aux)
  pass 2: K C += sum_P X_P Z_P,  J += sum_P (P|mu nu) c_P
Memory: naux * nocc^2 for T/Z and one block of raw integrals; the nao^2 * naux tensor is never stored.
This is the same operator as the stored-tensor DF (df.DF with the same aux basis and range separation); used by the
multi-fidelity SCF for its few exact evaluations when the DF tensor does not fit (or on request).
The long contractions over the AO index use INT8-CRT GEMMs when `nmod` is given (pass 1 needs ~15 moduli because
J^-1 amplifies the error of T, as in the DF build; pass 2 is not amplified).
'''
import numpy as np
import cupy as cp
from pyscf import lib
from gpu4pyscf.lib.cupy_helper import contract, get_avail_mem
from gpu4pyscf.gto.mole import SortedMole
from gpu4pyscf.df.int3c2e_bdiv import Int3c2eOpt, int2c2e
from gpu4pyscf.scf.jk import _check_rsh_factors


class DirectDFK:
    def __init__(self, mol, auxmol, omega=None, nmod_t=None, nmod_x=None, mem_frac=0.25):
        self.omega_in = omega
        with mol.with_range_coulomb(omega), auxmol.with_range_coulomb(omega):
            sorted_mol = SortedMole.from_mol(mol, decontract=True)
            self.opt = Int3c2eOpt(sorted_mol, auxmol).build()
        self.mol = self.opt.mol
        self.auxmol = self.opt.auxmol
        self.omega, self.lr, self.sr = _check_rsh_factors(self.mol, omega, None, None)
        self.nmod_t, self.nmod_x = nmod_t, nmod_x
        self.mem_frac = mem_frac
        self._setup = None

    def _prepare(self, nocc):
        if self._setup is not None and self._setup['nocc'] >= nocc:
            return self._setup
        opt = self.opt
        nao = self.mol.nao      # sorted, decontracted, cartesian
        pair_addresses = opt.pair_and_diag_indices(cart=True, original_ao_order=False)[0]
        i_addr, j_addr = divmod(pair_addresses, nao)
        nao_pair = len(pair_addresses)
        naux = self.auxmol.nao
        mem = get_avail_mem(exclude_memory_pool=True)
        mem_avail = mem - naux * nocc ** 2 * 8 * 2 - nao ** 2 * 8 * 2
        batch_size = max(1, min(naux, int(mem_avail * self.mem_frac / (nao_pair * 8))))
        eval_j3c, aux_sorting, _, aux_offsets = opt.int3c2e_evaluator(
            aux_batch_size=batch_size, reorder_aux=True, cart=True,
            omega=self.omega, lr_factor=self.lr, sr_factor=self.sr)
        batch_size = min(batch_size, int((aux_offsets[1:] - aux_offsets[:-1]).max()))
        blksize = max(1, min(batch_size, int(mem_avail * self.mem_frac / (nao * nao * 3 * 8))))
        # half metric in the evaluator's aux order: the same decomposition as the DF build (df._decompose_j2c:
        # Cholesky, or eigenvectors above LINEAR_DEP_THR for the near-singular long-range metric).  Applying the two
        # half metrics separately (T^ = W^T T, Z = W T^) keeps the rounding of the stored-tensor path; an explicit
        # W W^T would amplify it by the condition number (1e7 for the erf metric).
        from gpu4pyscf.df.df import _decompose_j2c
        aux_coef = _decompose_j2c(self.auxmol, aux_sorting, self.omega, self.lr, self.sr)[0]
        self._setup = dict(nocc=nocc, i_addr=cp.asarray(i_addr), j_addr=cp.asarray(j_addr), nao_pair=nao_pair,
                           eval_j3c=eval_j3c, aux_offsets=aux_offsets, batch_size=batch_size, blksize=blksize,
                           aux_coef=aux_coef, naux=naux)
        return self._setup

    def _blocks(self, st):
        '''yield (aux0, aux1, j3c [nao, nao, dk]) over all aux functions (raw integrals, sorted cart AOs)'''
        nao = self.mol.nao
        bs, blk = st['batch_size'], st['blksize']
        buf = cp.empty((st['nao_pair'] * bs))
        j3c_full = cp.zeros((nao, nao, blk))
        i_addr, j_addr = st['i_addr'], st['j_addr']
        aux1 = 0
        for kbatch in range(len(st['aux_offsets']) - 1):
            compressed = st['eval_j3c'](aux_batch_id=kbatch, out=buf)
            nb = compressed.shape[1]
            for k0, k1 in lib.prange(0, nb, blk):
                dk = k1 - k0
                aux0, aux1 = aux1, aux1 + dk
                j3c = j3c_full[:, :, :dk]
                j3c[j_addr, i_addr] = j3c[i_addr, j_addr] = compressed[:, k0:k1]
                yield aux0, aux1, j3c

    def _half(self, Cc, j3c, nmod):
        '''X[i, q, r] = sum_p C[p, i] j3c[p, q, r]'''
        nocc = Cc.shape[1]
        nao, _, dk = j3c.shape
        X = cp.empty((nocc, nao, dk))
        if nmod:
            from gpu4pyscf.df.grad import rhf as dgr
            old = dgr.JK_GRAD_CRT_MODULI
            dgr.JK_GRAD_CRT_MODULI = nmod
            try:
                dgr._crt_contract_r(cp.ascontiguousarray(Cc.T), j3c, X)
            finally:
                dgr.JK_GRAD_CRT_MODULI = old
        else:
            contract('pqr,pi->iqr', j3c, Cc, out=X)
        return X

    def kc_j(self, C, with_j=True):
        '''C: occupation-weighted occupied orbitals (nao, nocc) in the original (spherical) AO basis, D = C C^T.
        Returns (K(D) C, J(D)) in the original AO basis (J None if not with_j).'''
        mol = self.mol
        nocc = C.shape[1]
        st = self._prepare(nocc)
        Cc = cp.ascontiguousarray(mol.apply_C_dot(cp.asarray(C), axis=0))
        naux = st['naux']
        T = cp.empty((naux, nocc, nocc))
        for a0, a1, j3c in self._blocks(st):
            X = self._half(Cc, j3c, self.nmod_t)
            T[a0:a1] = contract('iqr,qj->rij', X, Cc)
        W = st['aux_coef']
        That = W.T.dot(T.reshape(naux, -1))
        Z = W.dot(That).reshape(naux, nocc, nocc)
        That = None
        T = None
        cvec = None
        if with_j:
            # rho_P = sum_i (P|ii)  ->  c = M rho  =  trace of Z over i
            cvec = cp.einsum('rii->r', Z)
        nao = mol.nao
        KC = cp.zeros((nao, nocc))
        Jc = cp.zeros((nao, nao)) if with_j else None
        for a0, a1, j3c in self._blocks(st):
            X = self._half(Cc, j3c, self.nmod_x)
            KC += contract('iqr,rij->qj', X, Z[a0:a1])
            if with_j:
                Jc += contract('pqr,r->pq', j3c, cvec[a0:a1])
        KC = mol.apply_CT_dot(KC, axis=0)
        J = mol.apply_CT_mat_C(Jc) if with_j else None
        return KC, J
