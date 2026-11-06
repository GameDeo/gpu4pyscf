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


X_CACHE = True          # keep pass-1 X for pass 2 when it fits
X_CACHE_FRAC = 0.35     # of the free device memory


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
        self.timers = None          # set to {} to accumulate synchronised per-step timings (profiling)

    def nmod_x_cached(self):
        '''moduli of the single half transform when X is cached: it feeds both T (amplified by J^-1, needs nmod_t)
        and K C (nmod_x), so the more accurate of the two'''
        if self.nmod_t is None or self.nmod_x is None:
            return None
        return max(self.nmod_t, self.nmod_x)

    def _t(self, key, t0):
        if self.timers is None:
            return None
        import time
        cp.cuda.Stream.null.synchronize()
        t = time.perf_counter()
        if t0 is not None:
            self.timers[key] = self.timers.get(key, 0.0) + t - t0
        return t

    def _prepare(self, nocc):
        if self._setup is not None and self._setup['nocc'] >= nocc:
            st = self._setup
            need = (st['nao_pair'] * st['batch_size'] + self.mol.nao ** 2 * st['blksize'] * 3) * 8 \
                + st['naux'] * nocc ** 2 * 8
            if get_avail_mem() > 1.1 * need:
                return st
            self._setup = None                 # less free memory than at planning time: re-plan the batches
        cp.get_default_memory_pool().free_all_blocks()
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
                           aux_sorting=cp.asarray(aux_sorting),
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
            t0 = self._t(None, None)
            compressed = st['eval_j3c'](aux_batch_id=kbatch, out=buf)
            t0 = self._t('int3c2e', t0)
            nb = compressed.shape[1]
            for k0, k1 in lib.prange(0, nb, blk):
                dk = k1 - k0
                aux0, aux1 = aux1, aux1 + dk
                j3c = j3c_full[:, :, :dk]
                t0 = self._t(None, None)
                j3c[j_addr, i_addr] = j3c[i_addr, j_addr] = compressed[:, k0:k1]
                self._t('unpack', t0)
                self._last_compressed = compressed[:, k0:k1]
                yield aux0, aux1, j3c

    def _compressed(self, st):
        '''yield (aux0, aux1, compressed integrals [nao_pair, dk]) without unpacking (J-only sweep)'''
        aux1 = 0
        buf = cp.empty((st['nao_pair'] * st['batch_size']))
        for kbatch in range(len(st['aux_offsets']) - 1):
            t0 = self._t(None, None)
            compressed = st['eval_j3c'](aux_batch_id=kbatch, out=buf)
            self._t('int3c2e (J sweep)', t0)
            aux0, aux1 = aux1, aux1 + compressed.shape[1]
            yield aux0, aux1, compressed

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

    def kc_j(self, C, with_j=True, keep_fit=False):
        '''C: occupation-weighted occupied orbitals (nao, nocc) in the original (spherical) AO basis, D = C C^T.
        Returns (K(D) C, J(D)) in the original AO basis (J None if not with_j).
        keep_fit: keep Z = J^-1 (P|ij) (the occupied-pair fitting coefficients of the DF nuclear gradient) for fit_oo().'''
        mol = self.mol
        nocc = C.shape[1]
        st = self._prepare(nocc)
        Cc = cp.ascontiguousarray(mol.apply_C_dot(cp.asarray(C), axis=0))
        naux = st['naux']
        T = cp.empty((naux, nocc, nocc))
        # cache the half-transformed X = C^T (pq|P) of pass 1 when it fits (nocc * nao * naux): pass 2 then skips the
        # integral recomputation, the unpack and the second half transform (only a J sweep over the integrals)
        from gpu4pyscf.lib.cupy_helper import get_avail_mem as _gam
        xbytes = 8 * nocc * mol.nao * naux
        cache = [] if (X_CACHE and xbytes < X_CACHE_FRAC * _gam()) else None
        for a0, a1, j3c in self._blocks(st):
            t0 = self._t(None, None)
            X = self._half(Cc, j3c, self.nmod_t if cache is None else self.nmod_x_cached())
            t0 = self._t('half pass1', t0)
            T[a0:a1] = contract('iqr,qj->rij', X, Cc)
            self._t('T = X C', t0)
            if cache is not None:
                cache.append((a0, a1, X))
        W = st['aux_coef']
        # Z = W (W^T T), in column blocks written back into T (naux * nocc^2 is GBs for large molecules)
        # T_P = C^T (P|pq) C is symmetric in (i, j): apply the metric to the packed i <= j columns only (exact 2x)
        t0 = self._t(None, None)
        T2 = T.reshape(naux, -1)
        iu, ju = np.triu_indices(nocc)
        pij = cp.asarray(iu * nocc + ju)
        pji = cp.asarray(ju * nocc + iu)
        cb = max(64, int(2.5e8 / (8 * naux)))
        for c0 in range(0, len(iu), cb):
            c1 = min(len(iu), c0 + cb)
            blk = W.dot(W.T.dot(T2[:, pij[c0:c1]]))
            T2[:, pij[c0:c1]] = blk
            T2[:, pji[c0:c1]] = blk
            blk = None
        Z = T
        self._t('metric', t0)
        # the kept fit waits in host memory (naux * nocc^2: GBs for large molecules) until fit_oo()
        self._fit = (Z.get(), C) if keep_fit else None
        T = T2 = None
        cvec = None
        if with_j:
            # rho_P = sum_i (P|ii)  ->  c = M rho  =  trace of Z over i
            cvec = cp.einsum('rii->r', Z)
        nao = mol.nao
        KC = cp.zeros((nao, nocc))
        Jp = cp.zeros(st['nao_pair']) if with_j else None
        if cache is not None:
            for a0, a1, X in cache:
                t0 = self._t(None, None)
                KC += contract('iqr,rij->qj', X, Z[a0:a1])
                self._t('KC += X Z', t0)
            cache = None
            if with_j:
                for a0, a1, comp in self._compressed(st):
                    t0 = self._t(None, None)
                    Jp += comp.dot(cvec[a0:a1])
                    self._t('J', t0)
        else:
          for a0, a1, j3c in self._blocks(st):
            t0 = self._t(None, None)
            X = self._half(Cc, j3c, self.nmod_x)
            t0 = self._t('half pass2', t0)
            KC += contract('iqr,rij->qj', X, Z[a0:a1])
            t0 = self._t('KC += X Z', t0)
            if with_j:
                # J on the compressed (significant) pairs: a GEMV over the aux block, no nao^2 x naux contraction
                Jp += self._last_compressed.dot(cvec[a0:a1])
                self._t('J', t0)
        KC = mol.apply_CT_dot(KC, axis=0)
        J = None
        if with_j:
            Jc = cp.zeros((nao, nao))
            Jc[st['j_addr'], st['i_addr']] = Jc[st['i_addr'], st['j_addr']] = Jp
            J = mol.apply_CT_mat_C(Jc)
        return KC, J

    def fit_oo(self, U=None):
        '''Fitting coefficients R[k, i, j] = (J^-1 (P|ij))_k of the last kc_j(keep_fit=True) call, in the native aux order
        used by df.grad.rhf.FIT_OO_CACHE (precompute_fit_oo); U rotates the occupied orbitals (C -> C U).'''
        Z, C = self._fit
        # host array (naux * nocc^2 is GBs for large molecules); rotated in aux chunks on the GPU
        R = Z[cp.asnumpy(self._setup['aux_sorting'])]
        if U is not None:
            U = cp.asarray(U)
            cb = max(1, int(2.5e8 / (8 * R.shape[1] * R.shape[2])))
            for c0 in range(0, R.shape[0], cb):
                Rc = cp.asarray(R[c0:c0 + cb])
                R[c0:c0 + cb] = cp.matmul(cp.matmul(U.T, Rc), U).get()
        return R
