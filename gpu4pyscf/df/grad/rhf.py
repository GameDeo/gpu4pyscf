# Copyright 2021-2025 The PySCF Developers. All Rights Reserved.
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

import ctypes
import numpy as np
from gpu4pyscf.lib import crt_gemm
import cupy as cp
from cupyx.scipy.linalg import solve_triangular
from pyscf import lib
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.utils import splits_by_blocksize
from gpu4pyscf.lib.cupy_helper import (
    contract, asarray, ndarray, cholesky, eigh, transpose_sum, get_avail_mem)
from gpu4pyscf.grad import rhf as rhf_grad
from gpu4pyscf.df.int3c2e_bdiv import (
    _split_l_ctr_pattern, argsort_aux, get_ao_pair_loc,
    int2c2e_ip1_per_atom, int3c2e_scheme,
    SHM_SIZE, LMAX, L_AUX_MAX, THREADS, libvhf_rys, Int3c2eOpt, int2c2e)
from gpu4pyscf.df import df
from gpu4pyscf.df.df_jk import factorize_dm, _DFHF
from gpu4pyscf.gto.mole import SortedMole, _split_l_ctr_groups
from gpu4pyscf.scf.jk import _check_rsh_factors

__all__ = ['Gradients']

# ------------------------------------------------------------------ Gamma-weighted precision tiers
# The DF J/K gradient contracts the derivative 3c integrals with the pseudo-density Gamma (known before the kernel
# runs).  Shell pairs whose Gamma weight w_sp = sum_{rows of sp, aux} |Gamma| is small contribute little; their
# derivative integrals can be evaluated by an FP32 copy of the same Rys kernel (gvhf-rys/gen_f32.py) with an error
# ~ eps32 * w_sp * |dI|.  Pairs are taken into the FP32 tier in ascending w_sp while the cumulated weight stays below
# GRAD_TIERS['frac'] of the total (per aux batch).  None: off (FP64 for every pair).
GRAD_TIERS = None
TIER_GROUP_LABELS = None   # (li, lj) of each shell-pair group of the last tiered gradient (diagnostics)
# digits of the Ozaki-I INT8 product for the pseudo-density build Gamma = C_r (Z C_l) (reduction length nocc); 0: FP64
GAMMA_OZ1_NDIG = 0
GAMMA_OZ1_BUDGET = 1.0e9
_libf32 = None


def _lib_f32():
    global _libf32
    if _libf32 is None:
        from gpu4pyscf.lib.cupy_helper import load_library
        try:
            _libf32 = load_library('libgvhf_rys_f32')
            _libf32.sum_ejk_int3c2e_ip1_f32.restype = ctypes.c_int
        except OSError:
            _libf32 = False
    return _libf32


def _envs_f32(envs):
    from gpu4pyscf.gto.mole import RysIntEnvVars
    atm, bas, env, ao_loc = envs._env_ref_holder
    obj = RysIntEnvVars.new(envs.natm, envs.nbas, atm, bas, env.astype(np.float32), ao_loc)
    return obj


def _gamma_oz1(dm_oo, fl, fr, out3, alpha, beta):
    '''out3[p, q, r] = alpha * sum_i fr[p, i] (sum_j dm_oo[r, j, i] fl[q, j]) + beta * out3  (Ozaki-I INT8 product,
    reduction length nocc), in column blocks of (q, r) under GAMMA_OZ1_BUDGET bytes of int32 digit products.'''
    ndig = GAMMA_OZ1_NDIG
    nao, _, dk = out3.shape
    nocc = fr.shape[1]
    tq = cp.empty((nao, dk, nocc))
    contract('rji,qj->qri', dm_oo, fl, out=tq)
    bt = tq.reshape(nao * dk, nocc)
    out2 = out3.reshape(nao, nao * dk)
    per_col = ndig * (nao + 16) * 4 + ndig * 64
    cb = max(dk, int(GAMMA_OZ1_BUDGET // per_col) // dk * dk)
    for c0 in range(0, nao * dk, cb):
        c1 = min(nao * dk, c0 + cb)
        crt_gemm.ozaki1_gemm_nt(fr, bt[c0:c1], ndig, out=out2[:, c0:c1], alpha=alpha, beta=beta)


_abs_rowsum = cp.ReductionKernel('T x', 'T y', 'abs(x)', 'a + b', 'y = a', '0', 'abs_rowsum')


def _tier_split(compressed, ao_pair_loc, group_sp, frac):
    '''Boolean mask over shell pairs (FP32 tier) from the Gamma weights of this aux batch.'''
    # fused |x| row sums: abs(compressed) would allocate a copy of the whole batch (GPU OOM at Taxol size)
    rowl1 = _abs_rowsum(compressed, axis=1)
    cs = cp.concatenate([cp.zeros(1), cp.cumsum(rowl1)])
    w = cs[ao_pair_loc[1:]] - cs[ao_pair_loc[:-1]]
    order = cp.argsort(w)
    cw = cp.cumsum(w[order])
    nlow = int(cp.searchsorted(cw, frac * float(cw[-1]), side='right'))
    low = cp.zeros(w.size, dtype=bool)
    low[order[:nlow]] = True
    return low


def _tier_tables(mask, bas_ij_idx, ao_pair_loc, group_sp, group_bs):
    '''(bas_ij_idx, shl_pair_offsets, ao_pair_loc) restricted to the shell pairs in mask; blocks never mix (li, lj)
    groups (the kernel requires one angular pattern per block).'''
    idx_parts, loc_parts, offs = [], [], []
    n = 0
    m = mask.get()
    for (s0, s1), bs in zip(group_sp, group_bs):
        sel = np.nonzero(m[s0:s1])[0] + s0
        if sel.size == 0:
            continue
        selg = cp.asarray(sel)
        idx_parts.append(bas_ij_idx[selg])
        loc_parts.append(ao_pair_loc[selg])
        offs.append(np.arange(n, n + sel.size, bs, dtype=np.int32))
        n += sel.size
    if n == 0:
        return None
    offs.append(np.array([n], dtype=np.int32))
    return (cp.asarray(cp.hstack(idx_parts), dtype=np.uint32), cp.asarray(np.hstack(offs), dtype=np.int32),
            cp.asarray(cp.hstack(loc_parts), dtype=np.int32))


def _gen_metric_solver(int2c, decompose_j2c='CD', lindep=df.LINEAR_DEP_THR):
    ''' generate a solver to solve Ax = b, RHS must be in (n,....) '''
    if decompose_j2c.upper() == 'CD':
        try:
            j2c = cholesky(int2c)
            def j2c_solver(b):
                b_ = b.reshape(j2c.shape[0], -1)
                y = solve_triangular(j2c, b_, lower=True, overwrite_b=False)
                x = solve_triangular(j2c.conj().T, y, lower=False, overwrite_b=False)
                return cp.asarray(x.reshape(b.shape), order='A')
            return j2c_solver
        except RuntimeError:
            pass

    w, v = eigh(int2c)
    mask = w > lindep
    inv_w = 1 / w[mask]
    v1 = v[:,mask]
    def j2c_solver(b): # noqa: F811
        x = v1.conj().T @ b.reshape(int2c.shape[0],-1)
        x = x * inv_w[:,None]
        x = v1 @ x
        return x.reshape(b.shape)
    return j2c_solver

# Number of INT8 moduli for the two large contractions of the DF J/K gradient (0: FP64).  Set by
# gpu4pyscf.scf.mixed_precision.grad_stage for the 'crt' scheme.
JK_GRAD_CRT_MODULI = 0
# Occupied-pair fitting coefficients R = J^-1 (P|ij) in the native decontracted aux order, precomputed from the SCF
# DF tensors by the multi-fidelity SCF (keys 'fr', 'lr'; 'C' = the occupation-weighted occupied orbitals they belong
# to).  When present and matching, the gradient skips the raw 3c recomputation and the metric solve for j3c_oo.
FIT_OO_CACHE = None
JK_GRAD_CRT_BUDGET = 6e8     # bytes of INT8-CRT temporaries per sub-block

def _crt_contract_r(a, src3, out3, alpha=1.0, beta=0.0):
    '''out3[m, q, r] = alpha * sum_k a[m, k] src3[k, q, r] + beta * out3[m, q, r] with INT8-CRT GEMMs, in chunks
    of the last (aux) index so that the residue planes and int32 products stay under JK_GRAD_CRT_BUDGET.'''
    nmod = JK_GRAD_CRT_MODULI
    K, nq, nr = src3.shape
    m = a.shape[0]
    per_r = nq * (K * (nmod + 8) + m * (4 * nmod + 16))
    sb = max(1, int(JK_GRAD_CRT_BUDGET // per_r))
    for r0 in range(0, nr, sb):
        r1 = min(nr, r0 + sb)
        b = cp.ascontiguousarray(src3[:, :, r0:r1]).reshape(K, nq * (r1 - r0))
        g = crt_gemm.gemm_dense(a, b, nmod, budget=JK_GRAD_CRT_BUDGET / 2).reshape(m, nq, r1 - r0)
        b = None
        if beta:
            out3[:, :, r0:r1] *= beta
            out3[:, :, r0:r1] += alpha * g
        elif alpha == 1.0:
            out3[:, :, r0:r1] = g
        else:
            out3[:, :, r0:r1] = alpha * g

def _cached_fit_oo(dm, omega, nocc, lr_factor=1, sr_factor=1):
    cache = FIT_OO_CACHE
    if cache is None:
        return None
    # only the plain operators the cache was built for: full range (omega 0, 1, 1) and pure long range (omega, 1, 0)
    if omega == 0 and (lr_factor, sr_factor) == (1, 1):
        key = 'fr'
    elif omega > 0 and (lr_factor, sr_factor) == (1, 0) and abs(omega - cache.get('omega', -1)) < 1e-12:
        key = 'lr'
    else:
        return None
    R = cache.get(key)
    if R is None or R.shape[1] != nocc or not hasattr(dm, 'mo_coeff'):
        return None
    from gpu4pyscf.df.df_jk import factorize_dm
    f = factorize_dm(dm, 1)[0]
    if f.shape != cache['C'].shape or float(abs(f - cache['C']).max()) > 1e-10:
        return None
    return R


def precompute_fit_oo(dfobj, C, omega=None, lr_factor=None, sr_factor=None, nmod=15):
    """R[k, i, j] = (J^-1 (P|ij))_k in the native decontracted aux order of dfobj.intopt.auxmol, from the stored DF
    tensor B (= L^-1 c^T (mu nu|P)):  R = aux_coef (C^T B C).  C: occupation-weighted occupied orbitals (nao, nocc)."""
    from gpu4pyscf.df.df import _decompose_j2c
    nao, nocc = C.shape
    ct = cp.ascontiguousarray(C.T)
    T = []
    for cderi, _ in dfobj.loop(unpack=True):
        nL = len(cderi)
        view = cderi.transpose(1, 0, 2)                                   # [i, L, j], L contiguous
        if view.strides[1] != view.itemsize:
            view = cp.ascontiguousarray(cderi.transpose(1, 2, 0)).transpose(0, 2, 1)
        X, P1 = crt_gemm.gemm_planes(view, ct, nmod)                      # rows (i, L), cols k
        X = X.reshape(nao, P1, nocc)[:, :nL]
        T.append(cp.einsum('ilk,im->lkm', X, C))
    T = cp.concatenate(T, axis=0)
    aux_coef, _ = _decompose_j2c(dfobj.intopt.auxmol, None, omega, lr_factor, sr_factor)
    return cp.dot(aux_coef, T.reshape(T.shape[0], -1)).reshape(-1, nocc, nocc)


def _jk_energy_per_atom(int3c2e_opt, dm, j_factor=1, k_factor=1, hermi=0,
                        auxbasis_response=True, verbose=None,
                        omega=None, lr_factor=None, sr_factor=None):
    '''
    Computes the first-order derivatives of the energy contributions from
    J and K terms per atom.
    '''
    if hermi == 2:
        j_factor = 0
    if k_factor == 0:
        assert omega is None or omega == 0
        return _j_energy_per_atom(int3c2e_opt, dm, hermi, auxbasis_response,
                                  verbose) * j_factor

    if j_factor != 0 and (omega is not None and omega != 0):
        raise RuntimeError(
            'Cannot compute J contributions with range-separated Coulomb operator.')

    mol = int3c2e_opt.mol
    auxmol = int3c2e_opt.auxmol
    log = logger.new_logger(mol, verbose)
    t0 = log.init_timer()

    dm_factor_l, dm_factor_r = _factorize_dm(mol, dm, hermi)
    nao, nocc = dm_factor_l.shape
    log.debug1('dm_factor shape %s', dm_factor_l.shape)

    pair_addresses = int3c2e_opt.pair_and_diag_indices(
        cart=True, original_ao_order=False)[0]
    i_addr, j_addr = divmod(pair_addresses, nao)
    nao_pair = len(pair_addresses)
    naux = auxmol.nao

    omega, lr_factor, sr_factor = _check_rsh_factors(mol, omega, lr_factor, sr_factor)

    mem_free = get_avail_mem(exclude_memory_pool=True)
    mem_avail = mem_free - naux*nocc**2*8 - nao**2*8
    # the FP32 tier keeps an FP32 copy of each batch (+50 %)
    batch_frac = .5 if not (GRAD_TIERS and _lib_f32()) else .33
    batch_size = max(1, min(naux, int(mem_avail*batch_frac/(nao_pair*8))))
    eval_j3c, aux_sorting, _, aux_offsets = int3c2e_opt.int3c2e_evaluator(
        aux_batch_size=batch_size, reorder_aux=True, cart=True,
        omega=omega, lr_factor=lr_factor, sr_factor=sr_factor)
    batch_size = min(batch_size, int((aux_offsets[1:]-aux_offsets[:-1]).max()))
    aux_batches = len(aux_offsets) - 1

    blksize = max(1, min(batch_size, int(mem_avail*.4/(nao*nao*2*8))//8*8))
    log.debug1('%.3f GB free memory. nao_pair=%d naux=%d batch_size=%d blksize=%d',
               mem_free*1e-9, nao_pair, naux, batch_size, blksize)

    fit_oo = _cached_fit_oo(dm, omega, nocc, lr_factor, sr_factor)
    aux0 = aux1 = 0
    if fit_oo is not None:
        aux_batches = 0
        log.debug('DF gradient: occupied-pair fitting coefficients taken from the SCF DF tensor')
    else:
        j3c_full = cp.zeros((nao, nao, blksize))
        buf = cp.empty((batch_size, nao_pair))
        buf1 = cp.empty((blksize, nocc, nao))
        j3c_oo = cp.empty((naux, nocc, nocc))
    for kbatch in range(aux_batches):
        compressed = eval_j3c(aux_batch_id=kbatch, out=buf)
        naux_in_batch = compressed.shape[1]
        for k0, k1 in lib.prange(0, naux_in_batch, blksize):
            dk = k1 - k0
            aux0, aux1 = aux1, aux1 + dk
            j3c = j3c_full[:,:,:dk]
            j3c[j_addr,i_addr] = j3c[i_addr,j_addr] = compressed[:,k0:k1]
            tmp = ndarray((nocc, nao, dk), buffer=buf1)
            if JK_GRAD_CRT_MODULI:
                # long reduction over the AO index: INT8-CRT emulated FP64 GEMM
                _crt_contract_r(cp.ascontiguousarray(dm_factor_r.T), j3c, tmp)
            else:
                contract('pqr,pi->iqr', j3c, dm_factor_r, out=tmp)
            contract('iqr,qj->rij', tmp, dm_factor_l, out=j3c_oo[aux0:aux1])
    j3c_full = buf = buf1 = eval_j3c = j3c = tmp = compressed = None
    t0 = log.timer_debug1('contract dm', *t0)

    aux_coeff = cp.asarray(auxmol.ctr_coeff)
    aux_coeff, tmp = cp.empty_like(aux_coeff), aux_coeff
    aux_coeff[aux_sorting] = tmp
    tmp = None

    if fit_oo is not None:
        if isinstance(fit_oo, np.ndarray):
            # host-resident fit (large molecules): permute on the host, one device copy
            tmp = np.empty_like(fit_oo)
            tmp[cp.asnumpy(aux_sorting)] = fit_oo
            dm_oo = cp.asarray(tmp)
            tmp = None
        else:
            dm_oo = cp.empty_like(fit_oo)
            dm_oo[aux_sorting] = fit_oo
        aux_coeff = None
    else:
        j2c = int2c2e(auxmol, omega=omega, lr_factor=lr_factor, sr_factor=sr_factor)
        if mol.omega <= 0 and not auxmol.mol.cart:
            metric = aux_coeff.dot(_gen_metric_solver(j2c, 'CD')(aux_coeff.T))
        else:
            metric = aux_coeff.dot(_gen_metric_solver(j2c, 'ED')(aux_coeff.T))
        j2c = aux_coeff = None
        dm_oo = cp.einsum('uv,vij->uij', metric, j3c_oo)
        metric = j3c_oo = None
    if j_factor != 0:
        auxvec = dm_oo.trace(axis1=1, axis2=2)

    # contract the derivatives and the pseudo DM/rho
    nsp_per_block, gout_stride, shm_size = int3c2e_scheme(
        short_range=omega<0, gout_width=54, deriv=(1,0,0))
    gout_stride = cp.asarray(gout_stride, dtype=np.int32)
    lmax = mol.uniq_l_ctr[:,0].max()
    laux = auxmol.uniq_l_ctr[:,0].max()
    shm_size_max = shm_size[:laux+1,:lmax+1,:lmax+1].max()

    pair_batch_size = np.minimum(nsp_per_block[0]*4,
                                 np.maximum(nsp_per_block[0]//4, 16))
    bas_ij_idx, shl_pair_offsets = mol.aggregate_shl_pairs(
        int3c2e_opt.bas_ij_cache, pair_batch_size)
    ao_pair_loc = get_ao_pair_loc(mol.uniq_l_ctr[:,0], int3c2e_opt.bas_ij_cache)
    aux_loc = auxmol.ao_loc

    l_ctr_aux_offsets = np.append(0, np.cumsum(auxmol.l_ctr_counts))
    l_ctr_aux_offsets, uniq_l_ctr_aux = _split_l_ctr_pattern(
        l_ctr_aux_offsets, auxmol.uniq_l_ctr, batch_size)
    # assert cp.array_equal(aux_sorting, argsort_aux(l_ctr_aux_offsets, uniq_l_ctr_aux))
    ksh_offsets_cpu = l_ctr_aux_offsets
    ksh_offsets_gpu = cp.asarray(ksh_offsets_cpu+mol.nbas, dtype=np.int32)
    aux_offsets = aux_loc[ksh_offsets_cpu]
    aux_batches = len(aux_offsets) - 1

    if j_factor != 0:
        dm = dm_factor_l.dot(dm_factor_r.T)

    tiers = GRAD_TIERS if (GRAD_TIERS and _lib_f32()) else None
    if tiers:
        group_sp, group_bs = [], []
        sp1 = 0
        for (gi, gj), bas_ij in int3c2e_opt.bas_ij_cache.items():
            sp0, sp1 = sp1, sp1 + len(bas_ij)
            group_sp.append((sp0, sp1))
            group_bs.append(int(pair_batch_size) if np.ndim(pair_batch_size) == 0 else
                            int(pair_batch_size[mol.uniq_l_ctr[gi, 0], mol.uniq_l_ctr[gj, 0]]))
        envs32 = _envs_f32(int3c2e_opt.int3c2e_envs)
        pair_maxexp = None
        if 'exp_thr' in tiers:
            # tightest primitive exponent of each shell pair (diagnostic tier: FP32 only for diffuse/valence pairs)
            from pyscf.gto.mole import PTR_EXP, NPRIM_OF
            b_ = mol._bas
            mexp = np.array([mol._env[b_[i, PTR_EXP]:b_[i, PTR_EXP] + b_[i, NPRIM_OF]].max() for i in range(mol.nbas)])
            ij_ = bas_ij_idx.get().astype(np.int64)
            pair_maxexp = cp.asarray(np.maximum(mexp[ij_ // mol.nbas], mexp[ij_ % mol.nbas]))
        tier_stats = [0, 0]
        global TIER_GROUP_LABELS
        TIER_GROUP_LABELS = [(int(mol.uniq_l_ctr[gi, 0]), int(mol.uniq_l_ctr[gj, 0]))
                             for gi, gj in int3c2e_opt.bas_ij_cache.keys()]

    int3c2e_envs = int3c2e_opt.int3c2e_envs
    kern = libvhf_rys.sum_ejk_int3c2e_ip1
    aux0 = aux1 = 0
    buf = cp.empty((nao_pair*batch_size))
    buf1 = cp.empty((blksize, nao, nao))
    # Only non-Hermitian densities reuse this buffer for an AO-sized transpose.
    buf2 = cp.empty((blksize, nocc if hermi == 1 else nao, nao))
    ejk = cp.zeros((mol.natm, 3))
    ejk_aux = cp.zeros((mol.natm, 3))
    aux_splits = splits_by_blocksize(aux_offsets, batch_size)
    for kbatch, kend in zip(aux_splits[:-1], aux_splits[1:]):
        naux_in_batch = aux_offsets[kend] - aux_offsets[kbatch]
        aux_ao_offset = aux_loc[ksh_offsets_cpu[kbatch]]
        compressed = ndarray((nao_pair, naux_in_batch), buffer=buf)
        for k0, k1 in lib.prange(0, naux_in_batch, blksize):
            dk = k1 - k0
            aux0, aux1 = aux1, aux1 + dk
            dm_tensor = ndarray((nao,nao,dk), buffer=buf1)
            tmp = ndarray((nocc,nao,dk), buffer=buf2)
            beta = 0
            if j_factor != 0:
                cp.multiply(dm[:,:,None], auxvec[aux0:aux1], out=dm_tensor)
                beta = j_factor
            if GAMMA_OZ1_NDIG:
                _gamma_oz1(dm_oo[aux0:aux1], dm_factor_l, dm_factor_r, dm_tensor, -.5 * k_factor, beta)
            else:
                contract('rji,qj->iqr', dm_oo[aux0:aux1], dm_factor_l, out=tmp)
                if JK_GRAD_CRT_MODULI:
                    _crt_contract_r(cp.ascontiguousarray(dm_factor_r), tmp, dm_tensor,
                                    alpha=-.5 * k_factor, beta=beta)
                else:
                    contract('iqr,pi->pqr', tmp, dm_factor_r, -.5*k_factor, beta, out=dm_tensor)
            if hermi == 1:
                cp.take(dm_tensor.reshape(-1,dk), pair_addresses, axis=0,
                        out=compressed[:,k0:k1])
            else:
                dm_tensor1 = ndarray((nao,nao,dk), buffer=buf2)
                dm_tensor1[:] = dm_tensor.transpose(1,0,2)
                dm_tensor1[:] += dm_tensor
                cp.take(dm_tensor1.reshape(-1,dk), pair_addresses, axis=0,
                        out=compressed[:,k0:k1])
        if tiers:
            if 'groups' in tiers:
                # diagnostics: whole (li, lj) groups (indices into TIER_GROUP_LABELS) in the FP32 tier
                low = cp.zeros(len(ao_pair_loc) - 1, dtype=bool)
                for gidx in tiers['groups']:
                    low[group_sp[gidx][0]:group_sp[gidx][1]] = True
            else:
                low = _tier_split(compressed, ao_pair_loc, group_sp, tiers['frac']) if tiers.get('frac') else \
                    cp.zeros(len(ao_pair_loc) - 1, dtype=bool)
                if pair_maxexp is not None:
                    low = (low | (pair_maxexp < tiers['exp_thr'])) if tiers.get('exp_or', True) else \
                        (low & (pair_maxexp < tiers['exp_thr']))
            tab_hi = _tier_tables(~low, bas_ij_idx, ao_pair_loc, group_sp, group_bs)
            # 'skip': the low tier is dropped (Gamma-weighted screening) instead of evaluated in FP32
            tab_lo = None if tiers.get('skip') else _tier_tables(low, bas_ij_idx, ao_pair_loc, group_sp, group_bs)
            tier_stats[0] += int(low.sum()); tier_stats[1] += low.size
        else:
            tab_hi, tab_lo = (bas_ij_idx, shl_pair_offsets, ao_pair_loc), None
        if tab_hi is not None:
            t_idx, t_off, t_loc = tab_hi
            err = kern(
                ctypes.cast(ejk.data.ptr, ctypes.c_void_p),
                ctypes.cast(ejk_aux.data.ptr, ctypes.c_void_p),
                ctypes.cast(compressed.data.ptr, ctypes.c_void_p),
                lib.c_null_ptr(),
                ctypes.c_int(1),
                ctypes.c_double(omega),
                ctypes.c_double(lr_factor),
                ctypes.c_double(sr_factor),
                ctypes.byref(int3c2e_envs),
                ctypes.c_int(shm_size_max),
                ctypes.c_int(len(t_off) - 1),
                ctypes.c_int(kend-kbatch),
                ctypes.cast(t_off.data.ptr, ctypes.c_void_p),
                ctypes.cast(t_idx.data.ptr, ctypes.c_void_p),
                ctypes.cast(ksh_offsets_gpu[kbatch:].data.ptr, ctypes.c_void_p),
                ctypes.cast(gout_stride.data.ptr, ctypes.c_void_p),
                ctypes.cast(t_loc.data.ptr, ctypes.c_void_p),
                ctypes.c_int(aux_ao_offset),
                ctypes.c_int(nao), ctypes.c_int(nao_pair),
                ctypes.c_int(naux_in_batch), ctypes.c_int(mol.natm))
            if err != 0:
                raise RuntimeError('int3c2e_ejk_ip1 failed')
        if tab_lo is not None:
            t_idx, t_off, t_loc = tab_lo
            # FP64 accumulators (gen_f32.py ACC64): FP32 integrals, FP64 sums over aux shells and atoms
            ejk32 = cp.zeros((mol.natm, 3))
            ejk32_aux = cp.zeros((mol.natm, 3))
            comp32 = compressed.astype(np.float32)
            err = _lib_f32().sum_ejk_int3c2e_ip1_f32(
                ctypes.cast(ejk32.data.ptr, ctypes.c_void_p),
                ctypes.cast(ejk32_aux.data.ptr, ctypes.c_void_p),
                ctypes.cast(comp32.data.ptr, ctypes.c_void_p),
                lib.c_null_ptr(),
                ctypes.c_int(1),
                ctypes.c_float(omega),
                ctypes.c_float(lr_factor),
                ctypes.c_float(sr_factor),
                ctypes.byref(envs32),
                ctypes.c_int(shm_size_max),
                ctypes.c_int(len(t_off) - 1),
                ctypes.c_int(kend-kbatch),
                ctypes.cast(t_off.data.ptr, ctypes.c_void_p),
                ctypes.cast(t_idx.data.ptr, ctypes.c_void_p),
                ctypes.cast(ksh_offsets_gpu[kbatch:].data.ptr, ctypes.c_void_p),
                ctypes.cast(gout_stride.data.ptr, ctypes.c_void_p),
                ctypes.cast(t_loc.data.ptr, ctypes.c_void_p),
                ctypes.c_int(aux_ao_offset),
                ctypes.c_int(nao), ctypes.c_int(nao_pair),
                ctypes.c_int(naux_in_batch), ctypes.c_int(mol.natm))
            if err != 0:
                raise RuntimeError('int3c2e_ejk_ip1 (FP32 tier) failed')
            ejk += ejk32
            ejk_aux += ejk32_aux
            comp32 = None
    buf = buf1 = buf2 = compressed = dm_tensor = dm_tensor1 = tmp = None
    if hermi == 1:
        ejk *= 2
        ejk_aux *= 2
    if tiers:
        log.debug('DF gradient tiers: %d of %d shell-pair blocks in FP32', *tier_stats)
    t0 = log.timer_debug1('contract int3c2e_ejk_ip1', *t0)

    if auxbasis_response:
        # (d/dX P|Q) contributions
        if j_factor == 0:
            dm_aux = None
        else:
            dm_aux = auxvec[:,None] * auxvec
        if dm_factor_l is dm_factor_r:
            dm_aux = contract('rij,sij->rs', dm_oo, dm_oo,
                              alpha=-.5*k_factor, beta=j_factor, out=dm_aux)
        else:
            dm_aux = contract('rij,sji->rs', dm_oo, dm_oo,
                              alpha=-.5*k_factor, beta=j_factor, out=dm_aux)
        dm_aux = dm_aux[aux_sorting[:,None], aux_sorting]
        #ejk_aux += .5*contract_h1e_dm(auxmol, auxmol.intor('int2c2e_ip1'), dm_aux)
        ejk_aux -= cp.asarray(
            int2c2e_ip1_per_atom(auxmol, dm_aux, omega, lr_factor, sr_factor))
        t0 = log.timer_debug1('contract int2c2e_ip1', *t0)
        ejk += ejk_aux

    ejk = ejk.get()
    return ejk

def _j_energy_per_atom(int3c2e_opt, dm, hermi=0, auxbasis_response=True, verbose=None):
    '''
    Computes the first-order derivatives of the Coulomb energy
    '''
    mol = int3c2e_opt.mol
    auxmol = int3c2e_opt.auxmol
    log = logger.new_logger(mol, verbose)
    t0 = log.init_timer()

    dm = mol.apply_C_mat_CT(dm)
    if hermi != 1:
        dm = transpose_sum(dm, inplace=True)
        dm[:] *= .5
    auxvec = int3c2e_opt.contract_dm(dm, hermi=1)
    naux = len(auxvec)
    t0 = log.timer_debug1('contract dm', *t0)
    j2c = int2c2e(auxmol)

    auxvec = auxmol.CT_dot_mat(auxvec)
    if mol.omega <= 0 and not auxmol.mol.cart:
        auxvec = _gen_metric_solver(j2c, 'CD')(auxvec)
    else:
        auxvec = _gen_metric_solver(j2c, 'ED')(auxvec)
    auxvec = auxmol.C_dot_mat(auxvec)
    j2c = None

    omega, lr_factor, sr_factor = 0., 1., 1.

    nsp_per_block, gout_stride, shm_size = int3c2e_scheme(
        short_range=omega<0, gout_width=54, deriv=(1,0,0))
    lmax = mol.uniq_l_ctr[:,0].max()
    laux = auxmol.uniq_l_ctr[:,0].max()
    shm_size_max = shm_size[:laux+1,:lmax+1,:lmax+1].max()
    bas_ij_idx, shl_pair_offsets = mol.aggregate_shl_pairs(
        int3c2e_opt.bas_ij_cache, nsp_per_block[0]*16)
    # Split auxbasis into small batches for load balance
    _, aux_counts = _split_l_ctr_groups(
        auxmol.uniq_l_ctr, auxmol.l_ctr_counts, [16]*(laux+1))
    ksh_offsets_cpu = np.append(0, np.cumsum(aux_counts))
    ksh_offsets_gpu = cp.asarray(ksh_offsets_cpu+mol.nbas, dtype=np.int32)

    int3c2e_envs = int3c2e_opt.int3c2e_envs
    kern = libvhf_rys.sum_ejk_int3c2e_ip1
    ej = cp.zeros((mol.natm, 3))
    if auxbasis_response:
        ej_aux = cp.zeros_like(ej)
        ej_aux_ptr = ctypes.cast(ej_aux.data.ptr, ctypes.c_void_p)
    else:
        ej_aux_ptr = lib.c_null_ptr()

    err = kern(
        ctypes.cast(ej.data.ptr, ctypes.c_void_p), ej_aux_ptr,
        ctypes.cast(dm.data.ptr, ctypes.c_void_p),
        ctypes.cast(auxvec.data.ptr, ctypes.c_void_p),
        ctypes.c_int(1),
        ctypes.c_double(omega),
        ctypes.c_double(lr_factor),
        ctypes.c_double(sr_factor),
        ctypes.byref(int3c2e_envs),
        ctypes.c_int(shm_size_max),
        ctypes.c_int(len(shl_pair_offsets) - 1),
        ctypes.c_int(len(ksh_offsets_cpu) - 1),
        ctypes.cast(shl_pair_offsets.data.ptr, ctypes.c_void_p),
        ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
        ctypes.cast(ksh_offsets_gpu.data.ptr, ctypes.c_void_p),
        ctypes.cast(gout_stride.data.ptr, ctypes.c_void_p),
        lib.c_null_ptr(),
        ctypes.c_int(0),
        ctypes.c_int(0), ctypes.c_int(0),
        ctypes.c_int(naux), ctypes.c_int(mol.natm))
    if err != 0:
        raise RuntimeError('int3c2e_ejk_ip1 failed')
    ej *= 2
    ej = ej.get()
    t0 = log.timer_debug1('contract int3c2e_ejk_ip1', *t0)

    # (d/dX P|Q) contributions
    if auxbasis_response:
        ej_aux *= 2
        #ej_aux += .5*contract_h1e_dm(auxmol, auxmol.intor('int2c2e_ip1'), dm_aux)
        dm_aux = auxvec[:,None] * auxvec
        ej_aux -= cp.asarray(int2c2e_ip1_per_atom(auxmol, dm_aux))
        ej += ej_aux.get()
    t0 = log.timer_debug1('contract int2c2e_ip1', *t0)
    return ej

def _factorize_dm(mol, dm, hermi):
    dm_factor_l, dm_factor_r = factorize_dm(dm, hermi)
    axis = dm_factor_l.ndim - 2
    dm_factor_l = mol.apply_C_dot(dm_factor_l, axis=axis)
    if dm_factor_r is None:
        dm_factor_r = dm_factor_l
    else:
        dm_factor_r = mol.apply_C_dot(dm_factor_r, axis=axis)
    if hasattr(dm, 'symmetrize'):
        # See the convention in _make_factorized_dm provided by df_jk.py
        if dm.symmetrize == 1:
            dm_factor_l, dm_factor_r = (cp.hstack([dm_factor_l, dm_factor_r]),
                                        cp.hstack([dm_factor_r, dm_factor_l]))
        elif dm.symmetrize == 2:
            dm_factor_l, dm_factor_r = (cp.hstack([dm_factor_l, dm_factor_r]),
                                        cp.hstack([dm_factor_r,-dm_factor_l]))
    return dm_factor_l, dm_factor_r

class Gradients(rhf_grad.Gradients):

    _keys = {'with_df', 'auxbasis_response'}

    auxbasis_response = True

    def check_sanity(self):
        assert isinstance(self.base, df.df_jk._DFHF)

    def energy_ee(self, mol, dm):
        return self.jk_energy_per_atom(dm, hermi=1)

    def jk_energy_per_atom(self, dm=None, j_factor=1, k_factor=1, *,
                           omega=None, lr_factor=None, sr_factor=None,
                           hermi=0, verbose=None):
        r'''Compute the first-order derivatives of the Coulomb and exchange
        energies with respect to atomic positions.

            j_factor * dE_J / dR - k_factor * dE_K / dR

        Parameters
        ----------
        dm : ndarray or sequence of ndarray, optional
            Density matrix or (for UKS) a sequence of density matrices.
        j_factor : float, optional
            Scaling factor applied to the Coulomb (J) contribution.
        k_factor : float, optional
            Scaling factor applied to the exchange (K) contribution.
        omega : float, optional
            Range-separation parameter. Together with ``lr_factor`` and
            ``sr_factor``, defines the range-separated Coulomb operator used in
            exchange (K) matrix evaluation:
                lr_factor * erf(omega * r) / r + sr_factor * erfc(omega * r) / r.
        lr_factor : float, optional
            Scaling factor for the long-range interaction,
        sr_factor : float, optional
            Scaling factor for the short-range interaction,
        hermi : int, optional
            Symmetry of the density matrix:
            - ``0``: no symmetry
            - ``1``: Hermitian
            - ``2``: anti-Hermitian

        Notes
        -----
        When omega, lr_factor, and sr_factor parameters are specified, the K
        contribution is evaluated using the attenuated Coulomb operator.

        Returns
        -------
        Array of shape ``(natm, 3)`` containing the derivative of the weighted
        Coulomb and exchange energy with respect to the Cartesian coordinates of
        each atom.
        '''
        mf = self.base
        if dm is None: dm = mf.make_rdm1()
        mol = mf.with_df.mol
        auxmol = mf.with_df.auxmol
        mf.with_df.reset() # Release GPU memory
        with mol.with_range_coulomb(omega), auxmol.with_range_coulomb(omega):
            sorted_mol = SortedMole.from_mol(mol, decontract=True)
            int3c2e_opt = Int3c2eOpt(sorted_mol, auxmol).build()
        return _jk_energy_per_atom(
            int3c2e_opt, dm, j_factor, k_factor, hermi=hermi,
            auxbasis_response=self.auxbasis_response, verbose=verbose,
            omega=omega, lr_factor=lr_factor, sr_factor=sr_factor)

Grad = Gradients
