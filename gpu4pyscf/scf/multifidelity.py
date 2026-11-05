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
Multi-fidelity (defect-corrected) SCF for closed-shell density-fitted Kohn-Sham with range-separated hybrids
(wB97M-V and friends).  See benchmarks/dft/research/research_report4_novel_algorithms.md.

    exact Fock      F(D)   = h + J + V_xc^fine + V_nlc^fine - 1/2 K_exact
    surrogate       F_s(D) = h + J + V_xc^coarse - 1/2 [K_s(D_ref) + K_s^fp16(D - D_ref)] + Delta~

* The exact operator is only applied to the occupied orbitals (W = F C_occ) at a few "outer" steps.  The defect
  W - F_cheap(D_ref) C_occ is folded into the symmetric rank-2 nocc operator
      Delta~ = W C^T S + S C W^T - S C (C^T W) C^T S
  which reproduces F(D_ref) C_occ exactly, so the converged density is the one of the exact Fock matrix.
* Inner iterations use a coarse XC grid (no VV10), and an exchange surrogate: one DF tensor of the mixed operator
  alpha erf/r + hyb erfc/r in a smaller auxiliary basis, applied as K_s(D_ref) (accurate) plus the increment
  K_s(D - D_ref) on FP16 tensor cores (the rounding error scales with the small increment).
* The inner eigensolver is FP32 syevd + Ogita-Aishima refinement in FP64.

Usage: ``multifidelity.kernel(mf)`` for an RKS object with density fitting; or ``mf.multifidelity = True`` (or a
dict of options) before ``mf.kernel()``.
'''

import time
import numpy as np
import cupy
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import tag_array

DEFAULTS = dict(
    inner_grid=(30, 110),         # coarse XC grid of the surrogate ((20,50): 4 outers on BPA, (30,110): 3)
    stage1_nlc_grid=(10, 26),     # VV10 grid of the first (cheap) stage; later VV10 is frozen inside Delta~
    surrogate_aux='def2-svp-ri',  # aux basis of the exchange surrogate
    k_increment='fp32',           # 'fp32' | 'fp16' | 'exact' (exact = surrogate DF-K at every inner step)
                                  # fp16 noise (~1e-3 of K(dD)) stalls the inner loop for BPA-size systems
    stage1_conv_grad=1e-3,
    inner_factor=1e-2,            # inner tolerance = inner_factor * |g_true| of the last outer step
    conv_tol_grad=None,           # outer tolerance on |g| (pyscf orbital-gradient norm); default 0.1*sqrt(conv_tol)
    max_outer=12,
    max_inner=40,
    max_stage1=60,
    fine_vco=True,                # fine XC as V_xc C_occ (occupied projection) instead of the full matrix
    fuse_grad=True,               # XC nuclear gradient in the (predicted) last fine pass; used by the DF-RKS gradient
    eig='fp64',                   # 'fp64' | 'fp32oa' (FP32 syevd + Ogita-Aishima; slower than syevd at nao~600 on T4)
    precision=None,               # dict for mixed_precision.fast_precision; default from mf.mixed_precision / GPU
)


def _sync():
    cupy.cuda.Stream.null.synchronize()


class _Timer:
    def __init__(self):
        self.t = {}

    def __call__(self, key):
        tim = self

        class _Ctx:
            def __enter__(self):
                _sync()
                self.t0 = time.perf_counter()

            def __exit__(self, *a):
                _sync()
                v = tim.t.setdefault(key, [0.0, 0])
                v[0] += time.perf_counter() - self.t0
                v[1] += 1
        return _Ctx()


# ------------------------------------------------------------------ FP16 tensor-core GEMM with FP32 output
def gemm16_nt(a, b, out=None):
    '''out (m, n) float32 = a (m, k) @ b (n, k)^T for C-contiguous float16 a, b (cublasGemmEx, FP32 accumulate).'''
    from cupy.cuda import cublas, device
    assert a.dtype == np.float16 and b.dtype == np.float16
    assert a.flags.c_contiguous and b.flags.c_contiguous
    m, k = a.shape
    n = b.shape[0]
    assert b.shape[1] == k
    if out is None:
        out = cupy.empty((m, n), dtype=np.float32)
    alpha = np.array(1, dtype=np.float32)
    beta = np.array(0, dtype=np.float32)
    h = device.get_cublas_handle()
    # column-major view: out^T (n x m) = b (n x k) @ a^T (k x m)
    cublas.gemmEx(h, cublas.CUBLAS_OP_T, cublas.CUBLAS_OP_N, n, m, k,
                  alpha.ctypes.data, b.data.ptr, 2, k, a.data.ptr, 2, k,
                  beta.ctypes.data, out.data.ptr, 0, n, 68, 99)   # CUDA_R_16F=2, CUDA_R_32F=0, COMPUTE_32F=68
    return out


def _pad8(n):
    return (n + 7) // 8 * 8


def _to16(x):
    s = float(abs(x).max())
    if s == 0:
        s = 1.0
    return (x * (1.0 / s)).astype(np.float16), s


# ------------------------------------------------------------------ exchange surrogate
class SurrogateK:
    '''One DF tensor of the operator alpha erf(w r)/r + hyb erfc(w r)/r in a (smaller) auxiliary basis.
    k_ref(dm): accurate K_s(D) through the regular DF-K code (the precision globals of the caller apply).
    set_ref(C): cache X_L = B_L C_ref for the increments.   k_inc(C): K_s(2CC^T) - K_s(2C_ref C_ref^T) in FP16.'''

    def __init__(self, mf, auxbasis, omega, alpha, hyb, mode='fp16'):
        from gpu4pyscf.df import df as gdf
        self.mol = mf.mol
        self.dfs = gdf.DF(mf.mol, auxbasis=auxbasis)
        self.kw = dict(omega=omega, lr_factor=alpha, sr_factor=hyb)
        self.mode = mode
        self.B = None
        self.s1e = None

    def k_ref(self, dm):
        return self.dfs.get_jk(dm, hermi=1, with_j=False, with_k=True, **self.kw)[1]

    def _unpack(self):
        nao = self.mol.nao
        self.naop = naop = _pad8(nao)
        with self.dfs.range_coulomb(self.kw['omega'], self.kw['lr_factor'], self.kw['sr_factor']) as d:
            if d._cderi is None:
                d.build(**self.kw)
            naux = d._cderi[0].shape[0]
            dt = np.float16 if self.mode == 'fp16' else np.float32
            B = cupy.zeros((naux, nao, naop), dtype=dt)
            amax = 0.0
            for cderi, _ in d.loop(unpack=True):
                amax = max(amax, float(abs(cderi).max()))
            self.sB = amax if self.mode == 'fp16' else 1.0
            p0 = 0
            for cderi, _ in d.loop(unpack=True):
                n = len(cderi)
                B[p0:p0 + n, :, :nao] = (cderi * (1.0 / self.sB)).astype(dt)
                p0 += n
        self.naux = naux
        self.B = B
        self.B2 = B.reshape(naux * nao, naop)

    def _mm(self, a, b):
        '''a (m,k) @ b (n,k)^T with the surrogate precision; returns float32 (m, n) in true scale.'''
        if self.mode == 'fp16':
            ah, sa = (a, 1.0) if a.dtype == np.float16 else _to16(a)
            bh, sb = _to16(b)
            return gemm16_nt(ah, bh) * (sa * sb)
        return cupy.matmul(a.astype(np.float32, copy=False), b.astype(np.float32, copy=False).T)

    def ensure(self):
        if self.B is None:
            self._unpack()

    def k_abs(self, C):
        '''K_s(2 C C^T) directly in the surrogate precision (stage 1 only: no increment form).'''
        self.ensure()
        nao, nocc = C.shape
        noccp = _pad8(nocc)
        Ct = cupy.zeros((noccp, self.naop), dtype=np.float64)
        Ct[:nocc, :nao] = C.T
        X = self._mm(self.B2, Ct) * self.sB
        Xm = cupy.ascontiguousarray(X.reshape(self.naux, nao, noccp).transpose(1, 0, 2)).reshape(nao, -1)
        P = self._mm(Xm, Xm).astype(np.float64)
        return 2 * P

    def set_ref(self, Cref, s1e):
        self.ensure()
        nao, nocc = Cref.shape
        self.noccp = noccp = _pad8(nocc)
        self.nocc = nocc
        self.s1e = s1e
        self.Cref = Cref
        Ct = cupy.zeros((noccp, self.naop), dtype=np.float64)
        Ct[:nocc, :nao] = Cref.T
        X = self._mm(self.B2, Ct) * self.sB                                   # (naux*nao, noccp)
        self.Xm = cupy.ascontiguousarray(X.reshape(self.naux, nao, noccp).transpose(1, 0, 2)).reshape(nao, -1)

    def k_inc(self, C):
        nao, nocc = C.shape
        M = self.Cref.T @ (self.s1e @ C)
        U, _, Vt = cupy.linalg.svd(M)
        dC = C @ (Vt.T @ U.T) - self.Cref                                     # Procrustes-aligned increment
        Ct = cupy.zeros((self.noccp, self.naop), dtype=np.float64)
        Ct[:nocc, :nao] = dC.T
        Y = self._mm(self.B2, Ct) * self.sB
        Ym = cupy.ascontiguousarray(Y.reshape(self.naux, nao, self.noccp).transpose(1, 0, 2)).reshape(nao, -1)
        Z = self.Xm + 0.5 * Ym
        P = self._mm(Ym, Z)
        P = P.astype(np.float64)
        return 2 * (P + P.T)


# ------------------------------------------------------------------ eigensolver
def _oa_refine(A, X):
    '''Ogita & Aishima (2018) refinement of an approximate eigenvector matrix X of symmetric A (FP64 GEMMs).'''
    n = X.shape[1]
    R = cupy.eye(n) - X.T @ X
    S = X.T @ (A @ X)
    dR = cupy.diag(R)
    lam = cupy.diag(S) / (1 - dR)
    # cluster threshold of Ogita-Aishima: pairs closer than delta are treated as one (degenerate) cluster
    off = S - cupy.diag(cupy.diag(S))
    delta = 2 * (float(abs(off).max()) + float(abs(A).max()) * float(abs(R).max())) * n ** 0.5
    dl = lam[None, :] - lam[:, None]
    small = abs(dl) <= max(delta, 1e-12)
    E = (S + R * lam[None, :]) / cupy.where(small, 1.0, dl)
    E = cupy.where(small, 0.5 * R, E)
    idx = cupy.arange(n)
    E[idx, idx] = 0.5 * dR
    return X + X @ E, lam


class FastEig:
    def __init__(self, s1e, mode='fp32oa', nocc=None):
        w, U = cupy.linalg.eigh(s1e)
        keep = w > 1e-9 * float(w.max())
        self.X = U[:, keep] / cupy.sqrt(w[keep])
        self.mode = mode
        self.nocc = nocc

    def __call__(self, f):
        X = self.X
        A = X.T @ f @ X
        if self.mode == 'fp64':
            e, V = cupy.linalg.eigh(A)
            return e, X @ V
        w32, V32 = cupy.linalg.eigh(A.astype(np.float32))
        V = V32.astype(np.float64)
        for _ in range(2):
            V, lam = _oa_refine(A, V)
        order = cupy.argsort(lam)
        lam, V = lam[order], V[:, order]
        # orthonormalise (Cholesky QR, occupied columns first, so the occupied span is kept)
        L = cupy.linalg.cholesky(V.T @ V)
        V = cupy.linalg.solve(L, V.T).T
        return lam, X @ V


# ------------------------------------------------------------------ driver
def _grid(mol, atom_grid, like=None):
    from gpu4pyscf.dft import gen_grid
    g = gen_grid.Grids(mol)
    g.atom_grid = atom_grid
    if like is not None:
        for key in ('becke_scheme', 'prune', 'radi_method', 'radii_adjust', 'atomic_radii'):
            if hasattr(like, key):
                setattr(g, key, getattr(like, key))
    g.build(with_non0tab=False, sort_grids=True)
    return g


def _precision_ctx(mf, opt):
    from gpu4pyscf.scf import mixed_precision as mp
    if opt.get('precision'):
        return mp.fast_precision(opt['precision'])
    if getattr(mf, 'mixed_precision', None):
        return mp.final_stage(mf)
    import contextlib
    return contextlib.nullcontext()


def kernel(mf, dm0=None, **kwargs):
    opt = dict(DEFAULTS)
    cfg = getattr(mf, 'multifidelity', None)
    if isinstance(cfg, dict):
        opt.update(cfg)
    opt.update(kwargs)
    log = logger.new_logger(mf)
    tim = _Timer()
    t_start = time.perf_counter()
    mol = mf.mol
    ni = mf._numint
    xc = mf.xc
    assert mol.spin == 0, 'multi-fidelity SCF: closed shell only'
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(xc, spin=mol.spin)
    nocc = mol.nelectron // 2
    with_nlc = mf.do_nlc()
    nlc_xc = xc if ni.libxc.is_nlc(xc) else mf.nlc

    with tim('setup'):
        mf.build(mol)
        if mf.grids.coords is None:
            mf.grids.build(with_non0tab=False, sort_grids=True)
        if with_nlc and mf.nlcgrids.coords is None:
            mf.nlcgrids.build(with_non0tab=False, sort_grids=True)
        if ni.gdftopt is None:
            ni.build(mol, mf.grids.coords)
        g_in = _grid(mol, opt['inner_grid'], mf.grids)
        g_nlc1 = _grid(mol, opt['stage1_nlc_grid'], mf.nlcgrids) if with_nlc else None
        h1e = cupy.asarray(mf.get_hcore())
        s1e = cupy.asarray(mf.get_ovlp())
        e_nuc = mf.energy_nuc()
        eig = FastEig(s1e, opt['eig'], nocc)
        sk = SurrogateK(mf, opt['surrogate_aux'], omega, alpha, hyb,
                        mode=opt['k_increment'] if opt['k_increment'] != 'exact' else 'fp32')
    conv_tol_grad = opt['conv_tol_grad'] or 0.1 * np.sqrt(mf.conv_tol)

    def tagged(C_full, occ):
        D = 2 * C_full[:, occ > 0] @ C_full[:, occ > 0].T
        return tag_array(D, mo_coeff=C_full, mo_occ=occ)

    def solve(f):
        with tim('eig'):
            e, C = eig(f)
        occ = cupy.zeros(C.shape[1])
        occ[:nocc] = 2
        return e, C, occ

    def orb_grad(f, C):
        return 2 * float(cupy.linalg.norm(C[:, nocc:].T @ f @ C[:, :nocc]))

    def vxc_coarse(dm, nlc=False):
        with tim('xc_coarse'):
            v = ni.nr_rks(mol, g_in, xc, dm)[2]
        if nlc:
            with tim('nlc_coarse'):
                v = v + ni.nr_nlc_vxc(mol, g_nlc1, nlc_xc, dm)[2]
        return v

    def get_j(dm):
        with tim('J'):
            return mf.get_j(mol, dm, hermi=1)

    from gpu4pyscf.scf.diis import CDIIS

    def new_diis():
        d = CDIIS(mf)
        d.space = mf.diis_space
        d.Corth = eig.X
        return d

    ncyc = dict(stage1=0, inner=0, outer=0)
    with _precision_ctx(mf, opt):
        # ---------------------------------------------------------------- stage 1: everything cheap
        if dm0 is None:
            with tim('init_guess'):
                dm0 = mf.get_init_guess(mol, mf.init_guess)
        dm = cupy.asarray(dm0)
        if dm.ndim == 3:
            dm = dm[0] + dm[1]
        with tim('DF build'):
            if getattr(mf.with_df, '_cderi', None) is None:
                mf.with_df.build()                 # exact tensor first: J then takes the fast path
        diis = new_diis()
        f = None
        for it in range(opt['max_stage1']):
            vj = get_j(dm)
            with tim('K_s stage1'):
                if it == 0 or opt['k_increment'] == 'exact':
                    vk = sk.k_ref(dm)
                else:
                    vk = sk.k_abs(C[:, :nocc])
            f = h1e + vj + vxc_coarse(dm, nlc=with_nlc) - 0.5 * vk
            if it > 0:
                g = orb_grad(f, C)
                if g < opt['stage1_conv_grad']:
                    break
            with tim('diis'):
                fd = diis.update(s1e, dm, f)
            e, C, occ = solve(fd)
            dm = tagged(C, occ)
            ncyc['stage1'] += 1
        log.info('multi-fidelity: stage 1 %d cycles, |g| = %.2e', ncyc['stage1'], g)

        # ---------------------------------------------------------------- outer/inner
        hist = []
        e_tot = None
        mf._mf_xc_grad = None
        last_inner_tol = None
        xctype = ni._xc_type(xc)
        for outer in range(opt['max_outer']):
            ncyc['outer'] += 1
            Co = C[:, :nocc]
            want_grad = (opt['fuse_grad'] and opt['fine_vco'] and xctype in ('GGA', 'MGGA')
                         and last_inner_tol is not None and last_inner_tol < conv_tol_grad)
            # exact operator on the occupied space
            with tim('xc_fine'):
                if opt['fine_vco']:
                    from gpu4pyscf.dft.numint import nr_rks_vco
                    res = nr_rks_vco(ni, mol, mf.grids, xc, C, occ, with_grad=want_grad)
                    _, exc, Wxc = res[:3]
                    if want_grad:
                        mf._mf_xc_grad = (res[3], dm.copy())
                else:
                    _, exc, vf = ni.nr_rks(mol, mf.grids, xc, dm)
                    Wxc = vf @ Co
            enlc = 0.0
            if with_nlc:
                with tim('nlc_fine'):
                    _, enlc, vnlc = ni.nr_nlc_vxc(mol, mf.nlcgrids, nlc_xc, dm)
                Wxc = Wxc + vnlc @ Co
                vnlc = None
            with tim('K exact'):
                vk = hyb * mf.get_k(mol, dm, hermi=1)
                if omega != 0:
                    vk += (alpha - hyb) * mf.get_k(mol, dm, hermi=1, omega=omega)
                Wk = vk @ Co
                vk = None
            vj = get_j(dm)
            hc = h1e + vj
            Wt = hc @ Co + Wxc - 0.5 * Wk
            # energy of this density with the exact operator
            e1 = float(cupy.einsum('ij,ji->', dm, h1e))
            ecoul = 0.5 * float(cupy.einsum('ij,ji->', dm, vj))
            ek = -0.5 * float(cupy.einsum('ia,ia->', Co, Wk))      # -1/4 tr(D K), D = 2 C C^T
            e_tot = e1 + ecoul + exc + enlc + ek + e_nuc
            gt = 2 * float(cupy.linalg.norm(C[:, nocc:].T @ Wt))
            hist.append((gt, e_tot))
            log.info('multi-fidelity: outer %d  E = %.12f  |g| = %.3e', outer, e_tot, gt)
            if gt < conv_tol_grad:
                break
            # cheap operator at D_ref and the occupied-projected defect
            with tim('K_s exact'):
                vks_ref = sk.k_ref(dm)
            vc = vxc_coarse(dm)
            W = Wt - (hc + vc - 0.5 * vks_ref) @ Co
            SC = s1e @ Co
            Delta = W @ SC.T + SC @ W.T - SC @ (Co.T @ W) @ SC.T
            if opt['k_increment'] != 'exact':
                with tim('K_s set_ref'):
                    sk.set_ref(Co, s1e)
            f = hc + vc - 0.5 * vks_ref + Delta
            Dref_tag = dm
            diis = new_diis()
            inner_tol = max(opt['inner_factor'] * gt, 0.3 * conv_tol_grad * opt['inner_factor'])
            last_inner_tol = inner_tol
            for it in range(opt['max_inner']):
                with tim('diis'):
                    fd = diis.update(s1e, dm, f)
                e, C, occ = solve(fd)
                dm = tagged(C, occ)
                ncyc['inner'] += 1
                with tim('K_s inc'):
                    if opt['k_increment'] == 'exact':
                        vks = sk.k_ref(dm)
                    else:
                        vks = vks_ref + sk.k_inc(C[:, :nocc])
                f = h1e + get_j(dm) + vxc_coarse(dm) - 0.5 * vks + Delta
                gs = orb_grad(f, C)
                if gs < inner_tol:
                    break
            log.info('    inner cycles %d  |g_s| = %.2e', it + 1, gs)

    # canonical occupied orbitals of the exact operator; virtuals from the surrogate
    Co = C[:, :nocc]
    Mo = Co.T @ Wt
    eo, Uo = cupy.linalg.eigh(0.5 * (Mo + Mo.T))
    C = C.copy()
    C[:, :nocc] = Co @ Uo
    mo_energy = e.copy()
    mo_energy[:nocc] = eo
    mf.mo_coeff, mf.mo_occ, mf.mo_energy = C, occ, mo_energy
    mf.e_tot = e_tot
    mf.converged = hist[-1][0] < conv_tol_grad
    mf.multifidelity_stats = dict(cycles=ncyc, outer_hist=hist, timers=tim.t,
                                  wall=time.perf_counter() - t_start, conv_tol_grad=conv_tol_grad)
    mf.cycles = ncyc['inner'] + ncyc['stage1']
    log.note('multi-fidelity SCF: E = %.12f  converged = %s  outer %d inner %d stage1 %d',
             e_tot, mf.converged, ncyc['outer'], ncyc['inner'], ncyc['stage1'])
    return e_tot
