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
    coarse_xc='inc',              # 'inc': cached-AO FP32 increments of the coarse XC (stage 1: FP32 absolute); 'exact'
    grad_fit_reuse=True,          # precompute J^-1 (P|ij) for the DF gradient from the SCF tensors
    eig='fp64',                   # 'fp64' | 'fp32oa' (FP32 syevd + Ogita-Aishima; slower than syevd at nao~600 on T4)
    # precision follows the residual: an outer step whose predicted residual is far above the tolerance only needs a
    # proportionally accurate exact operator (inexact defect correction); the certifying last step is always exact
    schedule=False,
    cheap_grid=(75, 302),         # 'cheap' outer level: XC grid, VV10 grid, INT8 moduli of XC and K
    cheap_nlc_grid=(30, 110),
    cheap_moduli=9,
    medium_moduli=9,              # 'medium' level: fine grids, fewer moduli
    cheap_factor=100.,            # level from the predicted residual est = inner_factor * |g| of the previous outer:
    medium_factor=3.,             # est > cheap_factor * tol -> cheap, > medium_factor * tol -> medium, else exact
    exact_k='stored',             # 'stored': DF tensors in memory; 'direct': two streamed integral passes per outer
    direct_nmod=(15, 11),         # INT8-CRT moduli of the direct passes (pass 1 is amplified by J^-1); None: FP64
    inner_eig='full',             # 'full': diagonalise every inner cycle; 'rotate': preconditioned occupied-subspace
    rotate_steps=2,               # rotation (pseudo-diagonalisation, no nao^3 step) with the stage-1 virtuals as preconditioner
    precision=None,               # dict for mixed_precision.fast_precision; default from mf.mixed_precision / GPU
)


def _sync():
    cupy.cuda.Stream.null.synchronize()


_NVTX = bool(int(__import__('os').environ.get('GPU4PYSCF_MF_NVTX', '0')))


class _Timer:
    def __init__(self):
        self.t = {}

    def __call__(self, key):
        tim = self

        class _Ctx:
            def __enter__(self):
                _sync()
                if _NVTX:
                    cupy.cuda.nvtx.RangePush(key)
                self.t0 = time.perf_counter()

            def __exit__(self, *a):
                _sync()
                if _NVTX:
                    cupy.cuda.nvtx.RangePop()
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
        self._built = False

    def k_ref(self, dm):
        if not self._built:
            # the surrogate tensor is fixed for the whole SCF, so its build precision does not affect exactness
            # (any fixed B gives a smooth linear operator): build it with fewer INT8 moduli than the exact tensor
            from gpu4pyscf.df import df as gdf_mod
            old = gdf_mod.CDERI_CRT_MODULI
            gdf_mod.CDERI_CRT_MODULI = min(old, 8)
            try:
                with self.dfs.range_coulomb(self.kw['omega'], self.kw['lr_factor'], self.kw['sr_factor']) as d:
                    if d._cderi is None:
                        d.build(**self.kw)
            finally:
                gdf_mod.CDERI_CRT_MODULI = old
            self._built = True
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
        return self.k_inc_dC(dC)

    def k_inc_dC(self, dC):
        nao, nocc = dC.shape
        Ct = cupy.zeros((self.noccp, self.naop), dtype=np.float64)
        Ct[:nocc, :nao] = dC.T
        Y = self._mm(self.B2, Ct) * self.sB
        Ym = cupy.ascontiguousarray(Y.reshape(self.naux, nao, self.noccp).transpose(1, 0, 2)).reshape(nao, -1)
        Z = self.Xm + 0.5 * Ym
        P = self._mm(Ym, Z)
        P = P.astype(np.float64)
        return 2 * (P + P.T)


def align_increment(Cref, C, s1e):
    """dC = C R - Cref with the rotation R of the occupied space that brings C closest to Cref (Procrustes)."""
    M = Cref.T @ (s1e @ C)
    U, _, Vt = cupy.linalg.svd(M)
    return C @ (Vt.T @ U.T) - Cref


class CoarseXC:
    """Semilocal XC on the inner (coarse) grid from cached FP32 AO values.
    absolute(C): V_c(2CC^T) in FP32 (stage 1).   set_ref(C_ref) + inc(dC): V_c(D) - V_c(D_ref) with the
    density increment rho = rho_ref + delta rho (delta psi = A^T dC in FP32, psi_ref in FP64), libxc in FP64 and
    delta V = Phi^T diag(u - u_ref) Phi in FP32: the rounding error scales with the increment."""

    def __init__(self, ni, mol, grid, xc):
        self.ni, self.mol, self.grid, self.xc = ni, mol, grid, xc
        self.xctype = ni._xc_type(xc)
        assert self.xctype in ('GGA', 'MGGA')
        self.opt = opt = ni.gdftopt
        self.nao = mol.nao
        self.blocks = []
        p0 = 0
        for ao, idx, weight, _ in ni.block_loop(opt._sorted_mol, grid, self.nao, 1, max_memory=None):
            self.blocks.append((ao[:4].astype(np.float32), idx, p0, weight.size))
            p0 += weight.size
        self.ng = p0
        self.w = cupy.asarray(grid.weights)

    def _sorted(self, C):
        return self.opt.sort_orbitals(C, axis=[0]) * np.sqrt(2.0)

    def _psi(self, Cs, dtype):
        nocc = Cs.shape[1]
        psi = cupy.empty((4, self.ng, nocc), dtype=dtype)
        for A, idx, p0, ng in self.blocks:
            Cb = cupy.ascontiguousarray(Cs[idx]).astype(dtype)
            for k in range(4):
                Ak = A[k] if dtype == np.float32 else A[k].astype(np.float64)
                cupy.matmul(Ak.T, Cb, out=psi[k, p0:p0 + ng])
        return psi

    def _u(self, psi):
        from gpu4pyscf.lib import crt_gemm
        rho = crt_gemm.rho_from_cT(cupy.ascontiguousarray(psi, dtype=np.float64), self.ng, self.xctype)
        exc, vxc = self.ni.eval_xc_eff(self.xc, rho, deriv=1, xctype=self.xctype, spin=0)[:2]
        return cupy.asarray(vxc, order='C') * self.w

    def _vmat(self, u):
        from gpu4pyscf.dft.numint import add_sparse
        V = cupy.zeros((self.nao, self.nao))
        u32 = u.astype(np.float32)
        mgga = self.xctype == 'MGGA'
        for A, idx, p0, ng in self.blocks:
            ub = u32[:, p0:p0 + ng]
            B0 = 0.5 * ub[0] * A[0] + ub[1] * A[1] + ub[2] * A[2] + ub[3] * A[3]
            H = A[0] @ B0.T
            if mgga:
                q = 0.25 * ub[4]
                for x in range(1, 4):
                    H += A[x] @ (q * A[x]).T
            H = H.astype(np.float64)
            add_sparse(V, H + H.T, idx)
        return self.opt.unsort_orbitals(V, axis=[0, 1])

    def absolute(self, C):
        return self._vmat(self._u(self._psi(self._sorted(C), np.float32)))

    def set_ref(self, Cref):
        self.psi_ref = self._psi(self._sorted(Cref), np.float64)
        self.u_ref = self._u(self.psi_ref)

    def inc(self, dC):
        psi = self.psi_ref + self._psi(self._sorted(dC), np.float32)
        return self._vmat(self._u(psi) - self.u_ref)


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


class _moduli_ctx:
    '''Temporarily lower the INT8-CRT moduli of the XC and DF-K GEMMs (no-op for None or non-crt paths).'''
    def __init__(self, n):
        self.n = n

    def __enter__(self):
        if self.n is None:
            return
        from gpu4pyscf.dft import numint
        from gpu4pyscf.df import df_jk
        self.old = numint.XC_CRT_MODULI, df_jk.K_CRT_MODULI
        numint.XC_CRT_MODULI = min(self.n, numint.XC_CRT_MODULI)
        df_jk.K_CRT_MODULI = min(self.n, df_jk.K_CRT_MODULI)

    def __exit__(self, *a):
        if self.n is None:
            return
        from gpu4pyscf.dft import numint
        from gpu4pyscf.df import df_jk
        numint.XC_CRT_MODULI, df_jk.K_CRT_MODULI = self.old


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
        dk_fr = dk_lr = None
        if opt['exact_k'] == 'direct':
            from gpu4pyscf.df.direct_k import DirectDFK
            if getattr(mf.with_df, 'auxmol', None) is None:
                mf.with_df.build(build_cderi=False) if 'build_cderi' in mf.with_df.build.__code__.co_varnames \
                    else mf.with_df.build()
            nm = opt['direct_nmod'] or (None, None)
            dk_fr = DirectDFK(mol, mf.with_df.auxmol, None, nmod_t=nm[0], nmod_x=nm[1])
            if omega != 0:
                dk_lr = DirectDFK(mol, mf.with_df.auxmol, omega, nmod_t=nm[0], nmod_x=nm[1])
        g_cheap = g_nlc_cheap = None
        if opt['schedule']:
            g_cheap = _grid(mol, opt['cheap_grid'], mf.grids)
            g_nlc_cheap = _grid(mol, opt['cheap_nlc_grid'], mf.nlcgrids) if with_nlc else None
        h1e = cupy.asarray(mf.get_hcore())
        s1e = cupy.asarray(mf.get_ovlp())
        e_nuc = mf.energy_nuc()
        eig = FastEig(s1e, opt['eig'], nocc)
        cx = CoarseXC(ni, mol, g_in, xc) if opt['coarse_xc'] == 'inc' else None
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

    Sinv = eig.X @ eig.X.T if opt['inner_eig'] == 'rotate' else None

    def proj_grad(X_, Co_):
        '''2 |C_v^T X| for the orthonormal complement of the S-orthonormal Co_ (no virtual orbitals needed)'''
        t = float(cupy.einsum('ia,ia->', X_, Sinv @ X_)) - float(cupy.linalg.norm(Co_.T @ X_)) ** 2
        return 2 * np.sqrt(max(t, 0.0))

    def rotate(fd, C, e):
        '''occupied-subspace update without diagonalisation: canonicalise in the occupied block, first-order
        rotation into the S-projected reference virtuals (preconditioner: reference orbital energies), Loewdin.'''
        Co_, Cv0, ev = C[:, :nocc], C[:, nocc:], e[nocc:]
        SCv0 = s1e @ Cv0
        for _ in range(opt['rotate_steps']):
            G = fd @ Co_
            eo_, U = cupy.linalg.eigh(0.5 * (Co_.T @ G + G.T @ Co_))
            Co_ = Co_ @ U
            G = G @ U
            Cv = Cv0 - Co_ @ (Co_.T @ SCv0)
            kappa = (Cv.T @ G) / (eo_[None, :] - ev[:, None])
            Co_ = Co_ + Cv @ kappa
            w_, V_ = cupy.linalg.eigh(Co_.T @ s1e @ Co_)
            Co_ = Co_ @ ((V_ / cupy.sqrt(w_)) @ V_.T)
        G = fd @ Co_
        eo_, U = cupy.linalg.eigh(0.5 * (Co_.T @ G + G.T @ Co_))
        Co_ = Co_ @ U
        Cn = C.copy()
        Cn[:, :nocc] = Co_
        en = e.copy()
        en[:nocc] = eo_
        return en, Cn

    def orb_grad(f, C):
        if Sinv is not None:
            return proj_grad(f @ C[:, :nocc], C[:, :nocc])
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
            if it == 0 or cx is None:
                vc1 = vxc_coarse(dm, nlc=with_nlc)
            else:
                with tim('xc_coarse32'):
                    vc1 = cx.absolute(C[:, :nocc])
                if with_nlc:
                    with tim('nlc_coarse'):
                        vc1 = vc1 + ni.nr_nlc_vxc(mol, g_nlc1, nlc_xc, dm)[2]
            f = h1e + vj + vc1 - 0.5 * vk
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
        levels = []
        for outer in range(opt['max_outer']):
            ncyc['outer'] += 1
            Co = C[:, :nocc]
            level = 'exact'
            if opt['schedule'] and not (levels and levels[-1] != 'exact' and hist[-1][0] < conv_tol_grad):
                if outer == 0:
                    level = 'cheap'
                else:
                    est = opt['inner_factor'] * hist[-1][0]
                    if est > opt['cheap_factor'] * conv_tol_grad:
                        level = 'cheap'
                    elif est > opt['medium_factor'] * conv_tol_grad:
                        level = 'medium'
            levels.append(level)
            grids_x = g_cheap if level == 'cheap' else mf.grids
            grids_nlc = g_nlc_cheap if level == 'cheap' else mf.nlcgrids
            want_grad = (level == 'exact' and opt['fuse_grad'] and opt['fine_vco'] and xctype in ('GGA', 'MGGA')
                         and last_inner_tol is not None and last_inner_tol < conv_tol_grad)
            lvl_ctx = _moduli_ctx(opt['cheap_moduli'] if level == 'cheap' else
                                  opt['medium_moduli'] if level == 'medium' else None)
            # exact operator on the occupied space
            with lvl_ctx, tim(('xc_fine+grad' if want_grad else 'xc_fine') + ('' if level == 'exact' else '_' + level)):
                if opt['fine_vco']:
                    from gpu4pyscf.dft.numint import nr_rks_vco
                    res = nr_rks_vco(ni, mol, grids_x, xc, C, occ, with_grad=want_grad)
                    _, exc, Wxc = res[:3]
                    if want_grad:
                        mf._mf_xc_grad = (res[3], dm.copy())
                else:
                    _, exc, vf = ni.nr_rks(mol, grids_x, xc, dm)
                    Wxc = vf @ Co
            enlc = 0.0
            if with_nlc:
                with lvl_ctx, tim('nlc_fine' + ('' if level == 'exact' else '_' + level)):
                    _, enlc, vnlc = ni.nr_nlc_vxc(mol, grids_nlc, nlc_xc, dm)
                Wxc = Wxc + vnlc @ Co
                vnlc = None
            vj = None
            with lvl_ctx, tim('K exact' + ('' if level == 'exact' else '_' + level)):
                if opt['exact_k'] == 'direct':
                    Cw = Co * np.sqrt(2.0)
                    kc, vj = dk_fr.kc_j(Cw, with_j=True)
                    Wk = hyb * kc
                    if omega != 0:
                        Wk += (alpha - hyb) * dk_lr.kc_j(Cw, with_j=False)[0]
                    Wk /= np.sqrt(2.0)
                else:
                    vk = hyb * mf.get_k(mol, dm, hermi=1)
                    if omega != 0:
                        vk += (alpha - hyb) * mf.get_k(mol, dm, hermi=1, omega=omega)
                    Wk = vk @ Co
                    vk = None
            if vj is None:
                vj = get_j(dm)
            hc = h1e + vj
            Wt = hc @ Co + Wxc - 0.5 * Wk
            # energy of this density with the exact operator
            e1 = float(cupy.einsum('ij,ji->', dm, h1e))
            ecoul = 0.5 * float(cupy.einsum('ij,ji->', dm, vj))
            ek = -0.5 * float(cupy.einsum('ia,ia->', Co, Wk))      # -1/4 tr(D K), D = 2 C C^T
            e_tot = e1 + ecoul + exc + enlc + ek + e_nuc
            gt = proj_grad(Wt, Co) if Sinv is not None else 2 * float(cupy.linalg.norm(C[:, nocc:].T @ Wt))
            hist.append((gt, e_tot))
            log.info('multi-fidelity: outer %d (%s)  E = %.12f  |g| = %.3e', outer, level, e_tot, gt)
            if gt < conv_tol_grad and level == 'exact':
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
            if cx is not None:
                with tim('xc_coarse set_ref'):
                    cx.set_ref(Co)
            Co_ref = Co
            f = hc + vc - 0.5 * vks_ref + Delta
            Dref_tag = dm
            diis = new_diis()
            inner_tol = max(opt['inner_factor'] * gt, 0.3 * conv_tol_grad * opt['inner_factor'])
            last_inner_tol = inner_tol
            for it in range(opt['max_inner']):
                with tim('diis'):
                    fd = diis.update(s1e, dm, f)
                if opt['inner_eig'] == 'rotate':
                    with tim('eig rotate'):
                        e, C = rotate(fd, C, e)
                else:
                    e, C, occ = solve(fd)
                dm = tagged(C, occ)
                ncyc['inner'] += 1
                dC = align_increment(Co_ref, C[:, :nocc], s1e)
                with tim('K_s inc'):
                    if opt['k_increment'] == 'exact':
                        vks = sk.k_ref(dm)
                    else:
                        vks = vks_ref + sk.k_inc_dC(dC)
                if cx is not None:
                    with tim('xc_coarse inc'):
                        vci = vc + cx.inc(dC)
                else:
                    vci = vxc_coarse(dm)
                f = h1e + get_j(dm) + vci - 0.5 * vks + Delta
                gs = orb_grad(f, C)
                if gs < inner_tol:
                    break
            log.info('    inner cycles %d  |g_s| = %.2e', it + 1, gs)

    # canonical occupied orbitals of the exact operator; virtuals from the surrogate
    Co = C[:, :nocc]
    if Sinv is not None:
        # orthonormal virtual complement of the final occupied space, canonical for the last surrogate operator
        Cv = C[:, nocc:] - Co @ (Co.T @ (s1e @ C[:, nocc:]))
        w_, V_ = cupy.linalg.eigh(Cv.T @ s1e @ Cv)
        keep = w_ > 1e-10 * float(w_.max())
        Cv = Cv @ (V_[:, keep] / cupy.sqrt(w_[keep]))
        ev, Uv = cupy.linalg.eigh(Cv.T @ f @ Cv)
        Cv = Cv @ Uv
        C = cupy.hstack([Co, Cv])
        e = cupy.concatenate([e[:nocc], ev])
        occ = cupy.zeros(C.shape[1])
        occ[:nocc] = 2
    Mo = Co.T @ Wt
    eo, Uo = cupy.linalg.eigh(0.5 * (Mo + Mo.T))
    C = C.copy()
    C[:, :nocc] = Co @ Uo
    mo_energy = e.copy()
    mo_energy[:nocc] = eo
    mf.mo_coeff, mf.mo_occ, mf.mo_energy = C, occ, mo_energy
    mf.e_tot = e_tot
    from gpu4pyscf.df.grad import rhf as df_rhf_grad
    df_rhf_grad.FIT_OO_CACHE = None
    if opt['grad_fit_reuse'] and hist[-1][0] < conv_tol_grad and levels[-1] == 'exact':
        # occupied-pair fitting coefficients for the DF J/K gradient from the DF tensors still in memory (the gradient
        # frees them on entry): saves its raw 3c recomputation and metric solve
        with tim('grad fit_oo'):
            Cw = C[:, :nocc] * np.sqrt(2.0)
            cache = {'C': Cw, 'omega': abs(omega), 'fr': df_rhf_grad.precompute_fit_oo(mf.with_df, Cw)}
            if omega != 0:
                with mf.with_df.range_coulomb(omega, None, None) as d:
                    cache['lr'] = df_rhf_grad.precompute_fit_oo(d, Cw, omega=omega)
            df_rhf_grad.FIT_OO_CACHE = cache
    mf.converged = hist[-1][0] < conv_tol_grad and levels[-1] == 'exact'
    mf.multifidelity_stats = dict(cycles=ncyc, outer_hist=hist, timers=tim.t, levels=levels,
                                  wall=time.perf_counter() - t_start, conv_tol_grad=conv_tol_grad)
    mf.cycles = ncyc['inner'] + ncyc['stage1']
    log.note('multi-fidelity SCF: E = %.12f  converged = %s  outer %d inner %d stage1 %d',
             e_tot, mf.converged, ncyc['outer'], ncyc['inner'], ncyc['stage1'])
    return e_tot
