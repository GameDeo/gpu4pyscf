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
    stage1_nlc_grid=(10, 26),     # VV10 grid of the first (cheap) stage and of the inner-loop VV10 increments
    inner_nlc_grid=None,          # VV10 grid of the inner-loop increments (None: stage1_nlc_grid)
    inner_nlc=True,               # inner loop: VV10 response as coarse-grid increments V(D) - V(D_ref) (False: VV10
                                  # frozen inside Delta~; that response alone was |g| 1.6e-5 of the 1.9e-5 residual
                                  # after the first outer step on BPA wB97M-V)
    surrogate_aux='def2-svp-ri',  # aux basis of the exchange surrogate
    k_increment='fp32',           # 'fp32' | 'fp16' | 'exact' (exact = surrogate DF-K at every inner step)
                                  # fp16 noise (~1e-3 of K(dD)) stalls the inner loop for BPA-size systems
    stage1_conv_grad=1e-3,
    inner_factor=1e-3,            # inner tolerance = inner_factor * |g_true| of the last outer step (BPA: the surrogate
                                  # contracts |g| by ~400 per outer, so 1e-3 lets 2 exact evaluations reach 3e-5)
    conv_tol_grad=None,           # outer tolerance on |g| (pyscf orbital-gradient norm); default sqrt(conv_tol), pyscf's
                                  # own default (BPA: dE 2e-10, max|dg| 4.6e-7 vs 4e-11 / 4.2e-7 for the stock FP64 SCF)
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
    # Quantised (INT8/INT16) surrogate tensors carry rounding noise that is unstructured in the AO basis; the
    # orthogonalisation amplifies it by 1/lambda_min(S) (7e-6 on BPA/def2-TZVP, 3e-6 on Taxol) along near-linearly
    # dependent AO combinations, where exact operators are physically small.  They are therefore only used for
    # *increments* (noise ~ |D - D_ref|), through the spectral filter inc_filter (surrogate increments projected onto
    # the S eigenvectors with lambda > inc_filter); every absolute term stays exact or FP32.
    contraction_guess=400.,       # assumed |g| contraction of the first outer step (XC-gradient fusion prediction)
    inc_store=None,               # None: dense FP32 surrogate K increments + exact J | 'int8': INT8 K / INT16 J increments
    inc_filter=0.0,               # lambda threshold of the filter (0: off)
    ref_k='surrogate',            # K matrix at D_ref: 'surrogate' (dense K_s(D_ref)) | 'ace' (K C (C^T K C)^-1 (K C)^T
                                  # from the exact K C_occ of the checkpoint: no K tensor needed)
    surrogate_j_aux='def2-universal-jfit',
    surrogate_host='auto',        # stored surrogate tensors in pinned host memory: True | False | 'auto' (when a tensor
                                  # would take more than half of the free GPU memory); streamed per evaluation
    coarse_cache_frac=None,       # GPU-memory fraction for the FP32 coarse-grid AO cache (None: no limit); beyond it
                                  # blocks are stored FP16 (rounding of the AO values themselves is structured)
    occ_k=True,                   # stored exact K as K C_occ (X = B C, T = C^T X, KC = X T); T reused for the gradient fit
    exact_k='auto',               # 'stored': DF tensors in memory; 'direct': two streamed integral passes per outer;
                                  # 'cached': direct passes with a device cache of the raw integrals (fills the budget);
                                  # 'auto': direct when the stored FP64 tensors would exceed 45 % of the GPU memory
    direct_nmod=(15, 11),         # INT8-CRT moduli of the direct passes (pass 1 is amplified by J^-1); None: FP64
    inner_eig='auto',             # 'full': diagonalise every inner cycle; 'rotate': preconditioned occupied-subspace
    rotate_steps=2,               # rotation (pseudo-diagonalisation, no nao^3 step) with the stage-1 virtuals as preconditioner;
    rotate_nao=1500,              # 'auto': rotate when nao > rotate_nao (BPA, nao 623: only 0.2 s gain, kept on 'full')
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

    def _build(self):
        if self._built:
            return
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

    def k_ref(self, dm):
        self._build()
        return self.dfs.get_jk(dm, hermi=1, with_j=False, with_k=True, **self.kw)[1]

    def k_dm(self, dm, s1e):
        '''K_s(dm) for a general symmetric density (e.g. the initial guess) through its natural orbitals:
        dm = N diag(n) N^T, K_s(dm) = k_abs(N sqrt(n / 2)) (natural occupations below 1e-10 are dropped, which only
        perturbs the stage-1 start).'''
        w, U = cupy.linalg.eigh(s1e)
        X = U / cupy.sqrt(w)
        SX = s1e @ X
        n, V = cupy.linalg.eigh(SX.T @ dm @ SX)
        keep = n > 1e-10
        Cn = (X @ V[:, keep]) * cupy.sqrt(n[keep] / 2)
        return self.k_abs(Cn)

    def _unpack(self):
        '''Dense (naux, nao, naop) surrogate tensor, scattered straight from the compressed pair storage (one pass;
        no FP64 (nao, nao, blk) unpack + strided transposed copy).'''
        nao = self.mol.nao
        self.naop = naop = _pad8(nao)
        self._build()
        with self.dfs.range_coulomb(self.kw['omega'], self.kw['lr_factor'], self.kw['sr_factor']) as d:
            cs = d._cderi[cupy.cuda.Device().id]
            naux, npair = cs.shape
            pair = cupy.asarray(d._cderi_idx[0], dtype=cupy.int64)
            i_, j_ = pair // nao, pair % nao
            idx_ij, idx_ji = i_ * naop + j_, j_ * naop + i_
            dt = np.float16 if self.mode == 'fp16' else np.float32
            blk = max(1, min(naux, int(2e8 // (12 * max(npair, 1)))))
            if self.mode == 'fp16':
                amax = 0.0
                for p0 in range(0, naux, blk):
                    amax = max(amax, float(abs(cupy.asarray(cs[p0:p0 + blk])).max()))
                self.sB = amax
            else:
                self.sB = 1.0
            B = cupy.zeros((naux, nao * naop), dtype=dt)
            for p0 in range(0, naux, blk):
                p1 = min(naux, p0 + blk)
                v = cupy.asarray(cs[p0:p1])
                v = (v * (1.0 / self.sB)).astype(dt) if self.sB != 1.0 else v.astype(dt)
                B[p0:p1, idx_ji] = v
                B[p0:p1, idx_ij] = v
                v = None
        self.naux = naux
        self.B = B.reshape(naux, nao, naop)
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


class _NullK:
    '''Surrogate for functionals without exact exchange: every K term is zero (the multi-fidelity SCF then stages
    only the XC grids and J).'''

    def __init__(self, nao):
        self.nao = nao

    def _z(self):
        return cupy.zeros((self.nao, self.nao))

    def k_dm(self, dm, s1e):
        return self._z()

    def k_ref(self, dm):
        return self._z()

    def k_abs(self, C):
        return self._z()

    def ensure(self):
        pass

    def set_ref(self, Cref, s1e):
        pass

    def k_inc_dC(self, dC):
        return self._z()


class SurrogateK8:
    """SurrogateK on an INT8-stored tensor (df.int8_df.Int8DF): same interface, memory ~ npairs * naux bytes."""

    def __init__(self, mf, auxbasis, omega, alpha, hyb, bits=8, host='auto'):
        from gpu4pyscf.df.int8_df import Int8DF
        self.t = Int8DF(mf.mol, auxbasis, omega=omega if omega else None,
                        lr_factor=alpha if omega else None, sr_factor=hyb if omega else None, bits=bits, host=host)
        self.hyb = 1.0 if omega else hyb
        self._built = False

    def _ensure(self):
        if not self._built:
            self.t.build(mem_frac=0.1)   # after the exact tensor: CRT build temporaries stay small
            self._built = True

    def k_ref(self, dm):
        self._ensure()
        if getattr(dm, 'mo_coeff', None) is not None:
            occ = cupy.asarray(dm.mo_occ)
            C = cupy.asarray(dm.mo_coeff)[:, occ > 0] * cupy.sqrt(occ[occ > 0])
        else:
            w, v = cupy.linalg.eigh(cupy.asarray(dm))
            keep = w > 1e-8
            C = v[:, keep] * cupy.sqrt(w[keep])
        return self.hyb * self.t.k_pairs(C)

    def ensure(self):
        self._ensure()

    def k_abs(self, C):
        return 2 * self.hyb * self.t.k_pairs(C)

    def set_ref(self, Cref, s1e):
        self.Cref = Cref

    def k_inc_dC(self, dC):
        P = self.t.k_pairs(dC, self.Cref, 0.5)
        return 2 * self.hyb * (P + P.T)


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

    def __init__(self, ni, mol, grid, xc, cache_frac=None):
        self.ni, self.mol, self.grid, self.xc = ni, mol, grid, xc
        self.xctype = ni._xc_type(xc)
        assert self.xctype in ('GGA', 'MGGA')
        self.opt = opt = ni.gdftopt
        self.nao = mol.nao
        self.blocks = []
        p0 = 0
        # FP32 AO cache up to cache_frac of the GPU memory, FP16 beyond (large molecules); used as FP32 per block
        budget = (cache_frac or 1.0) * cupy.cuda.Device().mem_info[1]
        used = 0
        for ao, idx, weight, _ in ni.block_loop(opt._sorted_mol, grid, self.nao, 1, max_memory=None):
            dt = np.float32 if used + ao[:4].size * 4 <= budget else np.float16
            a = ao[:4].astype(dt)
            used += a.nbytes
            self.blocks.append((a, idx, p0, weight.size))
            p0 += weight.size
        self.cache_bytes = used
        self.ng = p0
        self.w = cupy.asarray(grid.weights)

    def _sorted(self, C):
        return self.opt.sort_orbitals(C, axis=[0]) * np.sqrt(2.0)

    def _psi(self, Cs, dtype):
        nocc = Cs.shape[1]
        psi = cupy.empty((4, self.ng, nocc), dtype=dtype)
        for A, idx, p0, ng in self.blocks:
            A = A.astype(np.float32, copy=False)
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
            A = A.astype(np.float32, copy=False)
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


def _proj_grad(X, Co, Sinv, s1e):
    '''2 |C_v^T X| for the orthonormal complement of the S-orthonormal Co (no virtual orbitals needed):
    R = X - S Co (Co^T X) is the complement part of X (formed first, so no cancellation between two large traces),
    |C_v^T X|^2 = tr(R^T S^-1 R).'''
    R = X - s1e @ (Co @ (Co.T @ X))
    return 2 * np.sqrt(max(float(cupy.einsum('ia,ia->', R, Sinv @ R)), 0.0))


def _rotate(fd, C, e, s1e, nocc, steps):
    '''Occupied-subspace update without diagonalisation: canonicalise in the occupied block, first-order rotation
    into the S-projected reference virtuals (preconditioner: reference orbital energies), Loewdin.'''
    Co, Cv0, ev = C[:, :nocc], C[:, nocc:], e[nocc:]
    SCv0 = s1e @ Cv0
    for _ in range(steps):
        G = fd @ Co
        eo, U = cupy.linalg.eigh(0.5 * (Co.T @ G + G.T @ Co))
        Co = Co @ U
        G = G @ U
        Cv = Cv0 - Co @ (Co.T @ SCv0)
        # guard: a stale reference virtual energy close to (or below) an occupied one would blow up the first-order step
        kappa = (Cv.T @ G) / cupy.minimum(eo[None, :] - ev[:, None], -0.05)
        Co = Co + Cv @ kappa
        w, V = cupy.linalg.eigh(Co.T @ s1e @ Co)
        Co = Co @ ((V / cupy.sqrt(w)) @ V.T)
    G = fd @ Co
    eo, U = cupy.linalg.eigh(0.5 * (Co.T @ G + G.T @ Co))
    Co = Co @ U
    Cn = C.copy()
    Cn[:, :nocc] = Co
    en = e.copy()
    en[:nocc] = eo
    return en, Cn


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


def _host_avail():
    '''available host memory in bytes (MemAvailable), inf when unknown'''
    try:
        with open('/proc/meminfo') as f:
            for ln in f:
                if ln.startswith('MemAvailable:'):
                    return float(ln.split()[1]) * 1024
    except OSError:
        pass
    return float('inf')


def _plan_storage(mf, opt, omega, free, no_k):
    '''Where the exact DF tensors go (sizes from the screened AO-pair count, the multi-fidelity tensors included).
    Greedy in build order -- FR exact (DF build), surrogates, LR exact (first checkpoint) -- on a device budget that
    keeps work space for XC / gradient, and a host budget below MemAvailable:
      all on device (dense FP32 surrogate)  >  all on device (stored FP32 surrogates)  >  LR in host memory
      >  FR + LR in host memory  >  'mixed' (FR stored, LR direct)  >  direct.'''
    from pyscf.df import addons
    df = mf.with_df
    mol = mf.mol
    nao = mol.nao
    if getattr(df, 'intopt', None) is None:
        df.build(build_cderi=False)            # pair screening only (no integrals)
    npairs = len(df.intopt.pair_and_diag_indices()[0])
    naux = df.auxmol.nao
    nks = addons.make_auxmol(mol, opt['surrogate_aux']).nao_nr()
    njs = addons.make_auxmol(mol, opt['surrogate_j_aux']).nao_nr()
    E = 8. * naux * npairs                     # one exact FP64 tensor
    nops = 2 if omega != 0 and not no_k else 1
    dense = 0. if no_k else 4. * nks * nao * nao + 8. * nks * npairs   # SurrogateK: FP64 DF + unpacked FP32
    f32 = 0. if no_k else 4. * npairs * (nks + njs)                     # stored FP32 K and J surrogates
    dev = free - max(4e9, 0.3 * free)         # work space: XC blocks, CRT build buffers, K loop, gradient
    host = 0.8 * _host_avail() - 1.5e9         # the process itself, pinned staging, gradient
    res = dict(mode='stored', f32=False, lr_host=False, lr_direct=False)
    if nops * E + dense <= dev:
        pass
    elif nops * E + f32 <= dev:
        res.update(f32=True)
    elif nops == 2 and E + f32 <= dev and E <= host:
        res.update(f32=True, lr_host=True)
    elif f32 <= dev and nops * E <= host:
        res.update(f32=True, lr_host=nops == 2)  # FR: the DF module places it (host when it does not fit)
    elif nops == 2 and opt['occ_k'] and (E + f32 <= dev or (f32 <= dev and E <= host)):
        res.update(f32=True, lr_direct=True)
    else:
        res.update(mode='direct')
    res['msg'] = ('%s (npairs %d naux %d: exact %.2f GB x %d, surrogates dense %.2f / f32 %.2f GB; budget device '
                  '%.1f GB host %.1f GB)%s%s' % (res['mode'] if not res['lr_direct'] else 'mixed FR-stored/LR-direct',
                                               npairs, naux, E / 1e9, nops, dense / 1e9, f32 / 1e9, dev / 1e9,
                                               host / 1e9, ' f32-surrogates' if res['f32'] else '',
                                               ' LR-on-host' if res['lr_host'] else ''))
    return res


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
    no_k = abs(hyb) < 1e-14 and abs(alpha) < 1e-14     # pure functional: no exchange operator at all
    if no_k:
        opt.update(inc_store=None, grad_fit_reuse=False, ref_k='none', exact_k='stored')
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
        g_nlc_in = (_grid(mol, opt['inner_nlc_grid'], mf.nlcgrids) if opt['inner_nlc_grid'] else g_nlc1) \
            if with_nlc else None
        dk_fr = dk_lr = None
        exact_on_host = False
        from gpu4pyscf.lib.cupy_helper import get_avail_mem as _gam
        free = _gam()                      # free device memory incl. the cached pool blocks
        lr_direct = False                  # 'mixed': FR tensor stored, LR operator direct
        cache_budget = None
        if opt['exact_k'] == 'cached':
            # one algorithm at every size: streamed direct passes whose raw 3c integrals are cached on the device up to
            # the budget left after the surrogates and the work space (no stored DF tensor, no host memory)
            opt['exact_k'] = 'direct'
            if not no_k:
                from pyscf.df import addons as _ad
                df_ = mf.with_df
                if getattr(df_, 'intopt', None) is None:
                    df_.build(build_cderi=False)
                npr = len(df_.intopt.pair_and_diag_indices()[0])
                sur = 4. * npr * (_ad.make_auxmol(mol, opt['surrogate_aux']).nao_nr() +
                                  _ad.make_auxmol(mol, opt['surrogate_j_aux']).nao_nr())
                cache_budget = max(0., free - max(4e9, 0.3 * free) - sur) / (2 if omega != 0 else 1)
        if opt['exact_k'] == 'auto':
            plan = _plan_storage(mf, opt, omega, free, no_k)
            opt['exact_k'], exact_on_host, lr_host, lr_direct = plan['mode'], plan['f32'], plan['lr_host'], \
                plan['lr_direct']
            log.info('multi-fidelity: storage plan %s', plan['msg'])
            if lr_host:
                # the LR tensor is built lazily at the first checkpoint, after the surrogates: keep it off the device
                # so that the XC / gradient work space stays (the DF module would take whatever is left)
                with mf.with_df.range_coulomb(omega, None, None) as d_lr:
                    d_lr.use_gpu_memory = False
        if (opt['exact_k'] == 'direct' or exact_on_host or lr_direct) and opt['inc_store'] is None and not no_k:
            # no exact DF tensor in memory: the inner-loop J must not fall back to mf.get_j (which streams the full
            # FP64 tensor from host memory every cycle): FP32 J and K surrogate tensors (pinned host memory when they
            # do not fit; the Taxol configuration), usable as absolute terms in stage 1 too
            opt['inc_store'] = 'f32'
        log.info('multi-fidelity: exact_k=%s inc_store=%s inc_filter=%g ref_k=%s',
                 opt['exact_k'], opt['inc_store'], opt['inc_filter'], opt['ref_k'])
        if opt['exact_k'] == 'direct' or lr_direct:
            from gpu4pyscf.df.direct_k import DirectDFK
            if getattr(mf.with_df, 'auxmol', None) is None:
                mf.with_df.build(build_cderi=False) if 'build_cderi' in mf.with_df.build.__code__.co_varnames \
                    else mf.with_df.build()
            nm = opt['direct_nmod'] or (None, None)
            if not lr_direct:
                dk_fr = DirectDFK(mol, mf.with_df.auxmol, None, nmod_t=nm[0], nmod_x=nm[1])
            if omega != 0:
                dk_lr = DirectDFK(mol, mf.with_df.auxmol, omega, nmod_t=nm[0], nmod_x=nm[1])
            if cache_budget:
                for d_ in (dk_fr, dk_lr):
                    if d_ is not None:
                        d_.cache_bytes = cache_budget
                log.info('multi-fidelity: direct passes with %.2f GB integral cache per operator', cache_budget / 1e9)
        g_cheap = g_nlc_cheap = None
        if opt['schedule']:
            g_cheap = _grid(mol, opt['cheap_grid'], mf.grids) if opt['cheap_grid'] else mf.grids
            g_nlc_cheap = (_grid(mol, opt['cheap_nlc_grid'], mf.nlcgrids) if opt['cheap_nlc_grid'] else mf.nlcgrids) \
                if with_nlc else None
        h1e = cupy.asarray(mf.get_hcore())
        s1e = cupy.asarray(mf.get_ovlp())
        e_nuc = mf.energy_nuc()
        eig = FastEig(s1e, opt['eig'], nocc)
        cx = CoarseXC(ni, mol, g_in, xc, opt['coarse_cache_frac']) if opt['coarse_xc'] == 'inc' else None
        sk = _NullK(mol.nao) if no_k else None
        if not no_k and (opt['ref_k'] != 'ace' or opt['inc_store'] in (None, 'j16') or opt['max_stage1'] > 0) \
                and opt['inc_store'] != 'f32':
            # dense FP32 surrogate (stage 1 and/or the K reference)
            sk = SurrogateK(mf, opt['surrogate_aux'], omega, alpha, hyb,
                            mode=opt['k_increment'] if opt['k_increment'] != 'exact' else 'fp32')
        sk8 = js = None
        if opt['inc_store']:
            # 'int8': INT8 K + INT16 J increments;  'k8' / 'k16' / 'k32': stored K only (k32: FP32, structured);
            # 'j16': INT16 J only;  'j16k32': INT16 J + FP32 K (K in pinned host memory when it does not fit)
            from gpu4pyscf.df.int8_df import Int8DF
            # 'f32': FP32 J and K tensors (pinned host memory when they do not fit): structured like the dense
            #        surrogate, so they replace it everywhere (stage 1, reference, increments) without a filter
            kb = {'int8': 8, 'k8': 8, 'k16': 16, 'k32': 32, 'j16k32': 32, 'f32': 32}.get(opt['inc_store'])
            if kb:
                sk8 = SurrogateK8(mf, opt['surrogate_aux'], omega, alpha, hyb, bits=kb, host=opt['surrogate_host'])
            if opt['inc_store'] in ('int8', 'j16', 'j16k32', 'f32'):
                js = Int8DF(mol, opt['surrogate_j_aux'], bits=32 if opt['inc_store'] == 'f32' else 16,
                            host=opt['surrogate_host'])
            if opt['inc_store'] == 'f32':
                sk = sk8
        flt = None
        if opt['inc_filter']:
            w_, U_ = cupy.linalg.eigh(s1e)
            keep = w_ > opt['inc_filter']
            Xk = U_[:, keep] / cupy.sqrt(w_[keep])
            flt = (Xk, s1e @ Xk)
            log.info('multi-fidelity: increment filter keeps %d of %d S eigenvectors', int(keep.sum()), len(w_))
            w_ = U_ = Xk = None
    conv_tol_grad = opt['conv_tol_grad'] or getattr(mf, 'conv_tol_grad', None) or np.sqrt(mf.conv_tol)

    def tagged(C_full, occ):
        D = 2 * C_full[:, occ > 0] @ C_full[:, occ > 0].T
        return tag_array(D, mo_coeff=C_full, mo_occ=occ)

    def solve(f):
        with tim('eig'):
            e, C = eig(f)
        occ = cupy.zeros(C.shape[1])
        occ[:nocc] = 2
        return e, C, occ

    if opt['inner_eig'] == 'auto':
        opt['inner_eig'] = 'rotate' if mol.nao > opt['rotate_nao'] else 'full'
    Sinv = eig.X @ eig.X.T if opt['inner_eig'] == 'rotate' else None

    def proj_grad(X_, Co_):
        return _proj_grad(X_, Co_, Sinv, s1e)

    def occ_grad(X_, Co_, C_):
        '''|g| of an occupied-column operator X_ = F Co_ (projected form, or the virtual block of C_)'''
        if Sinv is not None:
            return proj_grad(X_, Co_)
        return 2 * float(cupy.linalg.norm(C_[:, nocc:].T @ X_))

    def rotate(fd, C, e):
        return _rotate(fd, C, e, s1e, nocc, opt['rotate_steps'])

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

    j_abs = opt['inc_store'] == 'f32'      # surrogate J usable as an absolute term (FP32 storage)

    def get_j(dm):
        with tim('J'):
            if j_abs:
                return js.j(dm)
            return mf.get_j(mol, dm, hermi=1)

    def filt(V):
        if flt is None:
            return V
        Xk, SXk = flt
        return SXk @ ((Xk.T @ V @ Xk) @ SXk.T)

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
            if getattr(mf.with_df, '_cderi', None) is None and opt['exact_k'] != 'direct':
                mf.with_df.build()                 # exact tensor first: J then takes the fast path
            if opt.get('exact_store') == 'f32' and opt['exact_k'] != 'direct':
                # compact FP32 storage of the metric-whitened exact tensors B = L^-1 (P|mn) (no J^-1 amplification
                # of their rounding, unlike raw integrals); widened to FP64 block by block on use
                def _f32(d):
                    keys = list(d._cderi.keys()) if isinstance(d._cderi, dict) else range(len(d._cderi))
                    for k_ in keys:
                        v_ = d._cderi[k_]
                        if isinstance(v_, cupy.ndarray) and v_.dtype == np.float64:
                            d._cderi[k_] = v_.astype(np.float32)
                        v_ = None
                    cupy.get_default_memory_pool().free_all_blocks()
                _f32(mf.with_df)
                if omega != 0 and not no_k:
                    with mf.with_df.range_coulomb(omega, None, None) as d_lr:
                        if d_lr._cderi is None:
                            d_lr.build(omega=omega)
                        _f32(d_lr)
            if js is not None:
                js.build(mem_frac=0.1)             # stored Coulomb tensor (INT16 increments / FP32)
            if sk8 is not None:
                sk8.ensure()
            if j_abs:
                cupy.get_default_memory_pool().free_all_blocks()
        diis = new_diis()
        f = None
        e = None
        g = float('nan')
        if opt['max_stage1'] == 0:
            # no cheap stage (no structured absolute surrogate available): occupied natural orbitals of the guess
            Xs = eig.X
            SXs = s1e @ Xs
            n_, V_ = cupy.linalg.eigh(SXs.T @ dm @ SXs)
            C = Xs @ V_[:, ::-1]
            occ = cupy.zeros(C.shape[1])
            occ[:nocc] = 2
            dm = tagged(C, occ)
            Xs = SXs = n_ = V_ = None
        for it in range(opt['max_stage1']):
            vj = get_j(dm)
            with tim('K_s stage1'):
                if it == 0 and opt['k_increment'] != 'exact' and isinstance(sk, SurrogateK):
                    vk = sk.k_dm(dm, s1e)          # natural orbitals of the guess: no separate DF-K pass
                elif it == 0 or opt['k_increment'] == 'exact':
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
                log.debug('multi-fidelity: stage 1 cycle %d |g| = %.3e', it, g)
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
        e_models = []
        diag_prev = {}
        e_tot = None
        mf._mf_xc_grad = None
        last_inner_tol = None
        xctype = ni._xc_type(xc)
        levels = []
        occ_t = None
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
            # fuse the XC nuclear gradient only where this checkpoint is predicted to be the last one: |g| after the
            # last inner loop ~ |g_prev| / contraction (measured on the previous outer step, else contraction_guess)
            ratio = (hist[-2][0] / max(hist[-1][0], 1e-300)) if len(hist) >= 2 else opt['contraction_guess']
            pred = hist[-1][0] / max(ratio, 1.0) if hist else float('inf')
            want_grad = (level == 'exact' and opt['fuse_grad'] and opt['fine_vco'] and xctype in ('GGA', 'MGGA')
                         and last_inner_tol is not None and last_inner_tol < conv_tol_grad and pred < conv_tol_grad)
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
                if opt.get('diag'):
                    if diag_prev.get('vnlc') is not None:
                        # VV10 is frozen inside Delta in the inner loop: its response over the last outer step
                        log.info('    diag: |g| of the VV10 change since the last checkpoint %.3e',
                                 occ_grad((vnlc - diag_prev['vnlc']) @ Co, Co, C))
                    diag_prev['vnlc'] = vnlc
                    diag_prev['Wnlc'] = vnlc @ Co
                vnlc = None
            vj = None
            with lvl_ctx, tim('K exact' + ('' if level == 'exact' else '_' + level)):
                if no_k:
                    Wk = cupy.zeros_like(Co)
                elif opt['exact_k'] == 'direct':
                    cupy.get_default_memory_pool().free_all_blocks()
                    Cw = Co * np.sqrt(2.0)
                    # keep J^-1 (P|ij) (naux * nocc^2 per operator) only where this is predicted to be the last
                    # exact evaluation; otherwise the gradient forms its own fit
                    kf = last_inner_tol is not None and last_inner_tol < conv_tol_grad
                    kc, vj = dk_fr.kc_j(Cw, with_j=True, keep_fit=kf)
                    Wk = hyb * kc
                    if omega != 0:
                        Wk += (alpha - hyb) * dk_lr.kc_j(Cw, with_j=False, keep_fit=kf)[0]
                    Wk /= np.sqrt(2.0)
                    log.info('    direct K: integral cache holds %.0f %% of the aux batches',
                             100 * dk_fr.cached_fraction())
                elif opt['occ_k']:
                    # K(D) C_occ directly (no nao^2 K matrix); keeps T = C^T B C for the gradient fit
                    from gpu4pyscf.df.df_jk import get_kc_occ
                    Cw = Co * np.sqrt(2.0)
                    kc, t_fr = get_kc_occ(mf.with_df, Cw, keep_t=True)
                    Wk = hyb * kc
                    t_lr = None
                    if omega != 0 and dk_lr is not None:
                        # 'mixed' storage: the LR operator does not fit, two direct passes (J from the stored FR)
                        cupy.get_default_memory_pool().free_all_blocks()
                        kf = last_inner_tol is not None and last_inner_tol < conv_tol_grad
                        Wk += (alpha - hyb) * dk_lr.kc_j(Cw, with_j=False, keep_fit=kf)[0]
                    elif omega != 0:
                        with mf.with_df.range_coulomb(omega, None, None) as d_lr:
                            kc, t_lr = get_kc_occ(d_lr, Cw, keep_t=True, omega=omega)
                        Wk += (alpha - hyb) * kc
                    kc = None
                    Wk /= np.sqrt(2.0)
                    occ_t = (t_fr, t_lr)
                else:
                    vk = hyb * mf.get_k(mol, dm, hermi=1)
                    if omega != 0:
                        vk += (alpha - hyb) * mf.get_k(mol, dm, hermi=1, omega=omega)
                    Wk = vk @ Co
                    vk = None
            if vj is None:
                with tim('J exact'):
                    vj = mf.get_j(mol, dm, hermi=1)        # exact (stored DF) J at the checkpoint
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
            if len(hist) >= 2 and len(e_models) == len(hist) - 1:
                log.info('    model energy error %.3e (0.12 g_prev g = %.1e)', e_models[-1] - e_tot,
                         0.12 * hist[-2][0] * gt)
            if opt.get('diag') and outer == 0:
                # model error at the stage-1 exit, term by term (exact minus stage-1 operator, projected gradient)
                Wn = diag_prev.get('Wnlc', 0. * Co)
                gJ = occ_grad((vj - get_j(dm)) @ Co, Co, C)
                gK = occ_grad(-0.5 * (Wk - (sk.k_ref(dm) if not no_k else 0. * dm) @ Co), Co, C)
                gX = occ_grad(Wxc - Wn - vxc_coarse(dm) @ Co, Co, C)
                gN = occ_grad(Wn - (ni.nr_nlc_vxc(mol, g_nlc1, nlc_xc, dm)[2] @ Co if with_nlc else 0. * Co), Co, C)
                log.info('    diag: stage-1 model error |g| by term: J %.3e  K %.3e  XC %.3e  VV10 %.3e',
                         gJ, gK, gX, gN)
            if gt < conv_tol_grad and level == 'exact':
                break
            # cheap operator at D_ref and the occupied-projected defect
            with tim('K_s exact'):
                if no_k:
                    vks_ref = cupy.zeros_like(dm)
                elif opt['ref_k'] == 'ace':
                    # adaptively compressed exchange: rank-nocc, reproduces the exact K C_occ (K is positive definite
                    # on the occupied space for non-negative range-separation weights)
                    M = Co.T @ Wk
                    vks_ref = Wk @ cupy.linalg.solve(0.5 * (M + M.T), Wk.T)
                else:
                    vks_ref = sk.k_ref(dm)
            vc = vxc_coarse(dm)
            vn_ref = None
            if with_nlc and opt['inner_nlc']:
                with tim('nlc_coarse'):
                    vn_ref = ni.nr_nlc_vxc(mol, g_nlc_in, nlc_xc, dm)[2]
            # the exact J of the checkpoint is the reference; the stored surrogate J only supplies increments
            # (its rounding then scales with |D - D_ref|)
            vjs_ref = vj
            hs = hc
            W = Wt - (hs + vc - 0.5 * vks_ref) @ Co
            SC = s1e @ Co
            Delta = W @ SC.T + SC @ W.T - SC @ (Co.T @ W) @ SC.T
            if opt['k_increment'] != 'exact':
                with tim('K_s set_ref'):
                    (sk8 if sk8 is not None else sk).set_ref(Co, s1e)
            if cx is not None:
                with tim('xc_coarse set_ref'):
                    cx.set_ref(Co)
            Co_ref = Co
            f = hs + vc - 0.5 * vks_ref + Delta
            f_ref0 = f
            Dref_tag = dm
            diis = new_diis()
            inner_tol = max(opt['inner_factor'] * gt, 0.3 * conv_tol_grad * opt['inner_factor'])
            last_inner_tol = inner_tol
            for it in range(opt['max_inner']):
                with tim('diis'):
                    fd = diis.update(s1e, dm, f)
                if opt['inner_eig'] == 'rotate' and e is not None:
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
                    elif sk8 is not None:
                        vks = vks_ref + (sk8.k_inc_dC(dC) if sk8.t.bits == 32 else filt(sk8.k_inc_dC(dC)))
                    else:
                        vks = vks_ref + sk.k_inc_dC(dC)
                if cx is not None:
                    with tim('xc_coarse inc'):
                        vci = vc + cx.inc(dC)
                else:
                    vci = vxc_coarse(dm)
                if vn_ref is not None:
                    with tim('nlc_coarse inc'):
                        vci = vci + (ni.nr_nlc_vxc(mol, g_nlc_in, nlc_xc, dm)[2] - vn_ref)
                if js is not None:
                    with tim('J'):
                        dJ = js.j(dm - Dref_tag)
                        vji = vjs_ref + (dJ if js.bits == 32 else filt(dJ))   # J(D_ref) + stored-tensor increment
                else:
                    vji = get_j(dm)
                f = h1e + vji + vci - 0.5 * vks + Delta
                gs = orb_grad(f, C)
                if gs < inner_tol:
                    break
            log.info('    inner cycles %d  |g_s| = %.2e', it + 1, gs)
            # model energy of the new density (trapezoid on the model Fock, exact on the occupied columns at D_ref):
            # E(D) ~ E(D_ref) + 1/2 tr[(F(D_ref) + F~(D)) (D - D_ref)]; checked against the next exact checkpoint
            dD = dm - Dref_tag
            e_model = e_tot + 0.5 * float(cupy.einsum('ij,ji->', f_ref0 + f, dD))
            e_models.append(e_model)
            log.info('    model energy of the next density %.12f', e_model)
            dD = f_ref0 = None

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
            if dk_fr is not None and dk_fr._fit is None:
                cache = None
            elif dk_fr is not None:
                # direct exact evaluations: the last pass already formed J^-1 (P|ij); rotate to the canonical orbitals
                cache = {'C': Cw, 'omega': abs(omega), 'fr': dk_fr.fit_oo(Uo)}
                if omega != 0:
                    cache['lr'] = dk_lr.fit_oo(Uo)
            elif occ_t is not None:
                # T = Cw^T B Cw of the last exact K: R = aux_coef T, rotated to the canonical orbitals
                from gpu4pyscf.df.df import _decompose_j2c

                def fit(dfobj, T, om):
                    cached = getattr(dfobj.intopt, '_aux_coef_cache', None)
                    if cached is not None and cached[0] == (abs(om or 0.), None, None) and cached[2] is not None:
                        ac = cached[1][cupy.asarray(cached[2])]          # sorted -> native aux order
                    else:
                        ac = _decompose_j2c(dfobj.intopt.auxmol, None, om, None, None)[0]
                    R = ac.dot(T.reshape(T.shape[0], -1)).reshape(-1, nocc, nocc)
                    return cupy.einsum('ki,rkl,lj->rij', Uo, R, Uo)
                cache = {'C': Cw, 'omega': abs(omega), 'fr': fit(mf.with_df, occ_t[0], None)}
                if omega != 0 and dk_lr is not None:
                    if dk_lr._fit is None:
                        cache = None
                    else:
                        cache['lr'] = dk_lr.fit_oo(Uo)
                elif omega != 0:
                    with mf.with_df.range_coulomb(omega, None, None) as d:
                        cache['lr'] = fit(d, occ_t[1], omega)
            else:
                cache = {'C': Cw, 'omega': abs(omega), 'fr': df_rhf_grad.precompute_fit_oo(mf.with_df, Cw)}
                if omega != 0:
                    with mf.with_df.range_coulomb(omega, None, None) as d:
                        cache['lr'] = df_rhf_grad.precompute_fit_oo(d, Cw, omega=omega)
            df_rhf_grad.FIT_OO_CACHE = cache
    # host-streamed surrogate tensors live in a pinned pool, which keeps freed blocks: return them to the OS
    # (tens of GB at Taxol size; the gradient needs that host memory).  gpu4pyscf installs its own pool as cupy's
    # pinned allocator (lib.cupy_helper.pinned_memory_pool); cupy's default pool is a different object.
    sk = sk8 = js = None
    import gc
    from gpu4pyscf.lib import cupy_helper as _ch
    gc.collect()
    _ch.pinned_memory_pool.free_all_blocks()
    cupy.get_default_pinned_memory_pool().free_all_blocks()
    mf.converged = hist[-1][0] < conv_tol_grad and levels[-1] == 'exact'
    mf.multifidelity_stats = dict(cycles=ncyc, outer_hist=hist, e_models=e_models, timers=tim.t, levels=levels,
                                  wall=time.perf_counter() - t_start, conv_tol_grad=conv_tol_grad)
    mf.cycles = ncyc['inner'] + ncyc['stage1']
    log.note('multi-fidelity SCF: E = %.12f  converged = %s  outer %d inner %d stage1 %d',
             e_tot, mf.converged, ncyc['outer'], ncyc['inner'], ncyc['stage1'])
    return e_tot
