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

'''
Tiled pair kernels for the VV10 non-local correlation (U, W, E sums), written as
cupy RawKernels (no libgdft rebuild needed).

For every grid point i the O(N^2) pair sums

    E_i =  1.5 sum_j (rho w)_j Phi_ij
    U_i = -1.5 sum_j (rho w)_j Phi_ij (1/a_ij + 1/s_ij)
    W_i = -1.5 sum_j (rho w)_j Phi_ij (1/a_ij + 1/s_ij) R_ij^2

are evaluated, with a_ij = omega_i R_ij^2 + kappa_i, b_ij = a_ji, s = a + b and
Phi_ij = -1/(a b s).  This is algebraically identical to
``vv10_fock_eval_UWE_kernel`` of libgdft/vv10.cu, but

  * the points are Morton-sorted and processed in tiles; coordinates are stored
    relative to the tile centre, so that FP32 differences r_i - r_j are accurate
    to ~1e-7 *relative to the tile size* (no hi/lo coordinate pairs needed);
  * the op count per pair is reduced (one reciprocal, 1/a + 1/s = b (a+b+a)/(a b s)...);
  * in 'sym' mode every unordered pair is evaluated once and contributes to both
    the row sums of i and of j (Phi_ij = Phi_ji; only the per-point factors
    differ).  Column sums use a register tile + shuffle reduce-scatter over
    the lanes of a warp and a shared-memory accumulator flushed with FP64
    atomics once per tile pair.

Precision modes
  'fp64': all FP64, identical to libgdft up to summation order (~1e-15).
  'fp32': pair arithmetic in FP32 with MUFU reciprocal, FP32 partial sums over
          short chunks / one tile, flushed to FP64 once per tile.
'''

import numpy as np
import cupy

# Defaults chosen by the sweep in benchmarks/dft/RESULTS_vv10.md
NTHREADS = 128
MI = 4          # target (row) points per thread; tile size = NTHREADS * MI

_SRC = r"""
#if PREC32
typedef float REAL;
typedef float4 REAL4;
typedef float2 REAL2;
__device__ __forceinline__ static float vrcp(float x)
{
    float y;
    asm("rcp.approx.ftz.f32 %%0, %%1;" : "=f"(y) : "f"(x));
    return y;
}
#else
typedef double REAL;
typedef double4 REAL4;
typedef double2 REAL2;
__device__ __forceinline__ static double vrcp(double x) { return 1.0 / x; }
#endif

#define NT %(NT)d
#define MI %(MI)d
#define TILE (NT * MI)
#define CH 16
#define FULL 0xffffffffu

/* One-sided: every row i sums over all j.  grid = (ntile,) */
extern "C" __global__
void vv10_onesided(double* __restrict__ Uo, double* __restrict__ Wo, double* __restrict__ Eo,
                   const double* __restrict__ xg,      // (Np, 3) global coordinates
                   const REAL4* __restrict__ pj,       // (xl, yl, zl, rho*w), tile-local
                   const REAL2* __restrict__ ok,       // (omega, kappa)
                   const double* __restrict__ cen,     // (ntile, 3) tile centres
                   const double* __restrict__ rad,     // (ntile) tile radii
                   const int ntile, const double rcut)
{
    const int tid = threadIdx.x;
    const int I = blockIdx.x;
    double gx[MI], gy[MI], gz[MI];
    REAL om[MI], ka[MI];
    double Ua[MI], Wa[MI], Ea[MI];
#pragma unroll
    for (int k = 0; k < MI; k++) {
        const int i = I * TILE + k * NT + tid;
        gx[k] = xg[i * 3 + 0]; gy[k] = xg[i * 3 + 1]; gz[k] = xg[i * 3 + 2];
        const REAL2 o = ok[i];
        om[k] = o.x; ka[k] = o.y;
        Ua[k] = 0; Wa[k] = 0; Ea[k] = 0;
    }
    __shared__ REAL4 s4[NT];
    __shared__ REAL2 s2[NT];
    const double cIx = cen[I * 3 + 0], cIy = cen[I * 3 + 1], cIz = cen[I * 3 + 2];
    const double rI = rad[I];

    for (int J = 0; J < ntile; J++) {
        const double cx = cen[J * 3 + 0], cy = cen[J * 3 + 1], cz = cen[J * 3 + 2];
        if (rcut > 0) {
            const double d = sqrt((cx - cIx) * (cx - cIx) + (cy - cIy) * (cy - cIy) + (cz - cIz) * (cz - cIz));
            if (d - rI - rad[J] > rcut) continue;       // block-uniform branch
        }
        REAL xr[MI], yr[MI], zr[MI];
        REAL Ut[MI], Wt[MI], Et[MI];
#pragma unroll
        for (int k = 0; k < MI; k++) {
            xr[k] = (REAL)(gx[k] - cx); yr[k] = (REAL)(gy[k] - cy); zr[k] = (REAL)(gz[k] - cz);
            Ut[k] = 0; Wt[k] = 0; Et[k] = 0;
        }
        for (int jc = 0; jc < TILE; jc += NT) {
            __syncthreads();
            s4[tid] = pj[J * TILE + jc + tid];
            s2[tid] = ok[J * TILE + jc + tid];
            __syncthreads();
            for (int j0 = 0; j0 < NT; j0 += CH) {
#if PREC32
                REAL Uc[MI], Wc[MI], Ec[MI];
#pragma unroll
                for (int k = 0; k < MI; k++) { Uc[k] = 0; Wc[k] = 0; Ec[k] = 0; }
#endif
#pragma unroll 4
                for (int jj = j0; jj < j0 + CH; jj++) {
                    const REAL4 p = s4[jj];
                    const REAL2 o = s2[jj];
#pragma unroll
                    for (int k = 0; k < MI; k++) {
                        const REAL dx = xr[k] - p.x, dy = yr[k] - p.y, dz = zr[k] - p.z;
                        const REAL r2 = fma(dx, dx, fma(dy, dy, dz * dz));
                        const REAL a = fma(om[k], r2, ka[k]);
                        const REAL b = fma(o.x, r2, o.y);
                        const REAL s = a + b;
                        const REAL ab = a * b;
                        const REAL P = vrcp(ab * s);
                        const REAL q = P * P;
                        const REAL m = (p.w * q) * fma(b, s, ab);     // b (s + a) = b s + a b
#if PREC32
                        Ec[k] = fma(p.w, P, Ec[k]); Uc[k] += m; Wc[k] = fma(m, r2, Wc[k]);
#else
                        Et[k] = fma(p.w, P, Et[k]); Ut[k] += m; Wt[k] = fma(m, r2, Wt[k]);
#endif
                    }
                }
#if PREC32
#pragma unroll
                for (int k = 0; k < MI; k++) { Et[k] += Ec[k]; Ut[k] += Uc[k]; Wt[k] += Wc[k]; }
#endif
            }
        }
#pragma unroll
        for (int k = 0; k < MI; k++) {
            Ea[k] += (double)Et[k]; Ua[k] += (double)Ut[k]; Wa[k] += (double)Wt[k];
        }
    }
#pragma unroll
    for (int k = 0; k < MI; k++) {
        const int i = I * TILE + k * NT + tid;
        Uo[i] = Ua[k]; Wo[i] = Wa[k]; Eo[i] = Ea[k];
    }
}

/* Symmetric: unordered tile pairs {I, I+o mod ntile}, o = 0..H, split over gridDim.y.
   o = 0 is the diagonal tile, evaluated one-sided; all other pairs update row sums of
   I and column sums of J.  The outputs must be zero initialised (atomicAdd). */
extern "C" __global__
void vv10_sym(double* __restrict__ Uo, double* __restrict__ Wo, double* __restrict__ Eo,
              const double* __restrict__ xg, const REAL4* __restrict__ pj, const REAL2* __restrict__ ok,
              const double* __restrict__ cen, const double* __restrict__ rad,
              const int ntile, const int H, const double rcut)
{
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int I = blockIdx.x;
    const int S = gridDim.y;
    double gx[MI], gy[MI], gz[MI];
    REAL om[MI], ka[MI], rwi[MI];
    double Ua[MI], Wa[MI], Ea[MI];
#pragma unroll
    for (int k = 0; k < MI; k++) {
        const int i = I * TILE + k * NT + tid;
        gx[k] = xg[i * 3 + 0]; gy[k] = xg[i * 3 + 1]; gz[k] = xg[i * 3 + 2];
        const REAL2 o = ok[i];
        om[k] = o.x; ka[k] = o.y;
        rwi[k] = pj[i].w;
        Ua[k] = 0; Wa[k] = 0; Ea[k] = 0;
    }
    __shared__ REAL4 s4[NT];
    __shared__ REAL2 s2[NT];
    __shared__ REAL sc[3 * TILE];
    for (int t = tid; t < 3 * TILE; t += NT) sc[t] = 0;
    const double cIx = cen[I * 3 + 0], cIy = cen[I * 3 + 1], cIz = cen[I * 3 + 2];
    const double rI = rad[I];

    for (int o = blockIdx.y; o <= H; o += S) {
        const bool diag = (o == 0);
        if (!diag && 2 * o == ntile && I >= o) continue;     // half-offset pair counted once
        const int J = (I + o) %% ntile;
        const double cx = cen[J * 3 + 0], cy = cen[J * 3 + 1], cz = cen[J * 3 + 2];
        if (rcut > 0) {
            const double d = sqrt((cx - cIx) * (cx - cIx) + (cy - cIy) * (cy - cIy) + (cz - cIz) * (cz - cIz));
            if (d - rI - rad[J] > rcut) continue;
        }
        REAL xr[MI], yr[MI], zr[MI];
        REAL Ut[MI], Wt[MI], Et[MI];
#pragma unroll
        for (int k = 0; k < MI; k++) {
            xr[k] = (REAL)(gx[k] - cx); yr[k] = (REAL)(gy[k] - cy); zr[k] = (REAL)(gz[k] - cz);
            Ut[k] = 0; Wt[k] = 0; Et[k] = 0;
        }
        for (int jc = 0; jc < TILE; jc += NT) {
            __syncthreads();
            s4[tid] = pj[J * TILE + jc + tid];
            s2[tid] = ok[J * TILE + jc + tid];
            __syncthreads();
            for (int jg = 0; jg < NT; jg += 8) {
                REAL Uc[MI], Wc[MI], Ec[MI];
                REAL cE[8], cU[8], cW[8];
#pragma unroll
                for (int k = 0; k < MI; k++) { Uc[k] = 0; Wc[k] = 0; Ec[k] = 0; }
#pragma unroll
                for (int jl = 0; jl < 8; jl++) {
                    const REAL4 p = s4[jg + jl];
                    const REAL2 oj = s2[jg + jl];
                    REAL ce = 0, cu = 0, cw = 0;
#pragma unroll
                    for (int k = 0; k < MI; k++) {
                        const REAL dx = xr[k] - p.x, dy = yr[k] - p.y, dz = zr[k] - p.z;
                        const REAL r2 = fma(dx, dx, fma(dy, dy, dz * dz));
                        const REAL a = fma(om[k], r2, ka[k]);
                        const REAL b = fma(oj.x, r2, oj.y);
                        const REAL s = a + b;
                        const REAL ab = a * b;
                        const REAL P = vrcp(ab * s);
                        const REAL q = P * P;
                        const REAL m = (p.w * q) * fma(b, s, ab);     // b (s + a) = b s + a b
                        Ec[k] = fma(p.w, P, Ec[k]); Uc[k] += m; Wc[k] = fma(m, r2, Wc[k]);
                        const REAL mc = (rwi[k] * q) * fma(a, s, ab); // a (s + b) = a s + a b
                        ce = fma(rwi[k], P, ce); cu += mc; cw = fma(mc, r2, cw);
                    }
                    cE[jl] = ce; cU[jl] = cu; cW[jl] = cw;
                }
#pragma unroll
                for (int k = 0; k < MI; k++) { Et[k] += Ec[k]; Ut[k] += Uc[k]; Wt[k] += Wc[k]; }
                if (!diag) {
                    // reduce-scatter of the 3 x 8 column partials over the 32 lanes of the warp
                    REAL u[3];
                    const bool h4 = lane & 16, h3 = lane & 8, h2 = lane & 4;
#pragma unroll
                    for (int q = 0; q < 3; q++) {
                        const REAL* c = (q == 0) ? cE : ((q == 1) ? cU : cW);
                        REAL v[4], w[2];
#pragma unroll
                        for (int t = 0; t < 4; t++) {
                            const REAL send = h4 ? c[t] : c[t + 4];
                            const REAL keep = h4 ? c[t + 4] : c[t];
                            v[t] = keep + __shfl_xor_sync(FULL, send, 16);
                        }
#pragma unroll
                        for (int t = 0; t < 2; t++) {
                            const REAL send = h3 ? v[t] : v[t + 2];
                            const REAL keep = h3 ? v[t + 2] : v[t];
                            w[t] = keep + __shfl_xor_sync(FULL, send, 8);
                        }
                        {
                            const REAL send = h2 ? w[0] : w[1];
                            const REAL keep = h2 ? w[1] : w[0];
                            u[q] = keep + __shfl_xor_sync(FULL, send, 4);
                        }
                        u[q] += __shfl_xor_sync(FULL, u[q], 2);
                        u[q] += __shfl_xor_sync(FULL, u[q], 1);
                    }
                    if ((lane & 3) == 0) {
                        const int jsel = ((lane >> 4) & 1) * 4 + ((lane >> 3) & 1) * 2 + ((lane >> 2) & 1);
                        const int jt = jc + jg + jsel;
                        atomicAdd(&sc[jt], u[0]);
                        atomicAdd(&sc[TILE + jt], u[1]);
                        atomicAdd(&sc[2 * TILE + jt], u[2]);
                    }
                }
            }
        }
#pragma unroll
        for (int k = 0; k < MI; k++) {
            Ea[k] += (double)Et[k]; Ua[k] += (double)Ut[k]; Wa[k] += (double)Wt[k];
        }
        if (!diag) {
            __syncthreads();
            for (int t = tid; t < TILE; t += NT) {
                const int j = J * TILE + t;
                atomicAdd(&Eo[j], (double)sc[t]);
                atomicAdd(&Uo[j], (double)sc[TILE + t]);
                atomicAdd(&Wo[j], (double)sc[2 * TILE + t]);
                sc[t] = 0; sc[TILE + t] = 0; sc[2 * TILE + t] = 0;
            }
        }
    }
#pragma unroll
    for (int k = 0; k < MI; k++) {
        const int i = I * TILE + k * NT + tid;
        atomicAdd(&Eo[i], Ea[k]); atomicAdd(&Uo[i], Ua[k]); atomicAdd(&Wo[i], Wa[k]);
    }
}
"""

_kernels = {}


def _get_kernels(prec32, nt, mi):
    key = (prec32, nt, mi)
    if key not in _kernels:
        src = '#define PREC32 %d\n' % int(prec32) + _SRC % {'NT': nt, 'MI': mi}
        mod = cupy.RawModule(code=src, options=('-std=c++14',))
        _kernels[key] = (mod.get_function('vv10_onesided'), mod.get_function('vv10_sym'))
    return _kernels[key]


def _morton_order(coords):
    lo = coords.min(axis=0)
    span = float((coords.max(axis=0) - lo).max()) + 1e-12
    q = cupy.minimum(((coords - lo) / span * 1024).astype(cupy.int64), 1023)
    code = cupy.zeros(coords.shape[0], dtype=cupy.int64)
    for b in range(10):
        for d in range(3):
            code |= ((q[:, d] >> b) & 1) << (3 * b + d)
    return cupy.argsort(code)


def prepare(coords, rho_weight, omega, kappa, prec32, tile, sort=True):
    '''Sort, pad and tile the input arrays. Returns a dict of device arrays.'''
    n = coords.shape[0]
    order = _morton_order(coords) if sort else cupy.arange(n)
    npad = (n + tile - 1) // tile * tile
    ntile = npad // tile
    xg = cupy.empty((npad, 3))
    xg[:n] = coords[order]
    xg[n:] = xg[n - 1]
    rw = cupy.zeros(npad); rw[:n] = rho_weight[order]
    om = cupy.ones(npad); om[:n] = omega[order]
    ka = cupy.ones(npad); ka[:n] = kappa[order]
    cen = xg.reshape(ntile, tile, 3).mean(axis=1)
    loc = xg.reshape(ntile, tile, 3) - cen[:, None, :]
    rad = cupy.sqrt((loc**2).sum(axis=2).max(axis=1))
    rt = cupy.float32 if prec32 else cupy.float64
    if prec32:
        om = cupy.minimum(om, 1e9)
    pj = cupy.empty((npad, 4), dtype=rt)
    pj[:, :3] = loc.reshape(npad, 3)
    pj[:, 3] = rw
    ok = cupy.empty((npad, 2), dtype=rt)
    ok[:, 0] = om; ok[:, 1] = ka
    return dict(n=n, npad=npad, ntile=ntile, order=order, xg=xg, pj=pj, ok=ok, cen=cen, rad=rad)


def eval_UWE(coords, rho_weight, omega, kappa, precision='fp64', symmetric=True, rcut=0.0,
             nthreads=NTHREADS, mi=MI, split=None, prepared=None):
    '''Returns U, W, E (float64, shape (n,)) in the input point order,
    with the same sign/scale convention as VXC_vv10nlc_fock_eval_UWE.

    rcut > 0 skips tile pairs whose bounding spheres are further apart than rcut (bohr);
    the neglected pair sums decay as R^-6 (see benchmarks/dft/RESULTS_vv10.md).
    '''
    prec32 = precision == 'fp32'
    tile = nthreads * mi
    d = prepared if prepared is not None else prepare(coords, rho_weight, omega, kappa, prec32, tile)
    npad, ntile = d['npad'], d['ntile']
    k1, ks = _get_kernels(prec32, nthreads, mi)
    args = (d['xg'], d['pj'], d['ok'], d['cen'], d['rad'])
    if symmetric:
        U = cupy.zeros(npad); W = cupy.zeros(npad); E = cupy.zeros(npad)
        H = ntile // 2
        if split is None:
            split = max(1, min(H + 1, 4096 // max(ntile, 1)))
        ks((ntile, split), (nthreads,), (U, W, E) + args + (np.int32(ntile), np.int32(H), np.float64(rcut)))
    else:
        U = cupy.empty(npad); W = cupy.empty(npad); E = cupy.empty(npad)
        k1((ntile,), (nthreads,), (U, W, E) + args + (np.int32(ntile), np.float64(rcut)))
    n, order = d['n'], d['order']
    out = []
    for A, f in ((U, 1.5), (W, 1.5), (E, -1.5)):
        o = cupy.empty(n)
        o[order] = f * A[:n]
        out.append(o)
    return tuple(out)
