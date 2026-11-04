# Copyright 2021-2026 The PySCF Developers. All Rights Reserved.
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
Fused per-grid-block kernels for the XC quadrature loops (numint.nr_rks and friends).

The grid-block loops of the DFT quadrature are launch bound on fast GPUs: one 4096-point block
of a meta-GGA evaluation used to issue ~37 small kernels (cupy glue for the rho assembly, the weight
scaling and the Vxc accumulation).  The kernels below replace those chains:

  rho_from_c : rho/|nabla rho|/tau from the half-transformed orbitals c = C^T [ao_0 .. ao_3],
               written straight into the global [nvar, ngrids] density array
  build_B    : the weighted AO factor(s) of the Vxc matrix build.  For meta-GGA this produces the
               4 operands B_0 = sum_x w_x ao_x and B_x = w_tau ao_x (x=1..3) so that
               Vxc += sum_x ao_x B_x^T is a single contraction of length 4*ngrids.

They are compiled with NVRTC on first use (no extension rebuild needed) in FP64 or FP32 flavour.
Sums over orbitals/components use a fixed thread mapping, hence the results are deterministic.
'''

import numpy as np
import cupy

_SRC = r'''
#ifndef REAL
#define REAL double
#endif
#define BX 32
#define BY 8

// c: [ncomp, nocc, ng] (ncomp = 1 for mode 0, 4 otherwise), rho: [nvar, rho_stride] (double)
// mode 0: rho = sum c0^2;  mode 1 (GGA): + 2 sum c0 c_x;  mode 2 (MGGA): + tau = 1/2 sum_x sum c_x^2
extern "C" __global__
void rho_from_c(double *rho, const REAL *c, long long ng, long long nocc, long long mode,
                long long rho_off, long long rho_stride)
{
    long long g = (long long)blockIdx.x * BX + threadIdx.x;
    int ty = threadIdx.y;
    REAL v0 = 0, v1 = 0, v2 = 0, v3 = 0, v4 = 0;
    if (g < ng) {
        unsigned long long cs = (unsigned long long)nocc * ng;
        for (long long i = ty; i < nocc; i += BY) {
            unsigned long long off = (unsigned long long)i * ng + g;
            REAL c0 = c[off];
            v0 += c0 * c0;
            if (mode >= 1) {
                REAL c1 = c[off + cs];
                REAL c2 = c[off + 2*cs];
                REAL c3 = c[off + 3*cs];
                v1 += c0 * c1;
                v2 += c0 * c2;
                v3 += c0 * c3;
                v4 += c1 * c1 + c2 * c2 + c3 * c3;
            }
        }
    }
    __shared__ REAL sh[5][BY][BX];
    sh[0][ty][threadIdx.x] = v0;
    sh[1][ty][threadIdx.x] = v1;
    sh[2][ty][threadIdx.x] = v2;
    sh[3][ty][threadIdx.x] = v3;
    sh[4][ty][threadIdx.x] = v4;
    __syncthreads();
    int nvar = mode == 0 ? 1 : (mode == 1 ? 4 : 5);
    // thread row k reduces variable k (fixed summation order)
    if (ty < nvar && g < ng) {
        double s = 0;
        for (int j = 0; j < BY; j++) {
            s += sh[ty][j][threadIdx.x];
        }
        if (ty >= 1 && ty <= 3) s *= 2;
        if (ty == 4) s *= 0.5;
        rho[ty * rho_stride + rho_off + g] = s;
    }
}

// ao: [ncomp, nao, ng], wv: [nvar, wv_stride] (block starts at wv_off), B: [ncomp_B, nao, ng]
// mode 0: B_0 = ao_0 w_0 (ao is [nao, ng]);  mode 1: B_0 = sum_{x<4} ao_x w_x;
// mode 2: as mode 1, plus B_x = w_4 ao_x (x = 1..3); w_4 must already contain the 1/4 tau factor.
extern "C" __global__
void build_B(REAL *B, const REAL *ao, const REAL *wv, long long ng, long long nao, long long mode,
             long long wv_off, long long wv_stride)
{
    long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long i = (long long)blockIdx.y * blockDim.y + threadIdx.y;
    if (g >= ng || i >= nao) return;
    unsigned long long Nag = (unsigned long long)nao * ng;
    unsigned long long ixy = (unsigned long long)i * ng + g;
    const REAL *w = wv + wv_off + g;
    REAL val = ao[ixy] * (REAL)w[0];
    if (mode >= 1) {
        val += ao[ixy + Nag] * (REAL)w[wv_stride];
        val += ao[ixy + 2*Nag] * (REAL)w[2*wv_stride];
        val += ao[ixy + 3*Nag] * (REAL)w[3*wv_stride];
    }
    B[ixy] = val;
    if (mode == 2) {
        REAL wt = w[4*wv_stride];
        B[ixy + Nag] = wt * ao[ixy + Nag];
        B[ixy + 2*Nag] = wt * ao[ixy + 2*Nag];
        B[ixy + 3*Nag] = wt * ao[ixy + 3*Nag];
    }
}

// out[i*3 + n] (+)= sum_d sum_g A[comps[d*3+n], i, g] * Y[d, i, g]       (ncomp == 3)
// A: [*, nao, ng], Y: [nd, nao, ng], out: [nao, 3].  One thread block per AO row, fixed reduction order.
extern "C" __global__
void row_dot3(double *out, const double *A, const double *Y, const int *comps,
              long long ng, long long nao, long long nd, long long accumulate)
{
    long long i = blockIdx.x;
    int tid = threadIdx.x;
    double v[3] = {0., 0., 0.};
    for (long long d = 0; d < nd; d++) {
        const double *y = Y + ((unsigned long long)d * nao + i) * ng;
        for (int n = 0; n < 3; n++) {
            const double *a = A + ((unsigned long long)comps[d*3+n] * nao + i) * ng;
            double s = 0.;
            for (long long g = tid; g < ng; g += 128) {
                s += a[g] * y[g];
            }
            v[n] += s;
        }
    }
    __shared__ double sh[3][128];
    sh[0][tid] = v[0];
    sh[1][tid] = v[1];
    sh[2][tid] = v[2];
    __syncthreads();
    for (int stride = 64; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sh[0][tid] += sh[0][tid + stride];
            sh[1][tid] += sh[1][tid + stride];
            sh[2][tid] += sh[2][tid + stride];
        }
        __syncthreads();
    }
    if (tid < 3) {
        double r = sh[tid][0];
        if (accumulate) r += out[i*3 + tid];
        out[i*3 + tid] = r;
    }
}
'''

_modules = {}

def _get_kernel(name, dtype):
    # keyed by device: a module/function loaded on one GPU cannot be launched on another
    dtype = np.dtype(dtype)
    dev = cupy.cuda.Device().id
    key = (name, dtype.str, dev)
    k = _modules.get(key)
    if k is None:
        mkey = ('module', dtype.str, dev)
        mod = _modules.get(mkey)
        if mod is None:
            real = 'double' if dtype == np.float64 else 'float'
            mod = cupy.RawModule(code=_SRC, options=('-DREAL=' + real,))
            _modules[mkey] = mod
        k = _modules[key] = mod.get_function(name)
    return k

_i64 = np.int64
_MODE = {'LDA': 0, 'HF': 0, 'GGA': 1, 'NLC': 1, 'MGGA': 2}

def rho_from_c(rho, p0, c, ng, nocc, xctype):
    '''rho[:, p0:p0+ng] = density variables from c [ncomp, nocc, ng] (see module doc).
    rho: C-contiguous float64 [nvar, ngrids_total]'''
    mode = _MODE[xctype]
    k = _get_kernel('rho_from_c', c.dtype)
    k(((ng + 31) // 32,), (32, 8), (rho, c, _i64(ng), _i64(nocc), _i64(mode),
                                    _i64(p0), _i64(rho.shape[-1])))
    return rho

def build_B(B, ao, wv, p0, ng, nao_sub, xctype):
    '''Operands of the Vxc contraction for the block wv[:, p0:p0+ng] (see module doc).'''
    mode = _MODE[xctype]
    k = _get_kernel('build_B', ao.dtype)
    k(((ng + 31) // 32, (nao_sub + 7) // 8), (32, 8), (B, ao, wv, _i64(ng), _i64(nao_sub), _i64(mode),
                                                     _i64(p0), _i64(wv.shape[-1])))
    return B

_comps = {}

def _comp_table(comps):
    key = (tuple(comps), cupy.cuda.Device().id)
    t = _comps.get(key)
    if t is None:
        t = _comps[key] = cupy.asarray(np.asarray(comps, dtype=np.int32))
    return t

def row_dot3(out, A, Y, comps, ng, nao_sub, nd=1, accumulate=False):
    '''out[i, n] (+)= sum_d sum_g A[comps[d][n], i, g] * Y[d, i, g] for n = 0..2.
    comps: flat tuple of nd*3 component indices of A.  FP64 only.'''
    k = _get_kernel('row_dot3', np.float64)
    k((nao_sub,), (128,), (out, A, Y, _comp_table(comps), _i64(ng), _i64(nao_sub), _i64(nd),
                           _i64(1 if accumulate else 0)))
    return out
