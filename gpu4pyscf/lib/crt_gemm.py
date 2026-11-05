# Copyright 2021-2026 The PySCF Developers. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.
'''
FP64-class GEMM from INT8 tensor-core products via the Chinese remainder theorem (Ozaki-scheme-II style).

    A (m, K) ~ diag(2^ea) Ai 2^-q,   B^T (n, K) ~ diag(2^eb) Bi 2^-q,   Ai, Bi integers with |.| < 2^q
    C_int = Ai Bi^T is recovered exactly from its residues  (Ai mod p)(Bi mod p)^T  for pairwise-coprime p <= 256;
    every residue product is an exact int8 x int8 -> int32 cuBLAS GEMM (K * 128^2 < 2^31).

The reduction axis K may be the concatenation of several 'component' slabs (comp, rows, ng) -> (rows, comp*ng); they share
the row scale, so the whole sum over components and grid points is one residue GEMM per modulus.

The number of moduli n sets the precision (about 8 bits each; q is chosen as large as the modulus product allows,
capped at `QMAX`).  Needs a GPU with INT8 GEMM support (Turing and newer; m, n, K padded to multiples of 16).
Design notes and measurements: benchmarks/dft/crt_int8.py, RESULTS_mixed_precision.md.
'''
import numpy as np
import cupy

ALL_MODULI = [256, 255, 253, 251, 247, 241, 239, 233, 229, 227, 223, 211, 199, 197, 193, 191, 181, 179, 173, 167]
QMAX = 52            # operand bits; above 30 bits the quantization uses int64 values and three-limb residues
MAX_K = 120000       # int32 accumulation limit of one residue GEMM: K * 128^2 < 2^31

_kern = {}


def _resid_code(i, p, wide, dst):
    '''C statements: r = x mod p (balanced) -> dst.  Wide (int64) values are split into three int32 limbs
    (x = xh 2^36 + xm 2^18 + xl), because 64-bit `%` is slow on the GPU.  Valid for |x| < 2^54.'''
    h = (p + 1) // 2
    if wide:
        c18, c36 = (1 << 18) % p, (1 << 36) % p
        core = f'int r = ((xh % {p}) * {c36} + xm * {c18} + xl) % {p};'
    else:
        core = f'int r = x % {p};'
    return f'{{ {core} if (r < 0) r += {p}; if (r >= {h}) r -= {p}; {dst} = (signed char)r; }}'


def plan(K, nmod, qmax=QMAX):
    '''q (bits of each operand) such that prod(p_i) > 2 K 2^{2q}.'''
    bits = float(np.sum(np.log2(ALL_MODULI[:nmod])))
    q = int((bits - np.log2(K) - 1) // 2)
    return max(1, min(q, qmax))


def _quant_src(n, wide):
    T = 'long long' if wide else 'int'
    pre = 'int xh = (int)(x >> 36), xm = (int)((x >> 18) & 0x3FFFF), xl = (int)(x & 0x3FFFF);' if wide else ''
    body = [_resid_code(i, p, wide, f'out[(size_t){i} * plane + idx]') for i, p in enumerate(ALL_MODULI[:n])]
    return r'''
extern "C" __global__ void quant(const double* __restrict__ a, const int* __restrict__ ex, signed char* __restrict__ out,
                                 int ncomp, int rows, int ng, int rp, int kp, int q, size_t plane,
                                 const int* __restrict__ cmap)
{
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (size_t)rp * kp) return;
    int row = (int)(gid / kp), col = (int)(gid % kp);
    size_t idx = gid;
    ''' + T + r''' x = 0;
    if (row < rows && col < ncomp * ng) {
        int c = col / ng, g = col - c * ng;
        double v = ldexp(a[((size_t)cmap[c] * rows + row) * ng + g], q - ex[row]);
        x = (''' + T + r''')llrint(v);
    }
    ''' + pre + '\n    ' + '\n    '.join(body) + '\n}\n'


def _recon_src(n):
    mods = ALL_MODULI[:n]
    lines = []
    for j, p in enumerate(mods):
        h = (p + 1) // 2
        lines.append(f'int c{j} = C[(size_t){j} * plane + idx] % {p}; if (c{j} < 0) c{j} += {p}; if (c{j} >= {h}) c{j} -= {p};')
    lines.append('int v0 = c0;')
    for j in range(1, n):
        p = mods[j]
        h = (p + 1) // 2
        lines.append(f'int u{j} = v{j-1};')
        for i in range(j - 2, -1, -1):
            lines.append(f'u{j} = (u{j} * {mods[i] % p} + v{i}) % {p};')
        prod = 1
        for i in range(j):
            prod = (prod * mods[i]) % p
        inv = pow(prod, -1, p)
        lines.append(f'int d{j} = (c{j} - u{j}) % {p}; if (d{j} < 0) d{j} += {p};')
        lines.append(f'int v{j} = (d{j} * {inv}) % {p}; if (v{j} >= {h}) v{j} -= {p};')
    lines.append(f'double x = (double)v{n-1};')
    for j in range(n - 2, -1, -1):
        lines.append(f'x = x * {float(mods[j])!r} + (double)v{j};')
    return r'''
extern "C" __global__ void recon(const int* __restrict__ C, const int* __restrict__ ea, const int* __restrict__ eb,
                                 double* __restrict__ out, double* __restrict__ acc_in, int m, int n, int np_, size_t plane,
                                 int shift, int accumulate)
{
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (size_t)m * n) return;
    int i = (int)(gid / n), j = (int)(gid % n);
    size_t idx = (size_t)i * np_ + j;
    ''' + '\n    '.join(lines) + r'''
    double r = ldexp(x, ea[i] + eb[j] - shift);
    out[gid] = accumulate ? out[gid] + r : r;
}
'''


def _quant3t_src(n, wide):
    T = 'long long' if wide else 'int'
    pre = 'int xh = (int)(x >> 36), xm = (int)((x >> 18) & 0x3FFFF), xl = (int)(x & 0x3FFFF);' if wide else ''
    body = [_resid_code(i, p, wide, f'out[(size_t){i} * plane + base]') for i, p in enumerate(ALL_MODULI[:n])]
    return r'''
extern "C" __global__ void quant3t(const double* __restrict__ a, const int* __restrict__ ex, signed char* __restrict__ out,
                                   int d0, int d1, int d2, long long s0, long long s2, int P1, int P2, int q, size_t plane,
                                   int slab_stride)
{
    __shared__ ''' + T + r''' tile[32][33];
    int i = blockIdx.z;
    int c1b = blockIdx.y * 32, c2b = blockIdx.x * 32;
    for (int t = 0; t < 4; t++) {
        int c1 = c1b + threadIdx.x, c2 = c2b + threadIdx.y + 8 * t;
        ''' + T + r''' x = 0;
        if (c1 < d1 && c2 < d2) {
            double v = ldexp(a[(long long)i * s0 + c1 + (long long)c2 * s2], q - ex[(long long)i * slab_stride + c1]);
            x = (''' + T + r''')llrint(v);
        }
        tile[threadIdx.y + 8 * t][threadIdx.x] = x;
    }
    __syncthreads();
    for (int t = 0; t < 4; t++) {
        int c1 = c1b + threadIdx.y + 8 * t, c2 = c2b + threadIdx.x;
        if (c1 < P1 && c2 < P2) {
            ''' + T + r''' x = tile[threadIdx.x][threadIdx.y + 8 * t];
            size_t base = ((size_t)i * P1 + c1) * P2 + c2;
            ''' + pre + '\n            ' + '\n            '.join(body) + r'''
        }
    }
}
'''


_COLMAX_SRC = r'''
extern "C" __global__ void colmax(const double* __restrict__ a, unsigned long long* __restrict__ mx,
                                  int d1, int d2, long long s0, long long s2, int slab_stride)
{
    int c1 = blockIdx.x * blockDim.x + threadIdx.x;
    if (c1 >= d1) return;
    const double* p = a + (long long)blockIdx.y * s0 + c1;
    double m = 0.;
    for (int c2 = 0; c2 < d2; c2++) m = fmax(m, fabs(p[(long long)c2 * s2]));
    atomicMax(mx + (long long)blockIdx.y * slab_stride + c1, (unsigned long long)__double_as_longlong(m));
}
'''


def _get(kind, n, wide=False):
    key = (kind, n, wide)
    k = _kern.get(key)
    if k is None:
        if kind == 'q':
            k = cupy.RawKernel(_quant_src(n, wide), 'quant')
        elif kind == 't':
            k = cupy.RawKernel(_quant3t_src(n, wide), 'quant3t')
        elif kind == 'cm':
            k = cupy.RawKernel(_COLMAX_SRC, 'colmax')
        else:
            k = cupy.RawKernel(_recon_src(n), 'recon')
        _kern[key] = k
    return k


def _pad16(n):
    return (n + 15) // 16 * 16


_ROWEXP_SRC = r'''
extern "C" __global__ void rowexp(const double* __restrict__ a, int* __restrict__ e, int ncomp, int rows, int ng,
                                  const int* __restrict__ cmap)
{
    int row = blockIdx.x;
    double m = 0.;
    for (int c = 0; c < ncomp; c++) {
        const double* p = a + ((size_t)cmap[c] * rows + row) * ng;
        for (int g = threadIdx.x; g < ng; g += blockDim.x) m = fmax(m, fabs(p[g]));
    }
    for (int o = 16; o > 0; o >>= 1) m = fmax(m, __shfl_down_sync(0xffffffff, m, o));
    __shared__ double sm[32];
    if ((threadIdx.x & 31) == 0) sm[threadIdx.x >> 5] = m;
    __syncthreads();
    if (threadIdx.x < 32) {
        m = threadIdx.x < (blockDim.x >> 5) ? sm[threadIdx.x] : 0.;
        for (int o = 16; o > 0; o >>= 1) m = fmax(m, __shfl_down_sync(0xffffffff, m, o));
        if (threadIdx.x == 0) {
            int ex = 0;
            if (m > 0.) frexp(m, &ex);
            e[row] = ex;
        }
    }
}
'''
_rowexp_kernel = None


_cmaps = {}


def _cmap(comps, ncomp):
    key = tuple(range(ncomp)) if comps is None else tuple(comps)
    m = _cmaps.get(key)
    if m is None:
        m = _cmaps[key] = cupy.asarray(np.asarray(key, dtype=np.int32))
    return m


def _row_exponent(src, comps=None):
    '''src [ncomp, rows, ng] -> int32 exponent e with max|src[comps, r, :]| < 2^e (comps: component subset).'''
    global _rowexp_kernel
    if _rowexp_kernel is None:
        _rowexp_kernel = cupy.RawKernel(_ROWEXP_SRC, 'rowexp')
    ncomp, rows, ng = src.shape
    nc = ncomp if comps is None else len(comps)
    e = cupy.empty(rows, dtype=cupy.int32)
    _rowexp_kernel((rows,), (256,), (src, e, np.int32(nc), np.int32(rows), np.int32(ng), _cmap(comps, ncomp)))
    return e


class Operand:
    '''Residues of an operand given as float64 [ncomp, rows, ng] (reduction axis = comp x ng).'''

    def __init__(self, src, nmod, q, e=None, comps=None):
        '''comps: use only these components of src (no copy), in this order.'''
        assert src.dtype == cupy.float64 and src.flags.c_contiguous
        ncomp_src, rows, ng = src.shape
        ncomp = ncomp_src if comps is None else len(comps)
        self.rows, self.nmod, self.q = rows, nmod, q
        self.rp, self.kp = _pad16(rows), _pad16(ncomp * ng)
        self.e = _row_exponent(src, comps) if e is None else e
        ex = cupy.zeros(self.rp, dtype=cupy.int32)
        ex[:rows] = self.e
        self.planes = cupy.empty((nmod, self.rp, self.kp), dtype=cupy.int8)
        tot = self.rp * self.kp
        thr = 256
        _get('q', nmod, q > 30)(((tot + thr - 1) // thr,), (thr,),
                                (src, ex, self.planes, np.int32(ncomp), np.int32(rows), np.int32(ng),
                                 np.int32(self.rp), np.int32(self.kp), np.int32(q), np.uint64(tot),
                                 _cmap(comps, ncomp_src)))


class PlanesT:
    '''Residues of a strided 3-D float64 view src (d0, d1, d2) whose axis 1 is the memory-contiguous one
    (stride 1), laid out as planes [nmod][d0][P1][P2] (d1, d2 zero-padded to multiples of 16): the GEMM reduction
    axis is d2, the rows are (d0, d1) and every row index d1 has its own power-of-two scale (max over d0 and d2).
    The memory-layout transposition (d1 contiguous in src -> d2 contiguous in the planes) happens in the quantize
    kernel via shared-memory tiles, so e.g. the DF tensor [i, j, L] (L contiguous) feeds the TN INT8 GEMM directly.'''

    def __init__(self, src, nmod, q, compact=False, per_slab=False):
        '''per_slab: every (i, l) row gets its own exponent (max over d2 only) instead of sharing it over i.'''
        d0, d1, d2 = src.shape
        isz = src.itemsize
        assert src.dtype == cupy.float64 and src.strides[1] == isz
        s0, s2 = src.strides[0] // isz, src.strides[2] // isz
        self.nmod, self.q = nmod, q
        P1, P2 = _pad16(d1), _pad16(d2)
        nslab = d0 if per_slab else 1
        sstride = P1 if per_slab else 0
        mx = cupy.zeros(nslab * P1, dtype=cupy.uint64)
        thr = 128
        _get('cm', 0)(((d1 + thr - 1) // thr, d0), (thr,),
                      (src, mx, np.int32(d1), np.int32(d2), np.int64(s0), np.int64(s2), np.int32(sstride)))
        amax = mx.view(cupy.float64)
        _, e = cupy.frexp(amax)
        ex = cupy.where(amax == 0, 0, e).astype(cupy.int32)
        self.rows, self.rp, self.kp = d0 * P1, d0 * P1, P2
        self.e = ex if per_slab else cupy.tile(ex, d0)       # exponent of row (i, l)
        if compact:                                        # d0 == 1: no zero-padded rows in the result
            assert d0 == 1
            self.rows = d1
        planes = cupy.empty((nmod, d0, P1, P2), dtype=cupy.int8)
        _get('t', nmod, q > 30)(((P2 + 31) // 32, (P1 + 31) // 32, d0), (32, 8),
                                (src, ex, planes, np.int32(d0), np.int32(d1), np.int32(d2), np.int64(s0), np.int64(s2),
                                 np.int32(P1), np.int32(P2), np.int32(q), np.uint64(d0 * P1 * P2), np.int32(sstride)))
        self.planes = planes.reshape(nmod, d0 * P1, P2)


def int8_mm_nt(a, bt, out):
    '''int32 out (m, n) = a (m, k) @ bt (n, k)^T for contiguous int8 arrays (cublasGemmEx, TN).'''
    from cupy.cuda import cublas, device
    m, k = a.shape
    n = bt.shape[0]
    alpha = np.array(1, dtype=np.int32)
    beta = np.array(0, dtype=np.int32)
    h = device.get_cublas_handle()
    cublas.gemmEx(h, cublas.CUBLAS_OP_T, cublas.CUBLAS_OP_N, n, m, k,
                  alpha.ctypes.data, bt.data.ptr, 3, k, a.data.ptr, 3, k,
                  beta.ctypes.data, out.data.ptr, 10, n, 72, -1)
    return out


def int8_mm_nt_batched(a, bt, out):
    '''out[b] (m, n) int32 = a[b] (m, k) @ bt[b] (n, k)^T for contiguous int8 stacks (one strided-batched call:
    fills the GPU when m, n are only a few hundred).'''
    from cupy.cuda import cublas, device
    nb, m, k = a.shape
    n = bt.shape[1]
    alpha = np.array(1, dtype=np.int32)
    beta = np.array(0, dtype=np.int32)
    h = device.get_cublas_handle()
    cublas.gemmStridedBatchedEx(h, cublas.CUBLAS_OP_T, cublas.CUBLAS_OP_N, n, m, k,
                                alpha.ctypes.data, bt.data.ptr, 3, k, n * k, a.data.ptr, 3, k, m * k,
                                beta.ctypes.data, out.data.ptr, 10, n, m * n, nb, 72, -1)
    return out


def matmul_operands(oa, ob, out, accumulate=False):
    '''out (rows_a, rows_b) float64 (+)= A B^T from two Operand objects with equal nmod, q and kp.'''
    assert oa.nmod == ob.nmod and oa.q == ob.q and oa.kp == ob.kp
    n, q = oa.nmod, oa.q
    C = cupy.empty((n, oa.rp, ob.rp), dtype=cupy.int32)
    int8_mm_nt_batched(oa.planes, ob.planes, C)
    ea = cupy.zeros(oa.rp, dtype=cupy.int32)
    ea[:oa.rows] = oa.e[:oa.rows]
    eb = cupy.zeros(ob.rp, dtype=cupy.int32)
    eb[:ob.rows] = ob.e[:ob.rows]
    tot = oa.rows * ob.rows
    thr = 256
    _get('r', n)(((tot + thr - 1) // thr,), (thr,),
                 (C, ea, eb, out, out, np.int32(oa.rows), np.int32(ob.rows), np.int32(ob.rp),
                  np.uint64(oa.rp * ob.rp), np.int32(2 * q), np.int32(1 if accumulate else 0)))
    return out


def gemm_nt(a, b, nmod, out=None, qmax=QMAX):
    '''A (ncomp, m, ng) , B (ncomp, n, ng) -> out (m, n) = sum_c A[c] B[c]^T in ~FP64 accuracy (K = ncomp*ng).
    Chunks the grid axis when ncomp*ng exceeds the int32-accumulation limit.'''
    if a.ndim == 2:
        a, b = a[None], b[None]
    ncomp, m, ng = a.shape
    n = b.shape[1]
    if out is None:
        out = cupy.empty((m, n), dtype=cupy.float64)
    step = max(1, MAX_K // ncomp)
    first = True
    for g0 in range(0, ng, step):
        g1 = min(ng, g0 + step)
        asub = a[..., g0:g1] if g1 - g0 != ng else a
        bsub = b[..., g0:g1] if g1 - g0 != ng else b
        if not asub.flags.c_contiguous:
            asub = cupy.ascontiguousarray(asub)
        if not bsub.flags.c_contiguous:
            bsub = cupy.ascontiguousarray(bsub)
        q = plan(ncomp * (g1 - g0), nmod, qmax)
        oa = Operand(asub, nmod, q)
        ob = Operand(bsub, nmod, q)
        matmul_operands(oa, ob, out, accumulate=not first)
        first = False
    return out


def gemm_planes(src3, bt, nmod, out=None, qmax=QMAX):
    '''out (d0 * P1, nb) float64 = sum_{d2} src3[i, l, d2] * bt[b, d2]   (rows ordered (i, l), l zero-padded to P1).
    src3: strided float64 view (d0, d1, d2) with stride-1 axis 1 (see PlanesT), bt: (nb, d2) float64 contiguous.'''
    d0, d1, d2 = src3.shape
    assert d2 <= MAX_K
    q = plan(d2, nmod, qmax)
    oa = PlanesT(src3, nmod, q)
    ob = Operand(bt[None], nmod, q)
    if out is None:
        out = cupy.empty((oa.rows, ob.rows), dtype=cupy.float64)
    matmul_operands(oa, ob, out)
    return out, _pad16(d1)


def gemm_dense(a, b, nmod, out=None, qmax=QMAX, budget=1.0e9):
    '''out (m, n) float64 = a (m, K) @ b (K, n) from INT8 residue GEMMs.  The rows of `a` are processed in chunks
    that keep the residue planes under `budget` bytes; the planes of `b` (static for the whole call) are built once.'''
    m, K = a.shape
    n = b.shape[1]
    assert K <= MAX_K
    if out is None:
        out = cupy.empty((m, n), dtype=cupy.float64)
    if not a.flags.c_contiguous:
        a = cupy.ascontiguousarray(a)
    q = plan(K, nmod, qmax)
    isz = b.itemsize
    if b.strides[1] == isz:                                   # C order [K][n]: transposing quantization
        view = cupy.lib.stride_tricks.as_strided(b, shape=(1, n, K), strides=(0, isz, b.strides[0]))
        ob = PlanesT(view, nmod, q, compact=True)
    else:                                                     # F order: b.T is C-contiguous [n][K]
        ob = Operand(cupy.ascontiguousarray(b.T)[None], nmod, q)
    kp = _pad16(K)
    rc = max(16, int(budget // (nmod * kp)) // 16 * 16)
    for r0 in range(0, m, rc):
        r1 = min(m, r0 + rc)
        oa = Operand(a[r0:r1][None], nmod, q)
        matmul_operands(oa, ob, out[r0:r1])
    return out


_RHO_SRC = r'''
extern "C" __global__ void rho_cT(double* __restrict__ rho, const double* __restrict__ c, long long ngp, long long nocc,
                                  long long ng, long long mode)
{
    // c: [ncomp][ngp][nocc] (rows = grid points); one warp per grid point; rho: [nvar][ng]
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    long long g = (long long)blockIdx.x * 4 + warp;
    if (g >= ng) return;
    double v0 = 0, v1 = 0, v2 = 0, v3 = 0, v4 = 0;
    const double* p0 = c + g * nocc;
    long long cs = ngp * nocc;
    for (long long i = lane; i < nocc; i += 32) {
        double c0 = p0[i];
        v0 += c0 * c0;
        if (mode >= 1) {
            double c1 = p0[i + cs], c2 = p0[i + 2 * cs], c3 = p0[i + 3 * cs];
            v1 += c0 * c1; v2 += c0 * c2; v3 += c0 * c3;
            v4 += c1 * c1 + c2 * c2 + c3 * c3;
        }
    }
    for (int o = 16; o > 0; o >>= 1) {
        v0 += __shfl_down_sync(0xffffffff, v0, o);
        v1 += __shfl_down_sync(0xffffffff, v1, o);
        v2 += __shfl_down_sync(0xffffffff, v2, o);
        v3 += __shfl_down_sync(0xffffffff, v3, o);
        v4 += __shfl_down_sync(0xffffffff, v4, o);
    }
    if (lane == 0) {
        rho[g] = v0;
        if (mode >= 1) { rho[ng + g] = 2 * v1; rho[2 * ng + g] = 2 * v2; rho[3 * ng + g] = 2 * v3; }
        if (mode >= 2) rho[4 * ng + g] = 0.5 * v4;
    }
}
'''
_rho_kernel = None


def rho_from_cT(c, ng, xctype):
    '''rho [nvar, ng] from c [ncomp, ngp, nocc] (INT8-GEMM output layout, grid points as rows).'''
    global _rho_kernel
    if _rho_kernel is None:
        _rho_kernel = cupy.RawKernel(_RHO_SRC, 'rho_cT')
    mode = {'LDA': 0, 'GGA': 1, 'NLC': 1, 'MGGA': 2}[xctype]
    nvar = {0: 1, 1: 4, 2: 5}[mode]
    rho = cupy.empty((nvar, ng))
    _, ngp, nocc = c.shape
    _rho_kernel(((ng + 3) // 4,), (128,), (rho, c, np.int64(ngp), np.int64(nocc), np.int64(ng), np.int64(mode)))
    return rho


# ------------------------------------------------------------------ Ozaki-I with INT8 digits (short reductions)
# For GEMMs with a short reduction (K ~ nocc) the CRT reconstruction per output costs as much as the product.  Ozaki
# scheme I with balanced base-2^7 digits needs no reconstruction: x = 2^e sum_s d_s 2^(-6-7s), |d_s| <= 64; the
# digit products of one group g = s + t are summed exactly in int32 by ONE INT8 GEMM over the concatenated K axis
# ([a_0 .. a_g] . [b_g .. b_0]), and the result is sum_g 2^(ea+eb-12-7g) C_g.  Exact int32 accumulation needs
# (g+1) * K * 64^2 < 2^31, i.e. K * (g+1) < 524288.  `ndig` digits give ~6 + 7 (ndig-1) bits per operand.
_OZ1_SLICE_SRC = r'''
extern "C" __global__ void oz1_slice(const double* __restrict__ a, const int* __restrict__ ex,
                                     signed char* __restrict__ out, int rows, int k, int kp, int ndig, size_t plane)
{
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (size_t)rows * kp) return;
    int row = (int)(gid / kp), col = (int)(gid % kp);
    double v = (col < k) ? ldexp(a[(size_t)row * k + col], 6 - ex[row]) : 0.;
    for (int s = 0; s < ndig; s++) {
        double d = rint(v);
        out[(size_t)s * plane + gid] = (signed char)d;
        v = (v - d) * 128.;
    }
}
'''
_OZ1_COMBINE_SRC = r'''
extern "C" __global__ void oz1_combine(const int* __restrict__ C, const int* __restrict__ ea, const int* __restrict__ eb,
                                       double* __restrict__ out, int m, int n, int np_, int ngrp, size_t plane,
                                       double alpha, double beta, long long ldo)
{
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (size_t)m * n) return;
    int i = (int)(gid / n), j = (int)(gid % n);
    size_t idx = (size_t)i * np_ + j;
    double x = 0.;
    for (int g = ngrp - 1; g >= 0; g--) x = x * (1. / 128.) + (double)C[(size_t)g * plane + idx];
    x = ldexp(x, ea[i] + eb[j] - 12);
    size_t o = (size_t)i * ldo + j;
    out[o] = (beta == 0.) ? alpha * x : alpha * x + beta * out[o];
}
'''


def _oz1_slices(a, ndig):
    '''a (rows, k) float64 C-contiguous -> (digits int8 [ndig, rows_pad, k_pad], exponents int32 [rows_pad])'''
    rows, k = a.shape
    rp, kp = _pad16(rows), _pad16(k)
    e = _row_exponent(a[None])
    ex = cupy.zeros(rp, dtype=cupy.int32)
    ex[:rows] = e
    dig = cupy.zeros((ndig, rp, kp), dtype=cupy.int8)
    k_ = _kern.get('oz1s')
    if k_ is None:
        k_ = _kern['oz1s'] = cupy.RawKernel(_OZ1_SLICE_SRC, 'oz1_slice')
    tot = rows * kp
    k_(((tot + 255) // 256,), (256,), (a, ex, dig, np.int32(rows), np.int32(k), np.int32(kp), np.int32(ndig),
                                      np.uint64(rp * kp)))
    return dig, ex


def ozaki1_gemm_nt(a, bt, ndig=6, out=None, alpha=1.0, beta=0.0, bslices=None):
    '''out (m, n) = alpha * a (m, k) @ bt (n, k)^T + beta * out, FP64-class (~6 + 7 (ndig-1) bits per operand) from
    exact INT8 tensor-core products (Ozaki scheme I, one GEMM per digit group).  For short k (k * ndig < 524288).
    bslices: precomputed _oz1_slices(bt, ndig) to reuse one operand.  out may be a column block of a C-contiguous
    matrix (row stride out.strides[0]).'''
    m, k = a.shape
    n = bt.shape[0]
    assert k * ndig < 524288
    da, ea = _oz1_slices(cupy.ascontiguousarray(a), ndig)
    db, eb = bslices if bslices is not None else _oz1_slices(cupy.ascontiguousarray(bt), ndig)
    mp, kp = da.shape[1], da.shape[2]
    npd = db.shape[1]
    C = cupy.empty((ndig, mp, npd), dtype=cupy.int32)
    for g in range(ndig):
        A = cupy.ascontiguousarray(da[:g + 1].transpose(1, 0, 2).reshape(mp, (g + 1) * kp))
        B = cupy.ascontiguousarray(db[g::-1].transpose(1, 0, 2).reshape(npd, (g + 1) * kp))
        int8_mm_nt(A, B, C[g])
    if out is None:
        out = cupy.empty((m, n))
        beta = 0.0
    assert out.dtype == cupy.float64 and out.strides[1] == 8
    ldo = out.strides[0] // 8
    kc = _kern.get('oz1c')
    if kc is None:
        kc = _kern['oz1c'] = cupy.RawKernel(_OZ1_COMBINE_SRC, 'oz1_combine')
    tot = m * n
    kc(((tot + 255) // 256,), (256,), (C, ea, eb, out, np.int32(m), np.int32(n), np.int32(npd), np.int32(ndig),
                                      np.uint64(mp * npd), np.float64(alpha), np.float64(beta), np.int64(ldo)))
    return out
