# Copyright 2021-2026 The PySCF Developers. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.
'''
Ozaki-scheme-II style FP64 GEMM emulation on INT8 tensor cores (own implementation, CRT / residue number system).

  A (m,k) ~ diag(2^ea) * Ai / 2^q,  B (k,n) ~ Bi / 2^q * diag(2^eb)   (Ai, Bi integers, |.| < 2^q)
  C_int = Ai @ Bi  (exact integer, |C_int| < k 2^{2q})
  C_int mod p_i  =  (Ai mod p_i)(Bi mod p_i) mod p_i,  p_i <= 256 pairwise coprime, residues symmetric in int8,
  every residue GEMM is an exact int8 x int8 -> int32 product (k * 128^2 < 2^31, i.e. k < 131k),
  C_int is recovered from its residues by Garner's (balanced mixed-radix) CRT, evaluated in double at the end.

Needs  prod(p_i) > 2 k 2^{2q}.  Number of moduli = precision knob (about 8 bits each).
One fused kernel quantizes + reduces an operand to all residues (read FP64 once, write n int8 planes), one fused
kernel reconstructs C (read n int32 planes, write FP64).  The int8 GEMM itself is the only tensor-core work.

Generated NVRTC sources bake the moduli and Garner constants as literals so that `%` compiles to mul-hi sequences.
'''
import numpy as np
import cupy

ALL_MODULI = [256, 255, 253, 251, 247, 241, 239, 233, 229, 227, 223, 211, 199, 197, 193, 191, 181, 179, 173, 167]


def _sym(r, p):
    '''balanced residue in [-(p//2) .. (p-1)//2 + (p even)] : [-128,127] for 256, [-(p-1)/2,(p-1)/2] for odd p'''
    r %= p
    return r - p if r >= (p + 1) // 2 else r


def plan_moduli(k, q, margin_bits=1):
    '''smallest n such that prod(p) > 2^margin * k * 2^{2q}'''
    need = np.log2(k) + 2 * q + margin_bits
    tot, n = 0.0, 0
    while tot <= need:
        tot += np.log2(ALL_MODULI[n])
        n += 1
    return n


_kern_cache = {}


def _quant_kernel(n, wide):
    key = ('q', n, wide)
    if key in _kern_cache:
        return _kern_cache[key]
    mods = ALL_MODULI[:n]
    T = 'long long' if wide else 'int'
    body = []
    for i, p in enumerate(mods):
        h = (p + 1) // 2
        body.append(f'{{ {T} r = x % {p}; if (r < 0) r += {p}; if (r >= {h}) r -= {p}; out[(size_t){i} * plane + idx] = (signed char)r; }}')
    src = r'''
extern "C" __global__ void quant(const double* __restrict__ a, const int* __restrict__ ex, signed char* __restrict__ out,
                                 int m, int k, int kp, int q, size_t plane, int rowmajor_scale)
{
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t tot = (size_t)m * kp;
    if (gid >= tot) return;
    int row = (int)(gid / kp), col = (int)(gid % kp);
    size_t idx = gid;
    ''' + T + r''' x = 0;
    if (col < k) {
        double v = ldexp(a[(size_t)row * k + col], q - ex[row]);
        x = (''' + T + r''')llrint(v);
    }
    ''' + '\n    '.join(body) + r'''
}
'''
    k_ = cupy.RawKernel(src, 'quant')
    _kern_cache[key] = k_
    return k_


def _recon_kernel(n):
    key = ('r', n)
    if key in _kern_cache:
        return _kern_cache[key]
    mods = ALL_MODULI[:n]
    lines = []
    # c_j = C_j mod p_j (balanced), then Garner digits v_j
    for j, p in enumerate(mods):
        h = (p + 1) // 2
        lines.append(f'int c{j} = C[(size_t){j} * plane + idx] % {p}; if (c{j} < 0) c{j} += {p}; if (c{j} >= {h}) c{j} -= {p};')
    lines.append('int v0 = c0;')
    for j in range(1, n):
        p = mods[j]
        h = (p + 1) // 2
        # u = v0 + p0 (v1 + p1 (v2 + ...)) mod p_j  evaluated Horner from the top, all mod p_j
        expr = f'v{j-1}'
        lines.append(f'int u{j} = v{j-1};')
        for i in range(j - 2, -1, -1):
            lines.append(f'u{j} = (u{j} * {mods[i] % p} + v{i}) % {p};')
        # inverse of prod_{i<j} p_i modulo p_j
        prod = 1
        for i in range(j):
            prod = (prod * mods[i]) % p
        inv = pow(prod, -1, p)
        lines.append(f'int d{j} = (c{j} - u{j}) % {p}; if (d{j} < 0) d{j} += {p};')
        lines.append(f'int v{j} = (d{j} * {inv}) % {p}; if (v{j} >= {h}) v{j} -= {p};')
    lines.append(f'double x = (double)v{n-1};')
    for j in range(n - 2, -1, -1):
        lines.append(f'x = x * {float(mods[j])!r} + (double)v{j};')
    src = r'''
extern "C" __global__ void recon(const int* __restrict__ C, const int* __restrict__ ea, const int* __restrict__ eb,
                                 double* __restrict__ out, int m, int n, int np_, size_t plane, int shift)
{
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (size_t)m * n) return;
    int i = (int)(gid / n), j = (int)(gid % n);
    size_t idx = (size_t)i * np_ + j;
    ''' + '\n    '.join(lines) + r'''
    out[gid] = ldexp(x, ea[i] + eb[j] - shift);
}
'''
    k_ = cupy.RawKernel(src, 'recon')
    _kern_cache[key] = k_
    return k_


def _pad(n, mult=16):
    return (n + mult - 1) // mult * mult


def int8_mm_nt(a, bt, out):
    '''int32 out (m, n) = a (m, k) @ bt (n, k)^T for contiguous int8, cublasGemmEx TN (dp4a on Turing, TC on Ampere+).'''
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


class Residues:
    '''Operand with rows scaled by 2^-e and reduced to n int8 residue planes [n, rows_pad, k_pad].  Rows of A, or rows of B^T.'''

    def __init__(self, M, n, q):
        '''M: (rows, k) float64 contiguous.'''
        rows, k = M.shape
        self.rows, self.k, self.n, self.q = rows, k, n, q
        amax = cupy.abs(M).max(axis=1)
        _, e = cupy.frexp(amax)                     # amax < 2^e
        e = e.astype(cupy.int32)
        e = cupy.where(amax == 0, 0, e).astype(cupy.int32)
        self.e = e
        rp, kp = _pad(rows), _pad(k)
        self.rp, self.kp = rp, kp
        wide = q > 30
        self.planes = cupy.empty((n, rp, kp), dtype=cupy.int8)
        ex = cupy.zeros(rp, dtype=cupy.int32)
        ex[:rows] = e
        tot = rp * kp
        thr = 256
        _quant_kernel(n, wide)(((tot + thr - 1) // thr,), (thr,),
                               (M, ex, self.planes, np.int32(rp), np.int32(k), np.int32(kp), np.int32(q), np.uint64(rp * kp), np.int32(0)))


def crt_matmul(ra, rb, out=None, timers=None):
    '''FP64-class A @ B from residue operands ra (rows of A) and rb (rows of B^T).'''
    assert ra.n == rb.n and ra.q == rb.q and ra.kp == rb.kp
    n, q = ra.n, ra.q
    m, nn = ra.rows, rb.rows
    mp, np_ = ra.rp, rb.rp
    C = cupy.empty((n, mp, np_), dtype=cupy.int32)
    for i in range(n):
        int8_mm_nt(ra.planes[i], rb.planes[i], C[i])
    res = out if out is not None else cupy.empty((m, nn), dtype=cupy.float64)
    ea = cupy.zeros(mp, dtype=cupy.int32); ea[:m] = ra.e
    eb = cupy.zeros(np_, dtype=cupy.int32); eb[:nn] = rb.e
    tot = m * nn
    thr = 256
    _recon_kernel(n)(((tot + thr - 1) // thr,), (thr,),
                     (C, ea, eb, res, np.int32(m), np.int32(nn), np.int32(np_), np.uint64(mp * np_), np.int32(2 * q)))
    return res


def crt_matmul_nt(A, Bt, n=None, q=None):
    '''A (m,k) @ Bt (n,k)^T, all steps (quantize both, int8 GEMMs, reconstruct).'''
    k = A.shape[1]
    if q is None:
        q = 30
    if n is None:
        n = plan_moduli(k, q)
    ra = Residues(A, n, q)
    rb = Residues(Bt, n, q)
    return crt_matmul(ra, rb)
