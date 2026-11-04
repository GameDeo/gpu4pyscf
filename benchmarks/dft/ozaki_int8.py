# Copyright 2021-2026 The PySCF Developers. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.
'''
Ozaki-scheme-I FP64 GEMM emulation on INT8 tensor cores (prototype, own implementation).

  A (m,k) = diag(mu) * sum_s 2^{-7(s+1)} A_s,   B (k,n) = (sum_t 2^{-7(t+1)} B_t) * diag(nu)
  with mu, nu powers of two (row / column maxima) and A_s, B_t int8 slices in [-127, 127],
  so   A B = diag(mu) [ sum_{s+t<S} 2^{-7(s+t+2)} (A_s B_t) ] diag(nu)
  where every A_s B_t is an exact int32 product (k * 127^2 < 2^31, i.e. k < 133k).  Only the S(S+1)/2
  terms with s+t < S are kept; the neglected terms are <= 2^{-7S} of the leading one, so S slices
  give ~7S bits (S=5: 35 bits, S=6: 42, S=7: 49 bits, S=8: 56 bits).

Because the slices of a *static* operand (the DF tensor) can be computed once and stored (S bytes per
element, ~ the size of the FP64 tensor at S=8), the precision of a contraction is chosen at run time by the
number of slices that are used: a "precision ladder" without recomputing anything.
'''
import numpy as np
import cupy

_acc = cupy.ElementwiseKernel('int32 c, float64 sc', 'float64 acc', 'acc += (double)c * sc', 'oz_acc')


def _pad(n, mult=8):
    return (n + mult - 1) // mult * mult


def slice_rows(A, S):
    '''Row-scaled int8 slicing of A (m,k) float64. Returns (mu (m,1), [S int8 arrays (m,k)]).'''
    amax = cupy.abs(A).max(axis=1, keepdims=True)
    _, e = cupy.frexp(amax)                  # amax = f * 2^e with f in [0.5, 1)  ->  amax < 2^e
    mu = cupy.exp2(e.astype(cupy.float64))
    x = A / mu                               # |x| < 1, exact
    out = []
    for _ in range(S):
        x = x * 128.0
        q = cupy.trunc(x)
        x = x - q
        out.append(q.astype(cupy.int8))
    return mu, out


def int8_mm_nt(a, bt, out=None):
    '''int32 (m, n) = a (m, k) @ bt (n, k)^T for contiguous int8 arrays, via cublasGemmEx (TN layout).
    m, n, k must be multiples of 16 (required on Turing).  Plain layouts work on every GPU generation with INT8 support
    (dp4a on Turing, tensor cores on Ampere and newer) - unlike cublasLt, which needs special
    data orders on sm_75.'''
    from cupy.cuda import cublas, device
    m, k = a.shape
    n = bt.shape[0]
    assert bt.shape[1] == k and m % 16 == 0 and n % 16 == 0 and k % 16 == 0
    if out is None:
        out = cupy.empty((m, n), dtype=cupy.int32)
    alpha = np.array(1, dtype=np.int32)
    beta = np.array(0, dtype=np.int32)
    h = device.get_cublas_handle()
    # row-major C(m,n) = A(m,k) Bt(n,k)^T  <=>  column-major C^T(n,m) = op_T(Bt_cm(k,n)) . A_cm(k,m)
    cublas.gemmEx(h, cublas.CUBLAS_OP_T, cublas.CUBLAS_OP_N, n, m, k,
                  alpha.ctypes.data, bt.data.ptr, 3, k, a.data.ptr, 3, k,
                  beta.ctypes.data, out.data.ptr, 10, n, 72, -1)     # CUDA_R_8I=3, CUDA_R_32I=10, COMPUTE_32I=72
    return out


class Sliced:
    '''Pre-sliced, padded operand: slices stored as row-major int8 [rows, k_pad]
    (side 'A': rows of A; side 'B': rows of B^T, i.e. columns of B).'''

    def __init__(self, M, S, side):
        self.side, self.S = side, S
        self.shape = M.shape
        if side == 'A':
            mu, sl = slice_rows(M, S)
            m, k = M.shape
            self.scale = mu                                   # (m, 1)
        else:
            nu, sl = slice_rows(cupy.ascontiguousarray(M.T), S)
            k, n = M.shape
            self.scale = nu.T                                 # (1, n)
            m = n
        mp, kp = _pad(m, 16), _pad(k, 16)       # cublasGemmEx INT8 on sm_75 needs m, n, k % 16 == 0
        self.t = [cupy.ascontiguousarray(cupy.pad(s_, ((0, mp - m), (0, kp - k)))) for s_ in sl]


def matmul_sliced(sa, sb, S, out=None):
    '''FP64-class product of two pre-sliced operands using S slices (S <= available).'''
    m, k = sa.shape
    n = sb.shape[1]
    acc = cupy.zeros((sa.t[0].shape[0], sb.t[0].shape[0]), dtype=cupy.float64)
    buf = cupy.empty(acc.shape, dtype=cupy.int32)
    for s in range(S):
        for t in range(S - s):
            int8_mm_nt(sa.t[s], sb.t[t], out=buf)
            _acc(buf, float(2.0 ** (-7 * (s + t + 2))), acc)
    return acc[:m, :n] * sa.scale * sb.scale


def ozaki_matmul(A, B, S):
    return matmul_sliced(Sliced(A, S, 'A'), Sliced(B, S, 'B'), S)
