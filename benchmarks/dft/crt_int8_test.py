#!/usr/bin/env python
'''Accuracy + timing of the INT8 CRT GEMM emulation (crt_int8.py) on synthetic operands with AO-like dynamic range.
usage: crt_int8_test.py [--m 624] [--k 61440] [--q 30]'''
import argparse
import time
import numpy as np
import cupy
import crt_int8 as ci


def sync():
    cupy.cuda.Stream.null.synchronize()


def timeit(fn, reps=5):
    fn(); sync()
    best = 1e9
    for _ in range(reps):
        t = time.perf_counter(); fn(); sync(); best = min(best, time.perf_counter() - t)
    return best * 1e3


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--m', type=int, default=624)
    p.add_argument('--k', type=int, default=61440)
    p.add_argument('--q', type=int, default=30)
    p.add_argument('--skip-fp64', action='store_true')
    args = p.parse_args()
    rng = np.random.default_rng(0)
    m, k = args.m, args.k
    # AO-like: values spanning many orders of magnitude (exp of gaussian-distributed exponent), random signs
    A = (rng.standard_normal((m, k)) * np.exp(-np.abs(rng.standard_normal((m, k))) * 3)).astype(np.float64)
    B = (rng.standard_normal((m, k)) * np.exp(-np.abs(rng.standard_normal((m, k))) * 3)).astype(np.float64)
    Ag, Bg = cupy.asarray(A), cupy.asarray(B)
    # reference in extended precision on a sub-block, plus full FP64 on GPU
    sub = 48
    ref = (A[:sub].astype(np.longdouble) @ B[:sub].astype(np.longdouble).T).astype(np.float64)
    scale = np.abs(ref).max()
    print(f'# m=n={m} k={k}  q={args.q}')
    if not args.skip_fp64:
        t64 = timeit(lambda: Ag @ Bg.T, 3)
        print(f'native FP64 GEMM: {t64:.1f} ms  ({2 * m * m * k / t64 / 1e9:.3f} TFLOP/s)')
    else:
        t64 = float('nan')
    for n in (5, 6, 7, 8, 9, 10, 12):
        if n > len(ci.ALL_MODULI):
            continue
        q = args.q
        # moduli product must cover 2 k 2^{2q}
        bits = sum(np.log2(ci.ALL_MODULI[:n]))
        if bits < 2 * q + np.log2(k) + 1:
            q = int((bits - np.log2(k) - 1) // 2)
        C = ci.crt_matmul_nt(Ag, Bg, n=n, q=q)
        sync()
        err = np.abs(cupy.asnumpy(C[:sub, :sub]) - ref[:, :sub]).max() / scale
        ra = ci.Residues(Ag, n, q); rb = ci.Residues(Bg, n, q)
        t_quant = timeit(lambda: (ci.Residues(Ag, n, q), ci.Residues(Bg, n, q)))
        out = cupy.empty((m, m))
        t_all = timeit(lambda: ci.crt_matmul(ra, rb, out=out))
        # GEMM only
        mp, kp = ra.rp, ra.kp
        Cb = cupy.empty((mp, mp), dtype=cupy.int32)
        t_gemm = timeit(lambda: [ci.int8_mm_nt(ra.planes[i], rb.planes[i], Cb) for i in range(n)])
        t_rec = t_all - t_gemm
        tot = t_quant + t_all
        print(f'n={n:2d} q={q:2d} maxrelerr={err:.2e} | quant(2 ops) {t_quant:6.1f} ms  gemm {t_gemm:6.1f} ms  recon {t_rec:5.1f} ms  total {tot:6.1f} ms'
              f'  speedup vs FP64 {t64 / tot:5.1f}x   int8 GEMM {2 * m * m * k * n / t_gemm / 1e9:.1f} TOPS')


if __name__ == '__main__':
    main()
