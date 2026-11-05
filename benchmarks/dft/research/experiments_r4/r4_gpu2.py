'''Round-4 GPU micro-tests (T4):
G2  symmetric eigensolver: FP64 syevd vs FP32 syevd + Ogita-Aishima refinement (FP64 GEMMs) on a Fock-like matrix
G3  FP16 tensor-core GEMM throughput on the shapes of an exchange increment K(dD) with a rank-r factor
'''
import time, json, os
import numpy as np
import cupy

here = os.path.dirname(os.path.abspath(__file__))
OUT = {}


def tm(f, n=3):
    f()
    cupy.cuda.Stream.null.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        r = f()
    cupy.cuda.Stream.null.synchronize()
    return (time.perf_counter() - t) / n, r


n, nocc = 623, 61
rng = np.random.default_rng(0)
# Fock-like spectrum: core (-20..-1), valence occupied (-1..-0.2), gap, virtual (0.05..30, dense at low end)
ev = np.concatenate([-20 * rng.random(10) - 1, -0.8 * rng.random(nocc - 10) - 0.2,
                     np.sort(np.exp(rng.uniform(np.log(0.05), np.log(30), n - nocc)))])
Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
A = cupy.asarray((Q * ev) @ Q.T)
A = (A + A.T) / 2
t64, (w64, V64) = tm(lambda: cupy.linalg.eigh(A))
D64 = V64[:, :nocc] @ V64[:, :nocc].T


def oa_refine(A, X):
    # Ogita & Aishima (2018) iterative refinement for symmetric eigendecomposition, all FP64 GEMMs
    R = cupy.eye(X.shape[1]) - X.T @ X
    S = X.T @ (A @ X)
    lam = cupy.diag(S) / (1 - cupy.diag(R))
    dl = lam[None, :] - lam[:, None]
    E = (S + R * lam[None, :]) / cupy.where(abs(dl) > 1e-12, dl, 1.0)
    E[cupy.arange(len(lam)), cupy.arange(len(lam))] = cupy.diag(R) / 2
    return X + X @ E, lam


def fp32_refined(steps):
    w, V = cupy.linalg.eigh(A.astype(cupy.float32))
    X = V.astype(cupy.float64)
    for _ in range(steps):
        X, lam = oa_refine(A, X)
    return X


res = {'fp64_syevd_s': t64}
for steps in (0, 1, 2):
    t, X = tm(lambda: fp32_refined(steps))
    D = X[:, :nocc] @ X[:, :nocc].T
    res[f'fp32+{steps}ref'] = {'s': t, 'max_dD': float(abs(D - D64).max()),
                               'orth': float(abs(X.T @ X - cupy.eye(n)).max())}
OUT['G2'] = res
print('G2', res, flush=True)

# G3: K(dD) = sum_L Y_L diag(lam) Y_L^T, Y_L = B_L V (nao x r): K1 (naux*nao, nao) x (nao, r); K2 (nao, naux*r) x (naux*r, nao)
G3 = {}
naux = 1904
for r in (64, 128):
    for dt in (cupy.float16, cupy.float32, cupy.float64):
        B = cupy.random.standard_normal((naux * n, n), dtype=cupy.float32).astype(dt)
        V = cupy.random.standard_normal((n, r), dtype=cupy.float32).astype(dt)
        t1, Y = tm(lambda: B @ V)
        Yr = Y.reshape(naux, n, r).transpose(1, 0, 2).reshape(n, naux * r)
        Yr = cupy.ascontiguousarray(Yr)
        t2, _ = tm(lambda: Yr @ Yr.T)
        fl = 2. * naux * n * n * r * 2
        G3[f'{np.dtype(dt).name}_r{r}'] = {'K1_s': t1, 'K2_s': t2, 'TFLOPs': fl / (t1 + t2) / 1e12}
        del B, V, Y, Yr
        cupy.get_default_memory_pool().free_all_blocks()
OUT['G3'] = G3
print('G3', G3, flush=True)
json.dump(OUT, open(os.path.join(here, 'r4_gpu2.json'), 'w'), indent=1)
print('done')
