# Copyright 2021-2025 The PySCF Developers. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.
"""Numpy emulation of R^2 = |xi|^2 + |xj|^2 - 2 xi.xj computed as a K=5 tensor-core GEMM
(TF32 / FP16 / BF16 operand rounding with 1..3-term splitting) on tile-local coordinates;
reports the R^2 error and the induced error of the tile sum E_i.
Usage: python tc_emul_vv10.py w33.npz"""
import numpy as np
import sys
z = np.load(sys.argv[1])
coords, weights, rd = z['coords'], z['weights'], z['rho']
kpre = 6.0 * 1.5 * np.pi * (9 * np.pi) ** (-1 / 6.)
m = (rd[0] >= 1e-10) & (np.abs(weights) > 1e-14)
rho = rd[0][m]; xyz = coords[m]; w = weights[m]
gam = (rd[1:4][:, m] ** 2).sum(0)
omega = np.sqrt(0.01 * gam ** 2 / rho ** 4 + 4 / 3 * np.pi * rho); kappa = kpre * rho ** (1 / 6.)
rw = rho * w


def rnd(x, bits):
    # round to `bits` explicit mantissa bits (RNE-ish)
    m_, e = np.frexp(x)
    return np.ldexp(np.round(m_ * 2 ** (bits + 1)) / 2 ** (bits + 1), e)


def split(x, bits, n):
    out = []; r = x.copy()
    for _ in range(n):
        h = rnd(r, bits); out.append(h); r = r - h
    return out


def gemm_emul(A, B, bits, nsplit):
    # A (n,K), B (m,K): sum over K with each operand split into nsplit pieces, FP32 accumulation, keep terms hi*hi, hi*lo, lo*hi
    As = split(A, bits, nsplit); Bs = split(B, bits, nsplit)
    acc = np.zeros((A.shape[0], B.shape[0]))
    for p in range(nsplit):
        for q in range(nsplit):
            if p + q < nsplit:
                acc += (As[p].astype(np.float64) @ Bs[q].astype(np.float64).T)
    return acc.astype(np.float32).astype(np.float64)


rng = np.random.default_rng(0)
T = 512
# Morton-ish tiles by sorting along a coarse cell
cell = np.floor((xyz - xyz.min(0)) / 2.0).astype(int)
key = cell[:, 0] * 100000 + cell[:, 1] * 300 + cell[:, 2]
idx = np.argsort(key)
for (ti, tj) in ((10, 10), (10, 11), (10, 60), (10, 400)):
    I = idx[ti * T:(ti + 1) * T]; J = idx[tj * T:(tj + 1) * T]
    cJ = xyz[J].mean(0)
    xi = xyz[I] - cJ; xj = xyz[J] - cJ
    # exact
    d = xi[:, None, :] - xj[None, :, :]
    R2 = (d ** 2).sum(-1)
    A = np.concatenate([-2 * xi, (xi ** 2).sum(1, keepdims=True), np.ones((T, 1))], axis=1)
    B = np.concatenate([xj, np.ones((T, 1)), (xj ** 2).sum(1, keepdims=True)], axis=1)
    A[:, :3] = xi * -2; B[:, :3] = xj
    res = {}
    for name, bits, ns in (('tf32x1', 10, 1), ('tf32x3', 10, 2), ('fp16x1', 10, 1), ('fp16x3', 10, 2), ('bf16x3', 7, 3)):
        R2t = gemm_emul(A, B, bits, ns)
        a = omega[I][:, None] * R2 + kappa[I][:, None]; b = omega[J][None, :] * R2 + kappa[J][None, :]
        a2 = omega[I][:, None] * R2t + kappa[I][:, None]; b2 = omega[J][None, :] * R2t + kappa[J][None, :]
        P = 1 / (a * b * (a + b)); P2 = 1 / (a2 * b2 * (a2 + b2))
        Ei = (P * rw[J][None]).sum(1); Ei2 = (P2 * rw[J][None]).sum(1)
        res[name] = (np.abs(R2t - R2).max(), np.abs(Ei2 - Ei).sum() / np.abs(Ei).sum())
    print('tiles', ti, tj, 'dist %.1f' % np.linalg.norm(xyz[I].mean(0) - cJ), {k: '%.1e/%.1e' % v for k, v in res.items()})
