# Copyright 2021-2025 The PySCF Developers. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.
"""CPU (numpy) study of the structure of the VV10 pair kernel on a real NLC grid:
(1) error of dropping all pairs with R > Rc,  (2) numerical rank of far-field blocks.
Usage: python analyze_vv10_structure.py w17.npz"""
import sys, numpy as np
np.set_printoptions(linewidth=200)
name = sys.argv[1]            # path of a dataset .npz written by bench_vv10.py --make
z = np.load(name)
coords, weights, rd = z['coords'], z['weights'], z['rho']
kpre = 6.0 * 1.5 * np.pi * (9 * np.pi) ** (-1 / 6.)
C = 0.01
m = (rd[0] >= 1e-10) & (np.abs(weights) > 1e-14)
rho = rd[0][m]; xyz = coords[m]; w = weights[m]
gam = (rd[1:4][:, m] ** 2).sum(0)
omega = np.sqrt(C * gam ** 2 / rho ** 4 + 4 / 3 * np.pi * rho)
kappa = kpre * rho ** (1 / 6.)
rw = rho * w
N = len(rho)
print(name, 'N', N, 'extent', xyz.max(0) - xyz.min(0))
print('omega pct', np.percentile(omega, [0, 1, 50, 99, 100]))
print('kappa pct', np.percentile(kappa, [0, 1, 50, 99, 100]))
print('omega/kappa pct', np.percentile(omega / kappa, [0, 1, 50, 99, 100]))
print('sum rw', rw.sum())


def kern(i, j):
    # returns Phi, Psi=Phi*(1/a+1/s), R2 for index arrays i (rows) j (cols)
    d = xyz[i][:, None, :] - xyz[j][None, :, :]
    R2 = (d ** 2).sum(-1)
    a = omega[i][:, None] * R2 + kappa[i][:, None]
    b = omega[j][None, :] * R2 + kappa[j][None, :]
    s = a + b
    P = 1 / (a * b * s)
    return P, P * (1 / a + 1 / s), R2


# ---------- 1. tail of the pair sums: error from dropping all pairs with R > Rc
rng = np.random.default_rng(0)
ni, nj = 3000, 80000
ii = rng.choice(N, ni, replace=False)
jj = rng.choice(N, nj, replace=False)
scale = N / nj
Rcs = np.array([6, 8, 10, 12, 15, 20, 25, 30, 40, 60, 1e9])
acc = {rc: np.zeros((3, ni)) for rc in Rcs}
for j0 in range(0, nj, 4000):
    j = jj[j0:j0 + 4000]
    P, Psi, R2 = kern(ii, j)
    R = np.sqrt(R2)
    vals = [P * rw[j][None], Psi * rw[j][None], Psi * R2 * rw[j][None]]
    for rc in Rcs:
        mk = R <= rc
        for q in range(3):
            acc[rc][q] += (vals[q] * mk).sum(1)
full = acc[1e9]
print('--- screening: error from dropping pairs with R > Rc (relative to the full sum, point-wise; rw-weighted mean and max over %d sample points)' % ni)
wi = rw[ii]
for rc in Rcs[:-1]:
    out = []
    for q, nm in enumerate('EUW'):
        rel = np.abs(full[q] - acc[rc][q]) / full[q]
        out.append('%s: wmean %.1e max %.1e' % (nm, (rel * wi).sum() / wi.sum(), rel.max()))
    # fraction of pairs with R<=rc (all sampled)
    print('Rc=%5.1f  ' % rc, '  '.join(out))
# pair fraction within Rc
fr = []
Rall = []
for j0 in range(0, 20000, 4000):
    j = jj[j0:j0 + 4000]
    d = xyz[ii[:500]][:, None, :] - xyz[j][None, :, :]
    Rall.append(np.sqrt((d ** 2).sum(-1)).ravel())
Rall = np.concatenate(Rall)
print('fraction of pairs with R<=Rc:', {int(rc): round(float((Rall <= rc).mean()), 3) for rc in Rcs[:-1]})

# ---------- 2. numerical rank of far-field blocks
print('--- numerical rank of Phi, Psi blocks (relative Frobenius accuracy), two clusters of ~n points of box size L at centre distance Dc')
order = np.lexsort((xyz[:, 2] // 3.0, xyz[:, 1] // 3.0, xyz[:, 0] // 3.0))


def cluster(center, L, nmax):
    sel = np.where(np.all(np.abs(xyz - center) < L / 2, axis=1))[0]
    if len(sel) > nmax:
        sel = rng.choice(sel, nmax, replace=False)
    return sel


lo, hi = xyz.min(0), xyz.max(0)
ctr = 0.5 * (lo + hi)
ext = hi - lo
ax = np.argmax(ext)
for L in (4.0, 8.0):
    for Dc in (12.0, 20.0, 30.0):
        if Dc > ext[ax] * 0.95:
            continue
        c1 = ctr.copy(); c1[ax] -= Dc / 2
        c2 = ctr.copy(); c2[ax] += Dc / 2
        i = cluster(c1, L, 1500); j = cluster(c2, L, 1500)
        if len(i) < 200 or len(j) < 200:
            continue
        P, Psi, R2 = kern(i, j)
        res = []
        for M in (P, Psi, Psi * R2):
            s = np.linalg.svd(M, compute_uv=False)
            cum = np.sqrt(np.cumsum(s[::-1] ** 2))[::-1]   # tail Frobenius
            r = [int((cum > t * cum[0]).sum()) for t in (1e-3, 1e-6, 1e-8)]
            res.append(r)
        print('L=%.0f D=%.0f n=(%d,%d): rank@1e-3,1e-6,1e-8 (Phi, Psi, Psi*R2) =' % (L, Dc, len(i), len(j)), res)
