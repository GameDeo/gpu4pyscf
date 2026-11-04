#!/usr/bin/env python
'''
CPU-only (NumPy + PySCF) check of the exact-up-to-rounding algebra used in gpu4pyscf/solvent/pcm.py after the solvent
audit, on the real PCM matrices (D, S, A, K, R) of PySCF's CPU implementation:

  1. DA-free left_multiply_R:  (D*A) x = D (A x),  (D*A)^T x = A (D^T x)       (A is a column scaling of D)
  2. C-PCM / COSMO: K = S symmetric and R = -f 1  =>  K^-1 R v == R^T K^-T v, the second solve of q_sym is redundant
     (and for IEF-PCM / SS(V)PE it is NOT: the check prints the size of the difference)
  3. the cache key: the surface charges are a function of the spin-summed density only.

    python solvent_algebra_cpu_check.py
'''
import numpy as np
from pyscf import gto, scf

mol = gto.M(atom='O 0 0 0.117; H -0.757 0 -0.470; H 0.757 0 -0.470', basis='6-31g', verbose=0)
rng = np.random.default_rng(1)
worst = {}
for method in ['C-PCM', 'IEF-PCM', 'SS(V)PE', 'COSMO']:
    mf = scf.RHF(mol).PCM()
    mf.with_solvent.method = method
    mf.with_solvent.lebedev_order = 17
    mf.with_solvent.eps = 35.9
    mf.kernel()
    sol = mf.with_solvent
    K, R = sol._intermediates['K'], sol._intermediates['R']
    n = K.shape[0]
    f = (35.9 - 1) / (35.9 + (0.5 if method == 'COSMO' else (0.0 if method == 'C-PCM' else 1.0)))
    f = (35.9 - 1) / 35.9 if method == 'C-PCM' else f
    v = rng.standard_normal((1, n))
    q = np.linalg.solve(K, R @ v.T).T
    qt = (R.T @ np.linalg.solve(K.T, v.T)).T
    diff = abs(q - qt).max() / abs(q).max()
    print(f'{method:8s}: |K^-1 R v - R^T K^-T v| / |q| = {diff:.2e}   K symmetric: {abs(K - K.T).max():.1e}   '
          f'R = -f*1: {abs(R + f * np.eye(n)).max() if method in ("C-PCM", "COSMO") else float("nan"):.1e}')
    if method in ('C-PCM', 'COSMO'):
        assert diff < 1e-10, 'single-solve shortcut would be wrong'
    else:
        assert diff > 1e-6, 'the second solve is needed here'
        # DA-free R multiplication.  PySCF stores R = -f (1 - DA/2pi), so DA is recovered from R.
        DA = (np.eye(n) + R / f) * 2 * np.pi
        # PySCF's own D and A: recompute D and the area vector from the same surface
        from pyscf.solvent.pcm import get_D_S, get_F_A
        D, S = get_D_S(sol.surface, with_S=True, with_D=True)
        _, A = get_F_A(sol.surface)
        assert abs(D * A - DA).max() < 1e-9
        x = rng.standard_normal((n, 1))
        for X in (x, x[:, 0]):
            Ac = A.reshape((-1,) + (1,) * (X.ndim - 1))
            e1 = abs(D @ (Ac * X) - (D * A) @ X).max()
            e2 = abs(Ac * (D.T @ X) - (D * A).T @ X).max()
            worst[method] = max(worst.get(method, 0), e1, e2)
print('DA-free R multiply, max abs deviation from the explicit D*A product:', {k: f'{v:.1e}' for k, v in worst.items()})
assert all(v < 1e-10 for v in worst.values())

# cache key: surface charges depend on the spin-summed density only
mf = scf.UHF(mol).PCM()
mf.with_solvent.lebedev_order = 17
mf.kernel()
dm = mf.make_rdm1()
sol = mf.with_solvent
e_a, _ = sol._get_vind(dm)
q_a = sol._intermediates['q_sym'].copy()
e_b, _ = sol._get_vind(dm[0] + dm[1])
print('UHF spin-resolved vs spin-summed density: |dq| =', abs(sol._intermediates['q_sym'] - q_a).max(), '|dE| =', abs(e_a - e_b))
assert abs(sol._intermediates['q_sym'] - q_a).max() < 1e-13
print('all algebra checks passed')
