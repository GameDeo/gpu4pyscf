'''Isolated test of the diagonalisation-free occupied rotation: fixed Fock, perturbed start, residual per call.'''
import numpy as np, cupy
from pyscf import gto
from gpu4pyscf import dft
from gpu4pyscf.scf import multifidelity as M
mol = gto.M(atom='''O -1.551007 -0.114520 0.0; H -1.934259 0.762503 0.0; H -0.599677 0.040712 0.0
O 1.350625 0.111469 0.0; H 1.680398 -0.373741 -0.758561; H 1.680398 -0.373741 0.758561''', basis='def2-tzvp', verbose=0)
mf = dft.RKS(mol, xc='pbe0').density_fit(); mf.kernel()
s1e = cupy.asarray(mf.get_ovlp()); f = mf.get_fock()
nocc = mol.nelectron // 2
eig = M.FastEig(s1e, 'fp64', nocc); Sinv = eig.X @ eig.X.T
rng = np.random.default_rng(1)
R = cupy.asarray(rng.standard_normal(f.shape)); R = R + R.T
R = s1e @ R @ s1e / float(abs(R).max())
for amp in (1e-2, 1e-3, 1e-4):
    for refresh in (False, True):
        e1, C1 = eig(f + amp * R)
        C, e = C1, e1
        res = [M._proj_grad(f @ C[:, :nocc], C[:, :nocc], Sinv, s1e)]
        for it in range(6):
            if refresh:
                Cv = C[:, nocc:]
                e = e.copy(); e[nocc:] = cupy.einsum('pa,pq,qa->a', Cv, f, Cv) / cupy.einsum('pa,pq,qa->a', Cv, s1e, Cv)
            e, C = M._rotate(f, C, e, s1e, nocc, 2)
            res.append(M._proj_grad(f @ C[:, :nocc], C[:, :nocc], Sinv, s1e))
        print('amp %.0e refresh %s' % (amp, refresh), ' '.join('%.1e' % r for r in res), flush=True)
