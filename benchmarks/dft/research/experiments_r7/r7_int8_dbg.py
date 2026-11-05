'''Debug: which INT8 surrogate (J or K) breaks stage 1; small grids, verbose stage-1 trace.'''
import os, sys
import numpy as np
import cupy
from pyscf import gto
from gpu4pyscf import dft
here = os.path.dirname(os.path.abspath(__file__))
mol = gto.M(atom=os.path.join(here, '..', '..', '..', 'molecules', 'organic', '033_Bisphenol_A.xyz'),
            basis='def2-tzvp', verbose=0, max_memory=14000)
for v in eval(sys.argv[1]):
    mf = dft.RKS(mol, xc='wb97m-v').density_fit()
    mf.grids.atom_grid = (50, 194); mf.nlcgrids.atom_grid = (30, 110)
    mf.conv_tol = 1e-9
    mf.mixed_precision = {'scheme': 'crt'}
    mf.verbose = 6
    mf.stdout = sys.stdout
    v = dict(v); v.setdefault('max_stage1', 25); v.setdefault('max_outer', 3); v.setdefault('max_inner', 15)
    mf.multifidelity = v
    print('=' * 20, v, flush=True)
    e = mf.kernel()
    print('E', e, 'outer', [('%.1e' % g) for g, _ in mf.multifidelity_stats['outer_hist']], flush=True)
