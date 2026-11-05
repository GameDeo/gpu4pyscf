import os, numpy as np, cupy
from pyscf import gto
from gpu4pyscf import dft
from gpu4pyscf.df.direct_k import DirectDFK
from gpu4pyscf.df import df as gdf
from gpu4pyscf.df.int3c2e_bdiv import int2c2e
from gpu4pyscf.lib.cupy_helper import cholesky
here = os.path.dirname(os.path.abspath(__file__))
mol = gto.M(atom=os.path.join(here, '..', '..', '..', 'molecules', 'organic', '033_Bisphenol_A.xyz'), basis='def2-tzvp', verbose=0)
mf = dft.RKS(mol, xc='wb97m-v').density_fit(); mf.grids.atom_grid=(20,50); mf.nlcgrids.atom_grid=(10,26)
mf.conv_tol=1e-6; mf.kernel()
for om in (None, 0.3):
    dk = DirectDFK(mol, mf.with_df.auxmol, om)
    j2c = int2c2e(dk.auxmol, omega=dk.omega, lr_factor=dk.lr, sr_factor=dk.sr)
    w = cupy.linalg.eigvalsh(j2c)
    try:
        cholesky(j2c); ok = True
    except Exception as e:
        ok = repr(e)[:80]
    print('omega', om, 'min eig %.2e max %.2e cond %.1e' % (float(w[0]), float(w[-1]), float(w[-1]/w[0])), 'cholesky', ok,
          'LINEAR_DEP_THR', gdf.LINEAR_DEP_THR, 'n below thr', int((w < gdf.LINEAR_DEP_THR).sum()))
    d = mf.with_df if om is None else None
    if om is not None:
        with mf.with_df.range_coulomb(om, None, None) as d:
            print('  stored tag', getattr(d, '_cderi_idx', None) is not None, getattr(d, 'cd_low', None) is None)
    else:
        print('  stored', type(d._cderi))
