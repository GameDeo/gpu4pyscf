import os, numpy as np, cupy
from pyscf import gto
from gpu4pyscf.df import df as gdf
from gpu4pyscf.df.int8_df import Int8DF
here = os.path.dirname(os.path.abspath(__file__))
mol = gto.M(atom=os.path.join(here, '..', '..', '..', 'molecules', 'organic', '033_Bisphenol_A.xyz'),
            basis='def2-tzvp', verbose=0)
a = Int8DF(mol, 'def2-universal-jfit').build()
B0 = a.Q.astype(np.float64) * a.s
for frac in (0.02, 0.002):
    b = Int8DF(mol, 'def2-universal-jfit').build(mem_frac=frac)
    B1 = b.Q.astype(np.float64) * b.s
    print('mem_frac', frac, 'max|dB| %.3e' % float(abs(B1 - B0).max()), flush=True)
dfj = gdf.DF(mol, auxbasis='def2-universal-jfit').build()
rng = np.random.default_rng(0)
C = cupy.asarray(rng.standard_normal((mol.nao, 60))) * 0.1
dm = C @ C.T
Jx = dfj.get_jk(dm, with_k=False)[0]
for _ in range(3):
    J8 = a.j(dm)
print('J rel %.2e' % float(abs(J8 - Jx).max() / abs(Jx).max()))
dm2 = cupy.asarray(dm, order='F')
print('J F-order rel %.2e' % float(abs(a.j(dm2) - Jx).max() / abs(Jx).max()))
