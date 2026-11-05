import os, numpy as np, cupy
from pyscf import gto
from gpu4pyscf import dft
from gpu4pyscf.df import df as gdf
from gpu4pyscf.df.int8_df import Int8DF
from gpu4pyscf.scf import mixed_precision as mp
here = os.path.dirname(os.path.abspath(__file__))
mol = gto.M(atom=os.path.join(here, '..', '..', '..', 'molecules', 'organic', '033_Bisphenol_A.xyz'),
            basis='def2-tzvp', verbose=0)
print('CDERI precision globals:', {k: getattr(gdf, k) for k in dir(gdf) if k.startswith('CDERI')})
a = Int8DF(mol, 'def2-universal-jfit').build()
B0 = a.Q.astype(np.float64) * a.s
print('fp64 build: npairs', a.npairs, 'max|B|', float(abs(B0).max()))
for nm in (8, 11, 15):
    old = gdf.CDERI_PRECISION if hasattr(gdf, 'CDERI_PRECISION') else None
    if old is not None:
        gdf.CDERI_PRECISION = 'crt'
    b = Int8DF(mol, 'def2-universal-jfit', nmod=nm).build()
    if old is not None:
        gdf.CDERI_PRECISION = old
    B1 = b.Q.astype(np.float64) * b.s
    print('crt nmod %d: max|B| %.3e  max|dB| %.3e  rel fro %.2e' % (nm, float(abs(B1).max()), float(abs(B1 - B0).max()),
          float(cupy.linalg.norm(B1 - B0) / cupy.linalg.norm(B0))), flush=True)
# pool reuse: fill memory with garbage then rebuild
junk = cupy.full(int(3e8), 1e30); del junk
c = Int8DF(mol, 'def2-universal-jfit').build()
B2 = c.Q.astype(np.float64) * c.s
print('after junk: max|B| %.3e max|dB| %.3e' % (float(abs(B2).max()), float(abs(B2 - B0).max())))
