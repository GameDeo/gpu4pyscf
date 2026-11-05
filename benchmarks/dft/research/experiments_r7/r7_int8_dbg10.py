import os, numpy as np, cupy
from pyscf import gto
from gpu4pyscf.df import df as gdf
from gpu4pyscf.df.int8_df import Int8DF
here = os.path.dirname(os.path.abspath(__file__))
mol = gto.M(atom=os.path.join(here, '..', '..', '..', 'molecules', 'organic', '033_Bisphenol_A.xyz'),
            basis='def2-tzvp', verbose=0)
a = Int8DF(mol, 'def2-universal-jfit', bits=16).build()
B0 = a.Q.astype(np.float64) * a.s
gdf.CDERI_PRECISION = 'crt'
for frac in (0.25, 0.02, 0.002):
    for nm in (11, 15):
        b = Int8DF(mol, 'def2-universal-jfit', bits=16, nmod=nm).build(mem_frac=frac)
        B1 = b.Q.astype(np.float64) * b.s
        print('crt frac %g nmod %d max|dB| %.3e' % (frac, nm, float(abs(B1 - B0).max())), flush=True)
