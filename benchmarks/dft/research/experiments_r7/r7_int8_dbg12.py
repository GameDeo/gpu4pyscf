import os, numpy as np, cupy
from pyscf import gto
from gpu4pyscf import dft
from gpu4pyscf.df import df as gdf
from gpu4pyscf.df.int8_df import Int8DF
here = os.path.dirname(os.path.abspath(__file__))
mol = gto.M(atom=os.path.join(here, '..', '..', '..', 'molecules', 'organic', '033_Bisphenol_A.xyz'),
            basis='def2-tzvp', verbose=0)
dm1 = cupy.asarray(np.load('/content/res/dm1.npy')); C1 = cupy.asarray(np.load('/content/res/C1.npy'))
S = cupy.asarray(mol.intor('int1e_ovlp'))
print('dm1: sym %.2e  tr(DS) %.4f  max %.3e  shape %s dtype %s' % (float(abs(dm1 - dm1.T).max()), float(cupy.trace(dm1 @ S)),
      float(abs(dm1).max()), dm1.shape, dm1.dtype))
print('C1 orthonormality max|C^T S C - 1| %.2e' % float(abs(C1.T @ S @ C1 - cupy.eye(C1.shape[1])).max()))
dfj = gdf.DF(mol, auxbasis='def2-universal-jfit').build()
a = Int8DF(mol, 'def2-universal-jfit', bits=16).build()
Jx = dfj.get_jk(dm1, with_k=False)[0]
print('J err dm1 %.3e (max %.3e)' % (float(abs(a.j(dm1) - Jx).max()), float(abs(Jx).max())))
# pair coverage: dm1 mass outside the screened pair list
mask = cupy.zeros(dm1.shape, dtype=bool); mask[a.rows, a.cols] = True; mask[a.cols, a.rows] = True
print('max|dm1| outside pairs %.3e' % float(abs(dm1[~mask]).max()))
Jx2 = dfj.get_jk(cupy.where(mask, dm1, 0), with_k=False)[0]
print('ref J with masked dm %.3e' % float(abs(Jx2 - Jx).max()))
