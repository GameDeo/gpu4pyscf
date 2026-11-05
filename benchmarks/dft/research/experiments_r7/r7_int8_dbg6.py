import os, numpy as np, cupy
from pyscf import gto
from gpu4pyscf import dft
from gpu4pyscf.df import df as gdf
from gpu4pyscf.df.int8_df import Int8DF
here = os.path.dirname(os.path.abspath(__file__))
mol = gto.M(atom=os.path.join(here, '..', '..', '..', 'molecules', 'organic', '033_Bisphenol_A.xyz'),
            basis='def2-tzvp', verbose=0)
mf = dft.RKS(mol, xc='wb97m-v').density_fit(); mf.grids.atom_grid = (30, 110); mf.nlcgrids.atom_grid = (10, 26)
mf.conv_tol = 1e-6; mf.kernel()
dm = mf.make_rdm1()
dm0 = cupy.asarray(mf.get_init_guess(mol, 'minao'))
C = mf.mo_coeff[:, mf.mo_occ > 0]
dC = cupy.asarray(np.random.default_rng(0).standard_normal(C.shape)) * 1e-3
dD = 2 * ((C + dC) @ (C + dC).T - C @ C.T)
dfj = gdf.DF(mol, auxbasis='def2-universal-jfit').build()
ref = {k: dfj.get_jk(x, with_k=False)[0] for k, x in (('D', dm), ('D0', dm0), ('dD', dD), ('D-D0', dm - dm0))}
for metric, nmod, bits in (('cd', 11, 8), ('cd', 11, 16), ('sym', 11, 16)):
    a = Int8DF(mol, 'def2-universal-jfit', nmod=nmod, metric=metric, bits=bits).build()
    errs = {k: float(abs(a.j(x) - ref[k]).max() / abs(ref[k]).max()) for k, x in (('D', dm), ('D0', dm0), ('dD', dD), ('D-D0', dm - dm0))}
    print(metric, nmod, bits, ' '.join('%s %.2e' % kv for kv in errs.items()), flush=True)
omega, alpha, hyb = mf._numint.rsh_and_hybrid_coeff(mf.xc)
kd = Int8DF(mol, 'def2-svp-ri', omega=omega, lr_factor=alpha, sr_factor=hyb).build()
kh = Int8DF(mol, 'def2-svp-ri', omega=omega, lr_factor=alpha, sr_factor=hyb, host=True).build()
Pd = kd.k_pairs(C); Ph = kh.k_pairs(C)
print('K host vs device max diff %.2e (max %.2e)' % (float(abs(Pd - Ph).max()), float(abs(Pd).max())))
