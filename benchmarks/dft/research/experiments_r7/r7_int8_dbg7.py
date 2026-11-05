import os, numpy as np, cupy
from pyscf import gto
from gpu4pyscf import dft
from gpu4pyscf.df import df as gdf
from gpu4pyscf.df.int8_df import Int8DF
from gpu4pyscf.scf import multifidelity as mfm
from gpu4pyscf.lib.cupy_helper import tag_array
here = os.path.dirname(os.path.abspath(__file__))
mol = gto.M(atom=os.path.join(here, '..', '..', '..', 'molecules', 'organic', '033_Bisphenol_A.xyz'),
            basis='def2-tzvp', verbose=0)
mf = dft.RKS(mol, xc='wb97m-v').density_fit(); mf.grids.atom_grid = (30, 110); mf.nlcgrids.atom_grid = (10, 26)
mf.conv_tol = 1e-6; mf.kernel()
dm = tag_array(mf.make_rdm1(), mo_coeff=mf.mo_coeff, mo_occ=mf.mo_occ)
mf.mixed_precision = {'scheme': 'crt'}
opt = dict(mfm.DEFAULTS)
dfj = gdf.DF(mol, auxbasis='def2-universal-jfit').build()
Jx = dfj.get_jk(cupy.asarray(dm), with_k=False)[0]
omega, alpha, hyb = mf._numint.rsh_and_hybrid_coeff(mf.xc)
skd = mfm.SurrogateK(mf, 'def2-svp-ri', omega, alpha, hyb, mode='fp32')
Kx = skd.k_ref(cupy.asarray(dm))
for ctx in ('none', 'crt'):
    c = mfm._precision_ctx(mf, opt) if ctx == 'crt' else __import__('contextlib').nullcontext()
    with c:
        print(ctx, 'CDERI', gdf.CDERI_PRECISION, gdf.CDERI_CRT_MODULI, flush=True)
        js = Int8DF(mol, 'def2-universal-jfit', bits=16).build()
        J1 = js.j(dm)
        print('  J tag dm rel %.2e' % float(abs(J1 - Jx).max() / abs(Jx).max()), flush=True)
        sk8 = mfm.SurrogateK8(mf, 'def2-svp-ri', omega, alpha, hyb)
        K1 = sk8.k_ref(dm)
        print('  K tag dm rel %.2e   host %s' % (float(abs(K1 - Kx).max() / abs(Kx).max()), sk8.t.host), flush=True)
        C = mf.mo_coeff[:, mf.mo_occ > 0]
        K2 = sk8.k_abs(C)
        print('  K abs rel %.2e' % float(abs(K2 - Kx).max() / abs(Kx).max()), flush=True)
        Kd = skd.k_abs(C) if skd.B is not None else None
        skd.ensure(); Kd = skd.k_abs(C)
        print('  dense k_abs vs k_ref rel %.2e' % float(abs(Kd - Kx).max() / abs(Kx).max()), flush=True)
        print('  sym check J %.2e K %.2e' % (float(abs(J1 - J1.T).max()), float(abs(K1 - K1.T).max())), flush=True)
        js = sk8 = None
