'''Bisect: which kernel() setup step makes a fresh Int8DF J build wrong.'''
import os, numpy as np, cupy
from pyscf import gto
from gpu4pyscf import dft
from gpu4pyscf.df import df as gdf
from gpu4pyscf.df.int8_df import Int8DF
from gpu4pyscf.scf import multifidelity as mfm
here = os.path.dirname(os.path.abspath(__file__))
mol = gto.M(atom=os.path.join(here, '..', '..', '..', 'molecules', 'organic', '033_Bisphenol_A.xyz'),
            basis='def2-tzvp', verbose=0)
dm0 = cupy.asarray(dft.RKS(mol).get_init_guess(mol, 'minao'))
Jx = gdf.DF(mol, auxbasis='def2-universal-jfit').build().get_jk(dm0, with_k=False)[0]
def check(tag):
    a = Int8DF(mol, 'def2-universal-jfit', bits=16).build()
    print('%-28s J err %.2e' % (tag, float(abs(a.j(dm0) - Jx).max() / abs(Jx).max())), flush=True)
check('start')
mf = dft.RKS(mol, xc='wb97m-v').density_fit()
mf.grids.atom_grid = (99, 590); mf.nlcgrids.atom_grid = (50, 194)
mf.mixed_precision = {'scheme': 'crt'}
mf.build(mol); check('mf.build')
mf.grids.build(with_non0tab=False, sort_grids=True); check('grids')
mf.nlcgrids.build(with_non0tab=False, sort_grids=True); check('nlcgrids')
ni = mf._numint
if ni.gdftopt is None:
    ni.build(mol, mf.grids.coords)
check('ni.build')
g_in = mfm._grid(mol, (30, 110), mf.grids); check('_grid')
h1e = cupy.asarray(mf.get_hcore()); s1e = cupy.asarray(mf.get_ovlp()); check('hcore/ovlp')
eig = mfm.FastEig(s1e, 'fp64', mol.nelectron // 2); check('FastEig')
cx = mfm.CoarseXC(ni, mol, g_in, mf.xc, 0.15); check('CoarseXC')
with mfm._precision_ctx(mf, dict(mfm.DEFAULTS)):
    check('in ctx')
    v = ni.nr_rks(mol, g_in, mf.xc, dm0)[2]; check('nr_rks coarse')
    mf.with_df.build(); check('exact DF build')
    vj = mf.get_j(mol, dm0); check('get_j')
