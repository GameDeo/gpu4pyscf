'''K C_occ for FR and LR: GPU stored DF, GPU direct DF, CPU pyscf DF (water dimer, def2-TZVP, def2-universal-jkfit).'''
import numpy as np, cupy, pyscf
from pyscf import gto, df as cdf
from gpu4pyscf import dft
from gpu4pyscf.df.direct_k import DirectDFK
mol = gto.M(atom='''O -1.551007 -0.114520 0.0; H -1.934259 0.762503 0.0; H -0.599677 0.040712 0.0
O 1.350625 0.111469 0.0; H 1.680398 -0.373741 -0.758561; H 1.680398 -0.373741 0.758561''', basis='def2-tzvp', verbose=0)
mf = dft.RKS(mol, xc='wb97m-v').density_fit(); mf.grids.atom_grid=(30,110); mf.nlcgrids.atom_grid=(10,26); mf.kernel()
nocc = mol.nelectron // 2; Co = mf.mo_coeff[:, :nocc]; dm = mf.make_rdm1()
cm = pyscf.dft.RKS(mol).density_fit(auxbasis=mf.with_df.auxbasis)
for om in (None, 0.3):
    ks = mf.get_k(mol, dm, hermi=1, omega=om) @ Co
    kd = DirectDFK(mol, mf.with_df.auxmol, om).kc_j(Co * np.sqrt(2.), with_j=False)[0] / np.sqrt(2.)
    dmc = dm.get(); Cc = Co.get()
    if om is None:
        kc = cm.with_df.get_jk(dmc, hermi=1, with_j=False)[1] @ Cc
    else:
        kc = cm.with_df.get_jk(dmc, hermi=1, with_j=False, omega=om)[1] @ Cc
    e = lambda k: -0.5 * float(np.einsum('ia,ia->', Cc, k))
    print('omega', om, 'E_K stored %.10f direct %.10f cpu %.10f' % (e(ks.get()), e(kd.get()), e(kc)),
          ' stored-cpu %.1e direct-cpu %.1e' % (e(ks.get()) - e(kc), e(kd.get()) - e(kc)), flush=True)
