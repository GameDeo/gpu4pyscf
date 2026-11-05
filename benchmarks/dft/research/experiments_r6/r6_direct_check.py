'''Direct (two-pass, no stored tensor) exact K C_occ and J vs the stored-tensor DF; timings. Water dimer + BPA.'''
import os, sys, time
import numpy as np
import cupy
from pyscf import gto
from gpu4pyscf import dft
from gpu4pyscf.df.direct_k import DirectDFK

here = os.path.dirname(os.path.abspath(__file__))
name = sys.argv[1] if len(sys.argv) > 1 else '033_Bisphenol_A'
mol = gto.M(atom=os.path.join(here, '..', '..', '..', 'molecules', 'organic', name + '.xyz'),
            basis='def2-tzvp', verbose=0, max_memory=14000)
mf = dft.RKS(mol, xc='wb97m-v').density_fit()
mf.grids.atom_grid = (30, 110); mf.nlcgrids.atom_grid = (10, 26)
mf.conv_tol = 1e-7
mf.kernel()
nocc = mol.nelectron // 2
Co = mf.mo_coeff[:, :nocc]
dm = mf.make_rdm1()
omega, alpha, hyb = mf._numint.rsh_and_hybrid_coeff(mf.xc)
for om in (None, omega):
    t0 = time.perf_counter()
    k_ref = mf.get_k(mol, dm, hermi=1, omega=om) @ Co
    j_ref = mf.get_j(mol, dm, hermi=1) if om is None else None
    cupy.cuda.Stream.null.synchronize(); t_ref = time.perf_counter() - t0
    for nm in [(None, None), (15, 11)]:
        dk = DirectDFK(mol, mf.with_df.auxmol, om, nmod_t=nm[0], nmod_x=nm[1])
        dk.kc_j(Co[:, :2] * 1.0, with_j=False)       # warm-up / setup
        cupy.cuda.Stream.null.synchronize(); t0 = time.perf_counter()
        kc, j = dk.kc_j(Co * np.sqrt(2.0), with_j=om is None)
        cupy.cuda.Stream.null.synchronize(); t = time.perf_counter() - t0
        kc = kc / np.sqrt(2.0)
        ek = float(abs(kc - k_ref).max()); ee = -0.5 * float(cupy.einsum('ia,ia->', Co, kc - k_ref))
        ej = float(abs(j - j_ref).max()) if j is not None else 0.0
        print('omega', om, 'nmod', nm, 'max|dKC| %.2e  dE_K %.2e  max|dJ| %.2e   direct %.2f s  stored %.2f s'
              % (ek, ee, ej, t, t_ref), flush=True)
print('done')
