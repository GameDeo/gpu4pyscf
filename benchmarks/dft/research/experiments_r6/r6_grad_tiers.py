'''BPA: one multi-fidelity SCF, then the nuclear gradient with Gamma precision variants (FP32 Rys tier fraction,
Ozaki-I INT8 Gamma build). Reports max|dg| vs the tight FP64 reference and the DF J/K gradient time.'''
import os, sys, time, json
import numpy as np
import cupy
from pyscf import gto
from gpu4pyscf import dft
from gpu4pyscf.df.grad import rhf as dgr

here = os.path.dirname(os.path.abspath(__file__))
ref = json.load(open(sys.argv[1]))
variants = eval(sys.argv[2]) if len(sys.argv) > 2 else [
    {}, {'ndig': 6}, {'ndig': 5}, {'frac': 1e-4}, {'frac': 1e-3}, {'frac': 1e-2}, {'frac': 3e-2}, {'ndig': 6, 'frac': 1e-2}]
T = {}
f0 = dgr._jk_energy_per_atom


def timed(*a, **k):
    cupy.cuda.Stream.null.synchronize(); t0 = time.perf_counter()
    r = f0(*a, **k)
    cupy.cuda.Stream.null.synchronize(); T['jk'] = T.get('jk', 0) + time.perf_counter() - t0
    return r


dgr._jk_energy_per_atom = timed
w = gto.M(atom='O 0 0 0; H 0 .76 .59; H 0 -.76 .59', basis='def2-tzvp', verbose=0)
wm = dft.RKS(w, xc='wb97m-v').density_fit()
wm.grids.atom_grid = (20, 50); wm.nlcgrids.atom_grid = (10, 26)
wm.mixed_precision = {'scheme': 'crt'}; wm.multifidelity = True
wm.kernel()
for v in variants:
    dgr.GAMMA_OZ1_NDIG = v.get('ndig', 0); dgr.GRAD_TIERS = {'frac': v['frac']} if 'frac' in v else None
    wm.nuc_grad_method().kernel()
mol = gto.M(atom=os.path.join(here, '..', '..', '..', 'molecules', 'organic', '033_Bisphenol_A.xyz'),
            basis='def2-tzvp', verbose=0, max_memory=14000)
mf = dft.RKS(mol, xc='wb97m-v').density_fit()
mf.grids.atom_grid = (99, 590); mf.nlcgrids.atom_grid = (50, 194)
mf.conv_tol = 1e-9
mf.mixed_precision = {'scheme': 'crt'}; mf.multifidelity = True
e = mf.kernel()
print('SCF dE %.2e' % (e - ref['energy']), flush=True)
out = []
for v in variants:
    dgr.GAMMA_OZ1_NDIG = v.get('ndig', 0); dgr.GRAD_TIERS = {'frac': v['frac']} if 'frac' in v else None
    T.clear()
    cupy.cuda.Stream.null.synchronize(); t0 = time.perf_counter()
    g = mf.nuc_grad_method().kernel()
    cupy.cuda.Stream.null.synchronize(); t = time.perf_counter() - t0
    dg = float(abs(np.asarray(g) - np.asarray(ref['grad'])).max())
    print('VARIANT', v, 'grad %.2f s  DF_JK %.2f s  max|dg| %.2e' % (t, T.get('jk', 0), dg), flush=True)
    out.append(dict(v=v, t=t, jk=T.get('jk', 0), dg=dg))
json.dump(out, open('/content/res/r6_grad_tiers.json', 'w'), indent=1)
print('done')
