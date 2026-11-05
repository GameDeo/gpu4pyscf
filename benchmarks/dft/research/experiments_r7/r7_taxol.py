'''Taxol wB97M-V/def2-TZVP (def2-universal-jkfit), grids (99,590)/(50,194), crt scheme + multi-fidelity SCF + nuclear
gradient on one T4.  Default options select the large-system path automatically: direct exact K/J at the checkpoints,
INT8-stored surrogate J and K for the inner loop.  argv[1]: json dict of extra multi-fidelity options.'''
import os, sys, time, json
import numpy as np
import cupy
from pyscf import gto
from gpu4pyscf import dft
from gpu4pyscf.lib import logger

here = os.path.dirname(os.path.abspath(__file__))
opts = eval(sys.argv[1]) if len(sys.argv) > 1 else {}
do_grad = not (len(sys.argv) > 2 and sys.argv[2] == 'nograd')
mol = gto.M(atom=os.path.join(here, '..', '..', '..', 'molecules', 'organic', '113_Taxol.xyz'),
            basis='def2-tzvp', verbose=4, max_memory=11000)
sync = cupy.cuda.Stream.null.synchronize
pool = cupy.get_default_memory_pool()
mf = dft.RKS(mol, xc='wb97m-v').density_fit()
mf.grids.atom_grid = (99, 590)
mf.nlcgrids.atom_grid = (50, 194)
mf.conv_tol = 1e-9
mf.mixed_precision = {'scheme': 'crt'}
mf.multifidelity = dict(opts) or True
print('options', mf.multifidelity, flush=True)
sync(); t0 = time.perf_counter()
e = mf.kernel()
sync(); ts = time.perf_counter() - t0
st = mf.multifidelity_stats
print('TAXOL SCF E = %.10f  %.1f s  converged %s  peak pool %.2f GB' % (e, ts, mf.converged, pool.total_bytes() / 1e9), flush=True)
print('   cycles', st['cycles'], 'outer |g|', ['%.1e' % x for x, _ in st['outer_hist']], flush=True)
for k, (s_, n_) in sorted(st['timers'].items(), key=lambda kv: -kv[1][0]):
    print(f'     {k:22s} {s_:7.2f} s {n_:4d} {s_ / max(n_, 1):7.3f}', flush=True)
res = dict(e=e, t_scf=ts, cycles=st['cycles'], outer=[x for x, _ in st['outer_hist']],
           timers={k: list(v) for k, v in st['timers'].items()})
if do_grad:
    sync(); t0 = time.perf_counter()
    g = mf.nuc_grad_method().kernel()
    sync(); tg = time.perf_counter() - t0
    g = np.asarray(g)
    print('TAXOL GRAD %.1f s  |sum g| %.1e  max|g| %.4f' % (tg, abs(g.sum(axis=0)).max(), abs(g).max()), flush=True)
    res.update(t_grad=tg, grad=g.tolist())
json.dump(res, open('/content/res/r7_taxol.json', 'w'), indent=1, default=float)
print('done')
