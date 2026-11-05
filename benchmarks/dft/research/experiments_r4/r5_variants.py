'''BPA multi-fidelity SCF variants (SCF only): which component limits the outer contraction.'''
import os, sys, time, json
import numpy as np
import cupy
from pyscf import gto
from gpu4pyscf import dft

here = os.path.dirname(os.path.abspath(__file__))
mol = gto.M(atom=os.path.join(here, '..', '..', '..', 'molecules', 'organic', '033_Bisphenol_A.xyz'),
            basis='def2-tzvp', verbose=0, max_memory=14000)
ref = json.load(open(sys.argv[1]))['energy'] if len(sys.argv) > 1 else None
variants = eval(sys.argv[2]) if len(sys.argv) > 2 else [
    {'eig': 'fp64', 'k_increment': 'exact'},
    {'eig': 'fp64', 'k_increment': 'fp16'},
    {'eig': 'fp32oa', 'k_increment': 'fp16'},
]
w = gto.M(atom='O 0 0 0; H 0 .76 .59; H 0 -.76 .59', basis='def2-tzvp', verbose=0)
wm = dft.RKS(w, xc='wb97m-v').density_fit()
wm.grids.atom_grid = (20, 50); wm.nlcgrids.atom_grid = (10, 26)
wm.mixed_precision = {'scheme': 'crt'}
wm.multifidelity = {}
wm.kernel()
out = []
for v in variants:
    mf = dft.RKS(mol, xc='wb97m-v').density_fit()
    mf.grids.atom_grid = (99, 590)
    mf.nlcgrids.atom_grid = (50, 194)
    mf.conv_tol = 1e-9
    mf.mixed_precision = {'scheme': 'crt'}
    mf.multifidelity = dict(v)
    mf.verbose = 4
    cupy.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    e = mf.kernel()
    cupy.cuda.Stream.null.synchronize()
    t = time.perf_counter() - t0
    st = mf.multifidelity_stats
    print('VARIANT', v, 'E %.10f' % e, 'dE %.2e' % (e - ref) if ref else '', 'time %.1f s' % t, 'cycles', st['cycles'],
          flush=True)
    print('   outer |g|', ['%.1e' % g for g, _ in st['outer_hist']], flush=True)
    for k, (s_, n_) in sorted(st['timers'].items(), key=lambda kv: -kv[1][0]):
        print(f'     {k:16s} {s_:7.2f} s {n_:4d} {s_ / max(n_, 1):7.3f}', flush=True)
    out.append(dict(variant=v, e=e, t=t, cycles=st['cycles'], outer=st['outer_hist'],
                    timers={k: list(x) for k, x in st['timers'].items()}))
    json.dump(out, open('/content/res/r5_variants.json', 'w'), indent=1, default=float)
print('done')
