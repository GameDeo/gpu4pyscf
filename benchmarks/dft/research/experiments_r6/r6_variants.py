'''BPA multi-fidelity variants: SCF time, outer levels/residuals, dE; optional gradient (argv[3] == "grad").'''
import os, sys, time, json
import numpy as np
import cupy
from pyscf import gto
from gpu4pyscf import dft

here = os.path.dirname(os.path.abspath(__file__))
ref = json.load(open(sys.argv[1]))
variants = eval(sys.argv[2])
do_grad = len(sys.argv) > 3 and sys.argv[3] == 'grad'
tag = sys.argv[4] if len(sys.argv) > 4 else 'variants'
mol = gto.M(atom=os.path.join(here, '..', '..', '..', 'molecules', 'organic', '033_Bisphenol_A.xyz'),
            basis='def2-tzvp', verbose=0, max_memory=14000)
w = gto.M(atom='O 0 0 0; H 0 .76 .59; H 0 -.76 .59', basis='def2-tzvp', verbose=0)
for v in variants[:1]:
    wm = dft.RKS(w, xc='wb97m-v').density_fit()
    wm.grids.atom_grid = (20, 50); wm.nlcgrids.atom_grid = (10, 26)
    wm.mixed_precision = {'scheme': 'crt'}
    wm.multifidelity = dict(v)
    wm.kernel()
    if do_grad:
        wm.nuc_grad_method().kernel()
out = []
for v in variants:
    mf = dft.RKS(mol, xc='wb97m-v').density_fit()
    mf.grids.atom_grid = (99, 590)
    mf.nlcgrids.atom_grid = (50, 194)
    mf.conv_tol = 1e-9
    mf.mixed_precision = {'scheme': 'crt'}
    mf.multifidelity = dict(v)
    cupy.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    e = mf.kernel()
    cupy.cuda.Stream.null.synchronize()
    t = time.perf_counter() - t0
    st = mf.multifidelity_stats
    r = dict(variant=v, e=e, dE=e - ref['energy'], t_scf=t, cycles=st['cycles'], levels=st.get('levels'),
             outer=[g for g, _ in st['outer_hist']], timers={k: list(x) for k, x in st['timers'].items()})
    if do_grad:
        t0 = time.perf_counter()
        g = mf.nuc_grad_method().kernel()
        cupy.cuda.Stream.null.synchronize()
        r['t_grad'] = time.perf_counter() - t0
        r['dg'] = float(abs(np.asarray(g) - np.asarray(ref['grad'])).max())
    print('VARIANT', v, 'dE %.2e' % r['dE'], 'SCF %.2f s' % t, 'GRAD %.2f s dg %.1e' % (r['t_grad'], r['dg']) if do_grad else '',
          'cycles', st['cycles'], 'levels', st.get('levels'), flush=True)
    print('   outer |g|', ['%.1e' % g for g in r['outer']], flush=True)
    for k, (s_, n_) in sorted(st['timers'].items(), key=lambda kv: -kv[1][0]):
        print(f'     {k:22s} {s_:7.2f} s {n_:4d} {s_ / max(n_, 1):7.3f}', flush=True)
    out.append(r)
    json.dump(out, open('/content/res/r6_%s.json' % tag, 'w'), indent=1, default=float)
print('done')
