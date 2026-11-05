'''BPA: multi-fidelity SCF (+ fused XC gradient) then a timed breakdown of the nuclear gradient.'''
import os, sys, time, json
import numpy as np
import cupy
from pyscf import gto
from gpu4pyscf import dft
from gpu4pyscf.grad import rks as rks_grad
from gpu4pyscf.df.grad import rhf as df_rhf_grad, rks as df_rks_grad

here = os.path.dirname(os.path.abspath(__file__))
ref = json.load(open(sys.argv[1]))
opts = eval(sys.argv[2]) if len(sys.argv) > 2 else {}
T = {}


def timed(mod, name, label):
    f = getattr(mod, name)

    def w(*a, **k):
        cupy.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        r = f(*a, **k)
        cupy.cuda.Stream.null.synchronize()
        v = T.setdefault(label, [0.0, 0])
        v[0] += time.perf_counter() - t0
        v[1] += 1
        return r
    setattr(mod, name, w)


timed(rks_grad, '_get_denlc', 'NLC grad')
timed(rks_grad, 'get_exc', 'XC grad (unfused)')
timed(df_rhf_grad, '_jk_energy_per_atom', 'DF J/K grad')
timed(df_rhf_grad, '_j_energy_per_atom', 'DF J grad')
timed(df_rhf_grad, 'int2c2e_ip1_per_atom', '  aux metric deriv')

mol = gto.M(atom=os.path.join(here, '..', '..', '..', 'molecules', 'organic', '033_Bisphenol_A.xyz'),
            basis='def2-tzvp', verbose=0, max_memory=14000)
w = gto.M(atom='O 0 0 0; H 0 .76 .59; H 0 -.76 .59', basis='def2-tzvp', verbose=0)
wm = dft.RKS(w, xc='wb97m-v').density_fit()
wm.grids.atom_grid = (20, 50); wm.nlcgrids.atom_grid = (10, 26)
wm.mixed_precision = {'scheme': 'crt'}
wm.multifidelity = dict(opts) or True
wm.kernel()
wm.nuc_grad_method().kernel()
T.clear()

mf = dft.RKS(mol, xc='wb97m-v').density_fit()
mf.grids.atom_grid = (99, 590)
mf.nlcgrids.atom_grid = (50, 194)
mf.conv_tol = 1e-9
mf.mixed_precision = {'scheme': 'crt'}
mf.multifidelity = dict(opts) or True
cupy.cuda.Stream.null.synchronize()
t0 = time.perf_counter()
e = mf.kernel()
cupy.cuda.Stream.null.synchronize()
t_scf = time.perf_counter() - t0
st = mf.multifidelity_stats
print('SCF %.2f s  E %.10f dE %.2e  cycles %s  outer %s' % (t_scf, e, e - ref['energy'], st['cycles'],
      ['%.1e' % g for g, _ in st['outer_hist']]), flush=True)
for k, (s_, n_) in sorted(st['timers'].items(), key=lambda kv: -kv[1][0]):
    print(f'   {k:16s} {s_:7.2f} s {n_:4d} {s_ / max(n_, 1):7.3f}', flush=True)
T.clear()
t0 = time.perf_counter()
gobj = mf.nuc_grad_method()
gobj.verbose = 6
g = gobj.kernel()
cupy.cuda.Stream.null.synchronize()
t_g = time.perf_counter() - t0
dg = float(abs(np.asarray(g) - np.asarray(ref['grad'])).max())
print('GRAD %.2f s  max|dg| %.2e  fused XC used %s' % (t_g, dg, mf._mf_xc_grad is not None), flush=True)
for k, (s_, n_) in T.items():
    print(f'   {k:20s} {s_:7.2f} s {n_:3d}', flush=True)
print('   rest (h1e/overlap/W/nuc)', '%.2f s' % (t_g - sum(v[0] for k, v in T.items() if not k.startswith(' '))))
json.dump(dict(t_scf=t_scf, e=e, t_grad=t_g, dg=dg, cycles=st['cycles'], grad_timers=T,
               scf_timers={k: list(v) for k, v in st['timers'].items()}),
          open('/content/res/r5_grad_%s.json' % ('_'.join(f'{k}{v}' for k, v in opts.items()) or 'default'), 'w'),
          default=float, indent=1)
print('done')
