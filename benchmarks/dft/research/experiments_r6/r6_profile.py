'''BPA wB97M-V/def2-TZVP multi-fidelity SCF + gradient (T4), for an nsys trace.
Warm-up on water first (JIT), then the profiled region is bracketed by cudaProfilerStart/Stop.
NVTX: multi-fidelity timer keys (GPU4PYSCF_MF_NVTX=1) and gradient pieces.'''
import os, sys, time, json
import numpy as np
import cupy
from pyscf import gto
from gpu4pyscf import dft
from gpu4pyscf.grad import rks as rks_grad
from gpu4pyscf.df.grad import rhf as df_rhf_grad

here = os.path.dirname(os.path.abspath(__file__))
ref = json.load(open(sys.argv[1]))
opts = eval(sys.argv[2]) if len(sys.argv) > 2 else {}
molname = sys.argv[3] if len(sys.argv) > 3 else '033_Bisphenol_A'


def nvtx_wrap(mod, name, label):
    f = getattr(mod, name)

    def w(*a, **k):
        cupy.cuda.nvtx.RangePush(label)
        try:
            return f(*a, **k)
        finally:
            cupy.cuda.Stream.null.synchronize()
            cupy.cuda.nvtx.RangePop()
    setattr(mod, name, w)


nvtx_wrap(rks_grad, '_get_denlc', 'g:NLC')
nvtx_wrap(df_rhf_grad, '_jk_energy_per_atom', 'g:DF_JK')
nvtx_wrap(df_rhf_grad, '_j_energy_per_atom', 'g:DF_J')


def build(mol):
    mf = dft.RKS(mol, xc='wb97m-v').density_fit()
    mf.mixed_precision = {'scheme': 'crt'}
    mf.multifidelity = dict(opts) or True
    return mf


w = gto.M(atom='O 0 0 0; H 0 .76 .59; H 0 -.76 .59', basis='def2-tzvp', verbose=0)
wm = build(w)
wm.grids.atom_grid = (20, 50); wm.nlcgrids.atom_grid = (10, 26)
wm.kernel(); wm.nuc_grad_method().kernel()

mol = gto.M(atom=os.path.join(here, '..', '..', '..', 'molecules', 'organic', molname + '.xyz'),
            basis='def2-tzvp', verbose=0, max_memory=14000)
mf = build(mol)
mf.grids.atom_grid = (99, 590)
mf.nlcgrids.atom_grid = (50, 194)
mf.conv_tol = 1e-9
cupy.cuda.Stream.null.synchronize()
cupy.cuda.profiler.start()
t0 = time.perf_counter()
cupy.cuda.nvtx.RangePush('SCF')
e = mf.kernel()
cupy.cuda.Stream.null.synchronize()
cupy.cuda.nvtx.RangePop()
t_scf = time.perf_counter() - t0
cupy.cuda.nvtx.RangePush('GRAD')
t1 = time.perf_counter()
g = mf.nuc_grad_method().kernel()
cupy.cuda.Stream.null.synchronize()
t_g = time.perf_counter() - t1
cupy.cuda.nvtx.RangePop()
cupy.cuda.profiler.stop()
st = mf.multifidelity_stats
dg = float(abs(np.asarray(g) - np.asarray(ref['grad'])).max()) if 'grad' in ref else float('nan')
print('SCF %.2f s GRAD %.2f s dE %.2e dg %.2e cycles %s' % (t_scf, t_g, e - ref['energy'], dg, st['cycles']), flush=True)
for k, (s_, n_) in sorted(st['timers'].items(), key=lambda kv: -kv[1][0]):
    print(f'   {k:16s} {s_:7.2f} s {n_:4d}', flush=True)
