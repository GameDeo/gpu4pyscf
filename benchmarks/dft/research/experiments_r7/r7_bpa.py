'''Round 7, BPA wB97M-V/def2-TZVP on T4.
mode "base": stock FP64 gpu4pyscf (DF, default conv_tol 1e-9 and conv_tol_grad) -> the "standard thresholds" bar.
mode "mf": multi-fidelity variants (argv[3], list of option dicts) with gradient.
mode "tiers": one multi-fidelity SCF, then the gradient with single (li, lj) shell-pair groups in the FP32 Rys tier.'''
import os, sys, time, json
import numpy as np
import cupy
from pyscf import gto
from gpu4pyscf import dft
from gpu4pyscf.df.grad import rhf as dgr

here = os.path.dirname(os.path.abspath(__file__))
ref = json.load(open(os.path.join(here, '..', '..', 'reference', 'bpa_tight.json')))
mode = sys.argv[1]
tag = sys.argv[2]
mol = gto.M(atom=os.path.join(here, '..', '..', '..', 'molecules', 'organic', '033_Bisphenol_A.xyz'),
            basis='def2-tzvp', verbose=0, max_memory=14000)


def sync():
    cupy.cuda.Stream.null.synchronize()


def warm():
    w = gto.M(atom='O 0 0 0; H 0 .76 .59; H 0 -.76 .59', basis='def2-tzvp', verbose=0)
    wm = dft.RKS(w, xc='wb97m-v').density_fit()
    wm.grids.atom_grid = (20, 50); wm.nlcgrids.atom_grid = (10, 26)
    if mode != 'base':
        wm.mixed_precision = {'scheme': 'crt'}
        wm.multifidelity = True
    wm.kernel(); wm.nuc_grad_method().kernel()


def new_mf():
    mf = dft.RKS(mol, xc='wb97m-v').density_fit()
    mf.grids.atom_grid = (99, 590)
    mf.nlcgrids.atom_grid = (50, 194)
    mf.conv_tol = 1e-9
    return mf


def run(mf, label):
    sync(); t0 = time.perf_counter()
    e = mf.kernel()
    sync(); ts = time.perf_counter() - t0
    g = mf.nuc_grad_method().kernel()
    sync(); tg = time.perf_counter() - t0 - ts
    r = dict(label=label, e=e, dE=e - ref['energy'], t_scf=ts, t_grad=tg,
             dg=float(abs(np.asarray(g) - np.asarray(ref['grad'])).max()))
    st = getattr(mf, 'multifidelity_stats', None)
    print('RESULT', label, 'dE %.2e  SCF %.2f s  GRAD %.2f s  total %.2f s  max|dg| %.1e' %
          (r['dE'], ts, tg, ts + tg, r['dg']), flush=True)
    if st:
        r.update(cycles=st['cycles'], outer=[g_ for g_, _ in st['outer_hist']], timers={k: list(x) for k, x in st['timers'].items()})
        print('   cycles', st['cycles'], 'outer |g|', ['%.1e' % g_ for g_, _ in st['outer_hist']], flush=True)
        for k, (s_, n_) in sorted(st['timers'].items(), key=lambda kv: -kv[1][0]):
            print(f'     {k:22s} {s_:7.2f} s {n_:4d} {s_ / max(n_, 1):7.3f}', flush=True)
    else:
        print('   cycles', getattr(mf, 'cycles', None), flush=True)
    return r, g


warm()
out = []
if mode == 'base':
    for rep in range(int(sys.argv[3]) if len(sys.argv) > 3 else 1):
        out.append(run(new_mf(), 'fp64-default')[0])
elif mode == 'mf':
    for v in eval(sys.argv[3]):
        mf = new_mf()
        mf.mixed_precision = {'scheme': 'crt'}
        mf.multifidelity = dict(v) or True
        out.append(run(mf, str(v))[0])
elif mode == 'tiers':
    mf = new_mf()
    mf.mixed_precision = {'scheme': 'crt'}
    mf.multifidelity = True
    r, g0 = run(mf, 'mf default')
    out.append(r)
    from gpu4pyscf.scf import mixed_precision as mp
    T = {}
    f0 = dgr._jk_energy_per_atom

    def timed(*a, **k):
        sync(); t0 = time.perf_counter()
        res = f0(*a, **k)
        sync(); T['jk'] = T.get('jk', 0) + time.perf_counter() - t0
        return res
    dgr._jk_energy_per_atom = timed

    def grad_with(tiers):
        # crt grad_stage would overwrite GRAD_TIERS: set the option it reads
        old = mp.DEFAULTS.get('grad_fp32_frac') if hasattr(mp, 'DEFAULTS') else None
        T.clear()
        dgr_tiers = tiers
        orig = mp.grad_stage
        import contextlib

        @contextlib.contextmanager
        def gs(*a, **k):
            with orig(*a, **k) as x:
                dgr.GRAD_TIERS = dgr_tiers
                yield x
        mp.grad_stage = gs
        try:
            sync(); t0 = time.perf_counter()
            g = mf.nuc_grad_method().kernel()
            sync()
            return np.asarray(g), time.perf_counter() - t0, T.get('jk', 0)
        finally:
            mp.grad_stage = orig
    gfp64, t64, jk64 = grad_with(None)
    print('FP64 tier only: grad %.2f s  DF_JK %.2f s  max|dg| vs ref %.1e' %
          (t64, jk64, abs(gfp64 - np.asarray(ref['grad'])).max()), flush=True)
    if len(sys.argv) > 3:
        res = []
        for tv in eval(sys.argv[3]):
            g, t, jk = grad_with(tv)
            d = float(abs(g - gfp64).max())
            print('TIERS %-40s grad %.2f s  DF_JK %.2f s (fp64 %.2f)  extra max|dg| %.1e  vs ref %.1e' % (
                tv, t, jk, jk64, d, abs(g - np.asarray(ref['grad'])).max()), flush=True)
            res.append(dict(tiers=tv, t=t, jk=jk, d=d))
        out.append(dict(fp64=dict(t=t64, jk=jk64), variants=res))
    else:
        grad_with({'groups': []})
        labels = list(dgr.TIER_GROUP_LABELS)
        print('groups', labels, flush=True)
        res = []
        for gi, lab in enumerate(labels):
            g, t, jk = grad_with({'groups': [gi]})
            d = float(abs(g - gfp64).max())
            print('GROUP %2d l=%s  DF_JK %.2f s (fp64 %.2f)  extra max|dg| %.1e' % (gi, lab, jk, jk64, d), flush=True)
            res.append(dict(group=gi, l=lab, jk=jk, d=d))
        out.append(dict(fp64=dict(t=t64, jk=jk64), groups=res))
json.dump(out, open('/content/res/r7_%s.json' % tag, 'w'), indent=1, default=float)
print('done')
