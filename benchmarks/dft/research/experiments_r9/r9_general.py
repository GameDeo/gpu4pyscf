'''Round 9: generalization matrix on one GPU.
Each case: stock FP64 DF at tight convergence (conv_tol 1e-11, conv_tol_grad 1e-7) as the reference, then the
multi-fidelity + mixed-precision defaults (and optional variants). Prints energy/gradient error and times.
usage: python r9_general.py "[(xyz, xc, basis, [variant dicts]), ...]"'''
import os, sys, time, json
import numpy as np
import cupy
from pyscf import gto
from gpu4pyscf import dft

here = os.path.dirname(os.path.abspath(__file__))
mdir = os.path.join(here, '..', '..', '..', 'molecules', 'organic')
cases = eval(sys.argv[1])


def sync():
    cupy.cuda.Stream.null.synchronize()


def warm(xc):
    w = gto.M(atom='O 0 0 0; H 0 .76 .59; H 0 -.76 .59', basis='def2-svp', verbose=0)
    for mfl in (False, True):
        wm = dft.RKS(w, xc=xc).density_fit()
        wm.grids.atom_grid = (20, 50)
        if mfl:
            wm.mixed_precision = {'scheme': 'crt'}
            wm.multifidelity = True
        wm.kernel(); wm.nuc_grad_method().kernel()


def new_mf(mol, xc):
    mf = dft.RKS(mol, xc=xc).density_fit()
    mf.grids.atom_grid = (99, 590)
    if xc.lower().endswith('-v'):
        mf.nlcgrids.atom_grid = (50, 194)
    mf.conv_tol = 1e-9
    return mf


def run(mf):
    sync(); t0 = time.perf_counter()
    e = mf.kernel()
    sync(); ts = time.perf_counter() - t0
    g = np.asarray(mf.nuc_grad_method().kernel())
    sync(); tg = time.perf_counter() - t0 - ts
    return e, g, ts, tg, bool(mf.converged)


out = []
for xyz, xc, basis, variants in cases:
    mol = gto.M(atom=os.path.join(mdir, xyz), basis=basis, verbose=0, max_memory=14000)
    warm(xc)
    tag = '%s %s/%s nao=%d' % (xyz.split('_', 1)[1][:-4], xc, basis, mol.nao)
    if os.environ.get('SKIP_REF') and os.path.exists('/content/res/ref_%s_%s_%s.npz' % (xyz, xc, basis)):
        z = np.load('/content/res/ref_%s_%s_%s.npz' % (xyz, xc, basis)); e0, g0, ts, tg, cv = float(z['e']), z['g'], 0., 0., True
    else:
        mf = new_mf(mol, xc); mf.conv_tol = 1e-11; mf.conv_tol_grad = 1e-7
        e0, g0, ts, tg, cv = run(mf)
        np.savez('/content/res/ref_%s_%s_%s.npz' % (xyz, xc, basis), e=e0, g=g0)
        mf = None; cupy.get_default_memory_pool().free_all_blocks()
    print('CASE %s  ref FP64 tight: SCF %.1f s GRAD %.1f s conv %s' % (tag, ts, tg, cv), flush=True)
    rec = dict(case=tag, ref=dict(e=e0, t_scf=ts, t_grad=tg), runs=[])
    if os.environ.get('REF_ONLY'):
        continue
    for v in [{}] + list(variants):
        mf = new_mf(mol, xc)
        mf.mixed_precision = {'scheme': 'crt'}
        vv = dict(v)
        if os.environ.get('DIAG'):
            vv['diag'] = True
        mf.multifidelity = vv or True
        if os.environ.get('VERB'):
            mf.verbose = int(os.environ['VERB'])
        try:
            e, g, ts, tg, cv = run(mf)
        except Exception as ex:
            import traceback; traceback.print_exc()
            print('   FAIL %s %r' % (v, ex), flush=True)
            mf = None; cupy.get_default_memory_pool().free_all_blocks(); continue
        st = getattr(mf, 'multifidelity_stats', {}) or {}
        dg = float(abs(g - g0).max())
        import resource
        print('   peak host RSS %.2f GB' % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6), flush=True)
        print('RESULT %-48s %-28s dE %.1e  max|dg| %.1e  SCF %.2f GRAD %.2f total %.2f conv %s outer %s' % (
            tag, str(v), e - e0, dg, ts, tg, ts + tg, cv, len(st.get('outer_hist', []))), flush=True)
        rec['runs'].append(dict(v=str(v), dE=e - e0, dg=dg, t_scf=ts, t_grad=tg, conv=cv))
        if st:
            em, oh = st.get('e_models', []), st.get('outer_hist', [])
            if em:
                print('   model-energy error vs next exact', ['%.1e' % (em[i] - oh[i + 1][1]) for i in range(min(len(em), len(oh) - 1))], flush=True)
            print('   cycles', st.get('cycles'), 'outer |g|', ['%.1e' % h_[0] for h_ in st.get('outer_hist', [])], flush=True)
            for k, (s_, n_) in sorted(st['timers'].items(), key=lambda kv: -kv[1][0])[:12]:
                print(f'     {k:22s} {s_:7.2f} s {n_:4d}', flush=True)
    out.append(rec)
    json.dump(out, open('/content/res/r9_general.json', 'w'), indent=1)
    mf = None; cupy.get_default_memory_pool().free_all_blocks()
print('ALLDONE')
