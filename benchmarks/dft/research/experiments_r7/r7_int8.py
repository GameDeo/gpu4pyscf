'''INT8-stored surrogate tensors (df.int8_df): operator accuracy on BPA, then the end-to-end multi-fidelity SCF +
gradient with exact_k='direct', surrogate_store='int8', surrogate_j='int8' (the Taxol configuration) vs reference.'''
import os, sys, time, json
import numpy as np
import cupy
from pyscf import gto
from gpu4pyscf import dft
from gpu4pyscf.df import df as gdf
from gpu4pyscf.df.int8_df import Int8DF

here = os.path.dirname(os.path.abspath(__file__))
ref = json.load(open(os.path.join(here, '..', '..', 'reference', 'bpa_tight.json')))
mol = gto.M(atom=os.path.join(here, '..', '..', '..', 'molecules', 'organic', '033_Bisphenol_A.xyz'),
            basis='def2-tzvp', verbose=0, max_memory=14000)
sync = cupy.cuda.Stream.null.synchronize
part = sys.argv[1] if len(sys.argv) > 1 else 'all'

if part in ('ops', 'all'):
    mf = dft.RKS(mol, xc='wb97m-v').density_fit()
    mf.grids.atom_grid = (30, 110); mf.nlcgrids.atom_grid = (10, 26)
    mf.conv_tol = 1e-6
    mf.kernel()
    dm = mf.make_rdm1()
    C = mf.mo_coeff[:, mf.mo_occ > 0] * np.sqrt(2.0)
    rng = np.random.default_rng(1)
    dC = cupy.asarray(rng.standard_normal(C.shape)) * 1e-4
    # J
    t0 = time.perf_counter()
    js = Int8DF(mol, 'def2-universal-jfit').build()
    sync(); print('J tensor build %.2f s  naux %d npairs %d' % (time.perf_counter() - t0, js.naux, js.npairs))
    dfj = gdf.DF(mol, auxbasis='def2-universal-jfit').build()
    Jx = dfj.get_jk(dm, with_k=False)[0]
    sync(); t0 = time.perf_counter()
    J8 = js.j(dm)
    sync(); print('J_s(D) %.4f s  rel err %.2e' % (time.perf_counter() - t0, float(abs(J8 - Jx).max() / abs(Jx).max())))
    dD = (C + dC) @ (C + dC).T - C @ C.T
    Jdx = dfj.get_jk(dD, with_k=False)[0]
    Jd8 = js.j(dD)
    print('J_s(dD) rel err %.2e' % float(abs(Jd8 - Jdx).max() / abs(Jdx).max()))
    js = dfj = None
    # K (wB97M-V operator, SVP-RI)
    omega, alpha, hyb = mf._numint.rsh_and_hybrid_coeff(mf.xc)
    from gpu4pyscf.scf.multifidelity import SurrogateK, SurrogateK8
    skd = SurrogateK(mf, 'def2-svp-ri', omega, alpha, hyb, mode='fp32')
    Kx = skd.k_ref(dm)
    t0 = time.perf_counter()
    sk8 = SurrogateK8(mf, 'def2-svp-ri', omega, alpha, hyb)
    sk8.ensure()
    sync(); print('K tensor build %.2f s' % (time.perf_counter() - t0))
    sync(); t0 = time.perf_counter()
    K8 = sk8.k_ref(dm)
    sync(); print('K_s(D) %.3f s  rel err %.2e' % (time.perf_counter() - t0, float(abs(K8 - Kx).max() / abs(Kx).max())))
    Co = C / np.sqrt(2.0)
    dCo = dC / np.sqrt(2.0)
    skd.set_ref(Co, None); sk8.set_ref(Co, None)
    Kdx = skd.k_inc_dC(dCo)
    sync(); t0 = time.perf_counter()
    Kd8 = sk8.k_inc_dC(dCo)
    sync(); print('K_s inc %.3f s  rel err %.2e (vs dense fp32 surrogate)' % (time.perf_counter() - t0,
                                                                           float(abs(Kd8 - Kdx).max() / abs(Kdx).max())))
    # exact increment of the dense surrogate in FP64 for reference
    Dn = 2 * (Co + dCo) @ (Co + dCo).T
    Kdt = skd.k_ref(Dn) - skd.k_ref(2 * Co @ Co.T)
    print('   dense fp32 inc rel err %.2e, int8 inc rel err %.2e (vs FP64 difference)' % (
        float(abs(Kdx - Kdt).max() / abs(Kdt).max()), float(abs(Kd8 - Kdt).max() / abs(Kdt).max())))
    skd = sk8 = mf = None
    cupy.get_default_memory_pool().free_all_blocks()

if part in ('scf', 'all'):
    for v in eval(sys.argv[2]) if len(sys.argv) > 2 else [{'exact_k': 'direct', 'surrogate_store': 'int8', 'surrogate_j': 'int8'}]:
        mf = dft.RKS(mol, xc='wb97m-v').density_fit()
        mf.grids.atom_grid = (99, 590); mf.nlcgrids.atom_grid = (50, 194)
        mf.conv_tol = 1e-9
        mf.mixed_precision = {'scheme': 'crt'}
        mf.multifidelity = dict(v) or True
        sync(); t0 = time.perf_counter()
        e = mf.kernel()
        sync(); ts = time.perf_counter() - t0
        g = mf.nuc_grad_method().kernel()
        sync(); tg = time.perf_counter() - t0 - ts
        st = mf.multifidelity_stats
        print('RESULT', v, 'dE %.2e SCF %.2f s GRAD %.2f s max|dg| %.1e' % (
            e - ref['energy'], ts, tg, abs(np.asarray(g) - np.asarray(ref['grad'])).max()), flush=True)
        print('   cycles', st['cycles'], 'outer |g|', ['%.1e' % x for x, _ in st['outer_hist']], flush=True)
        for k, (s_, n_) in sorted(st['timers'].items(), key=lambda kv: -kv[1][0]):
            print(f'     {k:22s} {s_:7.2f} s {n_:4d} {s_ / max(n_, 1):7.3f}', flush=True)
print('done')
