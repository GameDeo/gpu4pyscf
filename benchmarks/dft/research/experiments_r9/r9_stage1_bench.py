'''Round 9: where does the stage-1 surrogate K time go on BPA (T4)?  Times the SVP-RI surrogate DF build
(integrals + metric), k_ref and k_abs separately, inside the mixed-precision final-stage context.'''
import os, time
import numpy as np, cupy
from pyscf import gto
from gpu4pyscf import dft
from gpu4pyscf.scf import multifidelity as mfl, mixed_precision as mp
here = os.path.dirname(os.path.abspath(__file__))
mol = gto.M(atom=os.path.join(here, '..', '..', '..', 'molecules', 'organic', '033_Bisphenol_A.xyz'),
            basis='def2-tzvp', verbose=0, max_memory=14000)
mf = dft.RKS(mol, xc='wb97m-v').density_fit()
mf.mixed_precision = {'scheme': 'crt'}
omega, alpha, hyb = mf._numint.rsh_and_hybrid_coeff('wb97m-v')
sync = cupy.cuda.Stream.null.synchronize


def T(label, f, n=1):
    sync(); t = time.perf_counter()
    for _ in range(n):
        r = f()
    sync(); print('%-40s %.3f s' % (label, (time.perf_counter() - t) / n), flush=True)
    return r


dm = cupy.asarray(mf.get_init_guess())
w, V = cupy.linalg.eigh(cupy.asarray(mf.get_ovlp()))
C = (V / cupy.sqrt(w)) [:, -mol.nelectron // 2:]
with mp.final_stage(mf):
    for rep in range(2):
        sk = mfl.SurrogateK(mf, 'def2-svp-ri', omega, alpha, hyb, mode='fp32')
        from gpu4pyscf.df import df as gdf_mod
        old = gdf_mod.CDERI_CRT_MODULI; gdf_mod.CDERI_CRT_MODULI = min(old, 8)
        with sk.dfs.range_coulomb(omega, alpha, hyb) as d:
            T('surrogate DF build (SVP-RI, RSH)', lambda: d.build(omega=omega, lr_factor=alpha, sr_factor=hyb))
        gdf_mod.CDERI_CRT_MODULI = old
        sk._built = True
        T('k_ref (DF-K, CRT)', lambda: sk.k_ref(dm), 3)
        T('ensure (unpack FP32 B)', sk.ensure)
        print('B shape', sk.B.shape, flush=True)
        T('k_abs (FP32)', lambda: sk.k_abs(C), 5)
        s1e = cupy.asarray(mf.get_ovlp())
        k1 = T('k_dm (natural orbitals of the guess)', lambda: sk.k_dm(dm, s1e))
        k0 = sk.k_ref(dm)
        print('k_dm vs k_ref rel err %.1e' % float(abs(k1 - k0).max() / abs(k0).max()), flush=True)
        sk = None; cupy.get_default_memory_pool().free_all_blocks()
    # exact jkfit FR tensor build for comparison
    from gpu4pyscf.df import df as gdf
    dx = gdf.DF(mol, auxbasis='def2-universal-jkfit')
    T('exact DF build (jkfit, FR)', dx.build)
    print('naux surrogate', sk, 'nao', mol.nao)
