'''Round 9: Taxol exact-K checkpoint (DirectDFK, full-range operator) on T4: time split integrals / unpack /
half transforms / metric, and the effect of fewer CRT moduli in the GEMMs (accuracy of K C and time).'''
import os, time
import numpy as np, cupy
from pyscf import gto
from gpu4pyscf import dft
from gpu4pyscf.scf import mixed_precision as mp
from gpu4pyscf.df.direct_k import DirectDFK
here = os.path.dirname(os.path.abspath(__file__))
mol = gto.M(atom=os.path.join(here, '..', '..', '..', 'molecules', 'organic', '113_Taxol.xyz'),
            basis='def2-tzvp', verbose=0, max_memory=40000)
mf = dft.RKS(mol, xc='wb97m-v').density_fit()
mf.mixed_precision = {'scheme': 'crt'}
mf.with_df.build(build_cderi=False) if 'build_cderi' in mf.with_df.build.__code__.co_varnames else None
nocc = mol.nelectron // 2
print('nao', mol.nao, 'nocc', nocc, 'naux', mf.with_df.auxmol.nao, flush=True)
dm = cupy.asarray(mf.get_init_guess())
s = cupy.asarray(mf.get_ovlp())
w, U = cupy.linalg.eigh(s); X = U / cupy.sqrt(w); SX = s @ X
n, V = cupy.linalg.eigh(SX.T @ dm @ SX)
C = X @ V[:, -nocc:] * np.sqrt(2.0)          # occupation-weighted orbitals of the guess's top natural orbitals
sync = cupy.cuda.Stream.null.synchronize
ref = None
with mp.final_stage(mf):
    for nm in [(None, None), (15, 11), (8, 6), (5, 4)]:
        dk = DirectDFK(mol, mf.with_df.auxmol, None, nmod_t=nm[0], nmod_x=nm[1])
        dk.timers = {}
        sync(); t = time.perf_counter()
        kc, _ = dk.kc_j(C, with_j=False)
        sync(); tt = time.perf_counter() - t
        if ref is None:
            ref = kc
        err = float(abs(kc - ref).max() / abs(ref).max())
        print('nmod %s: total %.1f s  rel err vs default %.1e   ' % (nm, tt, err) +
              '  '.join('%s %.1f' % kv for kv in sorted(dk.timers.items(), key=lambda kv: -kv[1])), flush=True)
        dk = None; cupy.get_default_memory_pool().free_all_blocks()
