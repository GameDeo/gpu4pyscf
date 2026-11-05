'''Taxol def2-TZVP on the T4: cost and memory of one direct exact DF exchange evaluation (FR + J, LR), no stored tensor.'''
import os, sys, time
import numpy as np, cupy
from pyscf import gto
from gpu4pyscf import dft
from gpu4pyscf.df.direct_k import DirectDFK
from gpu4pyscf.df import df as gdf
here = os.path.dirname(os.path.abspath(__file__))
name = sys.argv[1] if len(sys.argv) > 1 else '113_Taxol'
mol = gto.M(atom=os.path.join(here, '..', '..', '..', 'molecules', 'organic', name + '.xyz'), basis='def2-tzvp', verbose=0,
            max_memory=12000)
nocc = mol.nelectron // 2
mf = dft.RKS(mol, xc='wb97m-v').density_fit()
auxmol = gdf.addons.make_auxmol(mol, mf.with_df.auxbasis)
rng = np.random.default_rng(0)
S = cupy.asarray(mol.intor('int1e_ovlp'))
C = cupy.asarray(rng.standard_normal((mol.nao, nocc)))
w, V = cupy.linalg.eigh(C.T @ S @ C); C = C @ (V / cupy.sqrt(w)) @ V.T
pool = cupy.get_default_memory_pool()
for om, nm in ((None, (15, 11)), (0.3, (15, 11))):
    t0 = time.perf_counter()
    dk = DirectDFK(mol, auxmol, om, nmod_t=nm[0], nmod_x=nm[1])
    st = dk._prepare(nocc)
    cupy.cuda.Stream.null.synchronize(); t1 = time.perf_counter()
    kc, J = dk.kc_j(C * np.sqrt(2.), with_j=om is None)
    cupy.cuda.Stream.null.synchronize(); t2 = time.perf_counter()
    print('%s omega %s: nao %d (sorted cart %d) naux %d nocc %d  setup %.1f s  K C (+J) %.1f s  aux batches %d blksize %d '
          'peak pool %.2f GB' % (name, om, mol.nao, dk.mol.nao, st['naux'], nocc, t1 - t0, t2 - t1,
                                 len(st['aux_offsets']) - 1, st['blksize'], pool.total_bytes() / 1e9), flush=True)
    del dk, st, kc, J
    pool.free_all_blocks()
