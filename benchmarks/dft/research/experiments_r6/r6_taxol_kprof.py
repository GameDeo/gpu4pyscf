import os, sys, time
import numpy as np, cupy
from pyscf import gto
from gpu4pyscf import dft
from gpu4pyscf.df.direct_k import DirectDFK
from gpu4pyscf.df import df as gdf
here = os.path.dirname(os.path.abspath(__file__))
name = sys.argv[1]; mem_frac = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25
mol = gto.M(atom=os.path.join(here, '..', '..', '..', 'molecules', 'organic', name + '.xyz'), basis='def2-tzvp', verbose=0)
nocc = mol.nelectron // 2
auxmol = gdf.addons.make_auxmol(mol, 'def2-universal-jkfit')
rng = np.random.default_rng(0)
S = cupy.asarray(mol.intor('int1e_ovlp'))
C = cupy.asarray(rng.standard_normal((mol.nao, nocc)))
w, V = cupy.linalg.eigh(C.T @ S @ C); C = C @ (V / cupy.sqrt(w)) @ V.T
dk = DirectDFK(mol, auxmol, None, nmod_t=15, nmod_x=11, mem_frac=mem_frac)
dk.timers = {}
t0 = time.perf_counter(); dk.kc_j(C * np.sqrt(2.), with_j=True); cupy.cuda.Stream.null.synchronize()
print(name, 'mem_frac', mem_frac, 'blksize', dk._setup['blksize'], 'total %.1f s' % (time.perf_counter() - t0),
      {k: round(v, 2) for k, v in dk.timers.items()}, 'pool %.2f GB' % (cupy.get_default_memory_pool().total_bytes() / 1e9), flush=True)
