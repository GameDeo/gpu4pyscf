#!/usr/bin/env python
'''A/B of numint.XC_MERGE_VMAT_GEMM (one K=4*blk GEMM vs four GEMMs per block) and XC_FUSED on nr_rks wall time.'''
import os, time, cupy
from pyscf import gto
from gpu4pyscf import dft
from gpu4pyscf.dft import numint
here = os.path.dirname(os.path.abspath(__file__))
mol = gto.M(atom=os.path.join(here, '..', 'molecules', 'organic', '033_Bisphenol_A.xyz'), basis='def2-tzvp', verbose=0, max_memory=30000)
mf = dft.RKS(mol, xc='wb97m-v').density_fit()
mf.grids.atom_grid = (99, 590); mf.nlcgrids.atom_grid = (50, 194)
mf.max_cycle = 3; mf.conv_tol = 1e-3
mf.kernel(); dm = mf.make_rdm1(); mf.grids.build()
ni = mf._numint
for fused, merge in ((False, True), (True, False), (True, True)):
    numint.XC_FUSED, numint.XC_MERGE_VMAT_GEMM = fused, merge
    ni.nr_rks(mol, mf.grids, mf.xc, dm); cupy.cuda.Stream.null.synchronize()
    best = 1e9
    for _ in range(3):
        t = time.perf_counter(); r = ni.nr_rks(mol, mf.grids, mf.xc, dm); cupy.cuda.Stream.null.synchronize()
        best = min(best, time.perf_counter() - t)
    print(f'XC_FUSED={fused} MERGE={merge}: nr_rks {best:.3f} s   E_xc={float(r[1]):.10f}', flush=True)
