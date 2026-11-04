#!/usr/bin/env python
'''
Where does the FP32-XC Fock error come from?  At a fixed (partially converged) density compare the
FP64 Vxc with variants that reduce only the rho stage, only the Vxc-matrix stage, or both.  Reports the
Frobenius norm of the error projected on the occupied-virtual block, i.e. the orbital-gradient error
that limits the FP64 |g| at an FP32-converged density.
'''
import argparse, os
import cupy
import numpy as np
from pyscf import gto


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mol', default='033_Bisphenol_A')
    p.add_argument('--basis', default='def2-tzvp')
    p.add_argument('--cycles', type=int, default=6)
    args = p.parse_args()
    from gpu4pyscf import dft
    from gpu4pyscf.dft import numint
    here = os.path.dirname(os.path.abspath(__file__))
    mol = gto.M(atom=os.path.join(here, '..', 'molecules', 'organic', args.mol + '.xyz'),
                basis=args.basis, verbose=0, max_memory=14000)
    mf = dft.RKS(mol, xc='wb97m-v').density_fit()
    mf.grids.atom_grid = (99, 590)
    mf.nlcgrids.atom_grid = (50, 194)
    numint.VV10_PRECISION = 'fp32'
    mf.max_cycle = args.cycles
    mf.conv_tol = 1e-3
    mf.kernel()
    dm = mf.make_rdm1()
    mo = cupy.asarray(mf.mo_coeff); occ = cupy.asarray(mf.mo_occ)
    co, cv = mo[:, occ > 0], mo[:, occ == 0]
    mf.grids.build()
    ni = mf._numint

    def vxc(prec, parts):
        numint.XC_PRECISION, numint.XC_REDUCED_PARTS = prec, parts
        try:
            return ni.nr_rks(mol, mf.grids, mf.xc, dm)
        finally:
            numint.XC_PRECISION, numint.XC_REDUCED_PARTS = 'fp64', ('rho', 'vmat')
    n0, e0, v0 = vxc('fp64', ('rho', 'vmat'))
    print(f'# ref: Exc={e0:.10f}  N={float(n0):.8f}')
    import time
    for name, parts in [('rho only', ('rho',)), ('vmat only', ('vmat',)), ('both', ('rho', 'vmat'))]:
        n, e, v = vxc('fp32', parts)
        dv = v - v0
        g = float(cupy.linalg.norm(co.T @ dv @ cv))
        print(f'{name:10s} dExc={e - e0:+.3e}  max|dV|={float(cupy.abs(dv).max()):.2e}  |dg_ov|_F={g:.3e}')
    for chunk in (2048, 1024, 512, 256, 128, 64):
        numint.XC_VMAT_CHUNK = chunk
        vxc('fp32', ('rho', 'vmat'))
        cupy.cuda.Stream.null.synchronize(); t0 = time.perf_counter()
        n, e, v = vxc('fp32', ('rho', 'vmat'))
        cupy.cuda.Stream.null.synchronize(); dt = time.perf_counter() - t0
        dv = v - v0
        g = float(cupy.linalg.norm(co.T @ dv @ cv))
        print(f'chunk {chunk:5d}: max|dV|={float(cupy.abs(dv).max()):.2e}  |dg_ov|_F={g:.3e}   time {dt:.2f} s')
    numint.XC_VMAT_CHUNK = 0

if __name__ == '__main__':
    main()
