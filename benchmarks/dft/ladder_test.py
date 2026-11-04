#!/usr/bin/env python
'''
Experiment: coarse-grid / few-moduli early SCF stage followed by a full-grid, 11-modulus CRT stage.
usage: ladder_test.py --coarse 50 194 --n1 8 --g1 1e-3 [--mol 033_Bisphenol_A]
'''
import argparse, copy, os, time
import cupy
from pyscf import gto


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mol', default='033_Bisphenol_A')
    p.add_argument('--basis', default='def2-tzvp')
    p.add_argument('--coarse', type=int, nargs=2, default=(50, 194))
    p.add_argument('--n1', type=int, default=8, help='moduli (XC and K) in the coarse stage')
    p.add_argument('--n2', type=int, default=11)
    p.add_argument('--g1', type=float, default=1e-3, help='coarse-stage |g| threshold')
    p.add_argument('--tol', type=float, default=1e-9)
    p.add_argument('--vv10', default='fp32')
    p.add_argument('--ref', type=float, default=-731.6494045102813)
    args = p.parse_args()
    from gpu4pyscf import dft
    from gpu4pyscf.dft import numint
    from gpu4pyscf.df import df_jk
    here = os.path.dirname(os.path.abspath(__file__))
    mol = gto.M(atom=os.path.join(here, '..', 'molecules', 'organic', args.mol + '.xyz'),
                basis=args.basis, verbose=0, max_memory=14000)
    def make():
        mf = dft.RKS(mol, xc='wb97m-v').density_fit()
        mf.grids.atom_grid = (99, 590)
        mf.nlcgrids.atom_grid = (50, 194)
        return mf
    # warm-up on a tiny problem (JIT)
    w = gto.M(atom='O 0 0 0; H 0 .75 .5; H 0 -.75 .5', basis='def2-svp', verbose=0)
    wm = dft.RKS(w, xc='wb97m-v').density_fit(); wm.grids.atom_grid = (20, 50); wm.nlcgrids.atom_grid = (10, 26)
    numint.XC_PRECISION = df_jk.K_PRECISION = 'crt'
    numint.VV10_PRECISION = args.vv10
    wm.kernel()
    cupy.cuda.Stream.null.synchronize()

    mf = make()
    mf.conv_tol = args.tol
    mf.max_cycle = 50
    full = mf.grids
    t0 = time.perf_counter()
    dm = None
    if args.g1 > 0:
        g = copy.copy(full)
        g.reset()
        g.atom_grid = tuple(args.coarse)
        mf.grids = g
        numint.XC_CRT_MODULI = df_jk.K_CRT_MODULI = args.n1
        mf.conv_tol_grad = args.g1
        mf.conv_tol = 1e-2
        mf.kernel()
        cupy.cuda.Stream.null.synchronize()
        t1 = time.perf_counter()
        print(f'# stage A (coarse {args.coarse}, n={args.n1}): E={mf.e_tot:.10f} cycles={mf.cycles} time={t1 - t0:.1f} s')
        dm = mf.make_rdm1()
        mf.grids = full
        mf.conv_tol_grad = None
        mf.conv_tol = args.tol
        t0 = time.perf_counter()
    numint.XC_CRT_MODULI = df_jk.K_CRT_MODULI = args.n2
    mf.kernel(dm0=dm)
    cupy.cuda.Stream.null.synchronize()
    t2 = time.perf_counter()
    print(f'# stage B (full, n={args.n2}): E={mf.e_tot:.10f} cycles={mf.cycles} time={t2 - t0:.1f} s  converged={mf.converged}')
    print(f'# dE vs ref = {mf.e_tot - args.ref:.2e}')


if __name__ == '__main__':
    main()
