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
    p.add_argument('--g0', type=float, default=3e-2)
    p.add_argument('--g1', type=float, default=1e-3, help='coarse-stage |g| threshold')
    p.add_argument('--basis0', default='', help='small basis for a cheap first SCF whose density seeds the coarse stage')
    p.add_argument('--n2a', type=int, default=0, help='moduli of the first full-grid cycles (until |g| < g2); 0 = n2')
    p.add_argument('--g2', type=float, default=1e-4)
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

    dm_seed = None
    if args.basis0:
        t00 = time.perf_counter()
        from pyscf import scf as pyscf_scf
        mol0 = gto.M(atom=mol.atom, basis=args.basis0, verbose=0, max_memory=14000)
        mf0 = dft.RKS(mol0, xc='wb97m-v').density_fit()
        mf0.grids.atom_grid = tuple(args.coarse)
        mf0.nlcgrids.atom_grid = (50, 194)
        numint.XC_CRT_MODULI = df_jk.K_CRT_MODULI = args.n1
        mf0.conv_tol = 1e-2
        mf0.conv_tol_grad = args.g0
        mf0.kernel()
        cupy.cuda.Stream.null.synchronize()
        dm_seed = pyscf_scf.addons.project_dm_nr2nr(mol0, mf0.make_rdm1().get(), mol)
        dm_seed = cupy.asarray(dm_seed)
        print(f'# stage 0 ({args.basis0}): E={mf0.e_tot:.8f} cycles={mf0.cycles} time={time.perf_counter() - t00:.1f} s')
        del mf0
    mf = make()
    mf.conv_tol = args.tol
    mf.max_cycle = 50
    full = mf.grids
    t0 = time.perf_counter()
    dm = dm_seed
    if args.g1 > 0:
        g = copy.copy(full)
        g.reset()
        g.atom_grid = tuple(args.coarse)
        mf.grids = g
        numint.XC_CRT_MODULI = df_jk.K_CRT_MODULI = args.n1
        mf.conv_tol_grad = args.g1
        mf.conv_tol = 1e-2
        mf.kernel(dm0=dm)
        cupy.cuda.Stream.null.synchronize()
        t1 = time.perf_counter()
        print(f'# stage A (coarse {args.coarse}, n={args.n1}): E={mf.e_tot:.10f} cycles={mf.cycles} time={t1 - t0:.1f} s')
        dm = mf.make_rdm1()
        mf.grids = full
        mf.conv_tol_grad = None
        mf.conv_tol = args.tol
        t0 = time.perf_counter()
    numint.XC_CRT_MODULI = df_jk.K_CRT_MODULI = args.n2a or args.n2
    if args.n2a:
        def cb(env):
            n = args.n2 if float(env['norm_gorb']) < args.g2 else args.n2a
            numint.XC_CRT_MODULI = df_jk.K_CRT_MODULI = n
        mf.callback = cb
    mf.kernel(dm0=dm)
    cupy.cuda.Stream.null.synchronize()
    t2 = time.perf_counter()
    print(f'# stage B (full, n={args.n2}): E={mf.e_tot:.10f} cycles={mf.cycles} time={t2 - t0:.1f} s  converged={mf.converged}')
    print(f'# dE vs ref = {mf.e_tot - args.ref:.2e}')


if __name__ == '__main__':
    main()
