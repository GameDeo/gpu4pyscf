#!/usr/bin/env python
'''
Finite-difference check of the analytic DF gradient (including grid response) for a range-separated hybrid (+VV10).
Central differences of the converged DF energy w.r.t. one Cartesian coordinate of two atoms, fixed grid parameters.
usage: fd_gradient_check.py [--xc wb97m-v] [--basis def2-tzvp] [--h 0.002] [--mode mix_outside_kernel|mix_inside_kernel] [--scheme crt]
'''
import argparse
import numpy as np
from pyscf import gto
from gpu4pyscf import dft

ATOM = '''C 0.0 0.0 0.0; C 1.52 0.05 0.0; O -0.6 1.05 0.1; O -0.6 -1.05 -0.1; H 1.9 1.0 0.1;
          H 1.9 -0.5 0.88; H 1.9 -0.5 -0.88; H -1.55 1.0 0.0'''


def energy(coords, args):
    mol = gto.M(atom=[(s, tuple(c)) for s, c in zip(SYMS, coords)], basis=args.basis, unit='B', verbose=0)
    mf = dft.RKS(mol, xc=args.xc).density_fit()
    mf.grids.atom_grid = (99, 590)
    mf.grids.prune = None
    mf.nlcgrids.atom_grid = (50, 194)
    mf.nlcgrids.prune = None
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-8
    mf.max_cycle = 100
    if args.mode:
        mf.range_separated_mode = args.mode
    if args.scheme:
        mf.mixed_precision = {'scheme': args.scheme}
    e = mf.kernel()
    assert mf.converged
    return e, mf


def main():
    global SYMS
    p = argparse.ArgumentParser()
    p.add_argument('--xc', default='wb97m-v')
    p.add_argument('--basis', default='def2-tzvp')
    p.add_argument('--h', type=float, default=0.002)
    p.add_argument('--mode', default='')
    p.add_argument('--scheme', default='')
    args = p.parse_args()
    mol0 = gto.M(atom=ATOM, basis=args.basis, verbose=0)
    SYMS = [mol0.atom_symbol(i) for i in range(mol0.natm)]
    x0 = mol0.atom_coords()
    e0, mf = energy(x0, args)
    g = mf.nuc_grad_method()
    g.grid_response = True
    ga = g.kernel()
    g2 = mf.nuc_grad_method()
    g2.grid_response = False
    ga_nr = g2.kernel()
    print(f'# {args.xc}/{args.basis} mode={args.mode or "default"} scheme={args.scheme or "fp64"}  E0={e0:.10f}')
    worst = 0
    for atom, comp in ((0, 0), (2, 1), (3, 2), (6, 0)):
        xp = x0.copy(); xp[atom, comp] += args.h
        xm = x0.copy(); xm[atom, comp] -= args.h
        ep, _ = energy(xp, args)
        em, _ = energy(xm, args)
        fd = (ep - em) / (2 * args.h)
        d = ga[atom, comp] - fd
        worst = max(worst, abs(d))
        print(f'  atom {atom} xyz[{comp}]  FD={fd:+.8f}  analytic(grid resp)={ga[atom, comp]:+.8f}  diff={d:+.2e}  '
              f'(no grid resp: {ga_nr[atom, comp]:+.8f})')
    print(f'# max |analytic - FD| = {worst:.2e}')


if __name__ == '__main__':
    main()
