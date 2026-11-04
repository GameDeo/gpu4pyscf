#!/usr/bin/env python
'''
Per-component SCF noise floor: run an SCF with a fixed precision configuration for a fixed number of
cycles (never "converging") and print |g| and E per cycle.  The plateau of |g| is the noise floor of
that configuration; compare it with the polish target (conv_tol_grad * final_grad_factor, default 3e-5).

usage: noise_floor.py --vv10 fp32 --xcprec fp32 --kprec fp64 [--cycles 22]
'''
import argparse
import os
from pyscf import gto


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mol', default='033_Bisphenol_A')
    p.add_argument('--basis', default='def2-tzvp')
    p.add_argument('--vv10', default='fp64')
    p.add_argument('--xcprec', default='fp64')
    p.add_argument('--kprec', default='fp64')
    p.add_argument('--cycles', type=int, default=22)
    args = p.parse_args()

    from gpu4pyscf import dft
    from gpu4pyscf.dft import numint
    from gpu4pyscf.df import df_jk
    numint.VV10_PRECISION = args.vv10
    numint.XC_PRECISION = args.xcprec
    df_jk.K_PRECISION = args.kprec
    here = os.path.dirname(os.path.abspath(__file__))
    mol = gto.M(atom=os.path.join(here, '..', 'molecules', 'organic', args.mol + '.xyz'),
                basis=args.basis, verbose=0, max_memory=14000)
    mf = dft.RKS(mol, xc='wb97m-v').density_fit()
    mf.grids.atom_grid = (99, 590)
    mf.nlcgrids.atom_grid = (50, 194)
    mf.max_cycle = args.cycles
    mf.conv_tol = 1e-14
    mf.conv_tol_grad = 1e-14
    mf.conv_check = False
    trace = []

    def cb(env):
        trace.append((env['cycle'] + 1, float(env['e_tot']), float(env['norm_gorb'])))
    mf.callback = cb
    mf.kernel()
    print(f'# noise floor  vv10={args.vv10} xc={args.xcprec} k={args.kprec}  {args.mol}')
    for c, e, g in trace:
        print(f'  cyc {c:3d}  E = {e:.10f}  |g| = {g:.3e}')
    gs = [g for _, _, g in trace[-5:]]
    print(f'# plateau |g| (last 5 cycles): min {min(gs):.2e}  max {max(gs):.2e}   '
          f'E drift (last 5): {max(e for _, e, _ in trace[-5:]) - min(e for _, e, _ in trace[-5:]):.2e}')


if __name__ == '__main__':
    main()
