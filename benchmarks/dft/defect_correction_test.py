#!/usr/bin/env python
'''
E2: is the reduced-precision error of K / Vxc a systematic (nearly linear-in-D) function of the density?

Defect correction:  F(D) ~ F_low(D) + [F_hi(D0) - F_low(D0)].
It works iff e(D) = F_low(D) - F_hi(D) is almost the same for D and D0, i.e. |e(D) - e(D0)| << |e(D)|
for the |D - D0| that occur during the SCF.  Rounding noise that decorrelates between D and D0
(pseudo-random accumulation error) would NOT be removed.  This script measures it on a real SCF trajectory.
'''
import argparse
import os
import numpy as np
import cupy
from pyscf import gto


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mol', default='033_Bisphenol_A')
    p.add_argument('--basis', default='def2-tzvp')
    p.add_argument('--ref_cycle', type=int, default=5)
    p.add_argument('--other_cycles', type=int, nargs='*', default=[6, 8, 11])
    args = p.parse_args()

    from gpu4pyscf import dft
    from gpu4pyscf.dft import numint
    from gpu4pyscf.df import df_jk
    here = os.path.dirname(os.path.abspath(__file__))
    mol = gto.M(atom=os.path.join(here, '..', 'molecules', 'organic', args.mol + '.xyz'),
                basis=args.basis, verbose=0, max_memory=40000)
    mf = dft.RKS(mol, xc='wb97m-v').density_fit()
    mf.grids.atom_grid = (99, 590)
    mf.nlcgrids.atom_grid = (50, 194)
    mf.conv_tol = 1e-9

    dms = {}

    def cb(env):
        dms[env['cycle'] + 1] = (cupy.asarray(env['dm']).copy(),
                                 env['mo_coeff'].copy(), env['mo_occ'].copy())
    mf.callback = cb
    mf.kernel()
    print(f'# {args.mol} converged in {len(dms)} cycles')

    def tagged(c):
        dm, mo, occ = dms[c]
        return mf.make_rdm1(mo, occ)

    def kmat(dm, prec, omega=None):
        old = df_jk.K_PRECISION
        df_jk.K_PRECISION = prec
        try:
            return mf.with_df.get_jk(dm, hermi=1, with_j=False, omega=omega)[1]
        finally:
            df_jk.K_PRECISION = old

    def vxc(dm, prec):
        old = numint.XC_PRECISION
        numint.XC_PRECISION = prec
        try:
            n, e, v = mf._numint.nr_rks(mol, mf.grids, mf.xc, dm)
            return float(e), v
        finally:
            numint.XC_PRECISION = old

    D0 = tagged(args.ref_cycle)
    quantities = {
        'K  (full range) fp32': lambda d: kmat(d, 'fp32') - kmat(d, 'fp64'),
        'K  (LR omega=.3) fp32': lambda d: kmat(d, 'fp32', 0.3) - kmat(d, 'fp64', 0.3),
        'Vxc (MGGA grids) fp32': lambda d: vxc(d, 'fp32')[1] - vxc(d, 'fp64')[1],
    }
    e0 = {k: f(D0) for k, f in quantities.items()}
    print(f'{"quantity":24s} {"|dD|_F":>9s} {"max|e(D)|":>10s} {"max|e(D)-e(D0)|":>16s} {"ratio":>7s}   (ref cycle {args.ref_cycle})')
    for c in args.other_cycles:
        if c not in dms:
            continue
        D = tagged(c)
        dnorm = float(cupy.linalg.norm(cupy.asarray(D) - cupy.asarray(D0)))
        for k, f in quantities.items():
            e = f(D)
            a = float(cupy.abs(e).max())
            b = float(cupy.abs(e - e0[k]).max())
            print(f'{k:24s} {dnorm:9.2e} {a:10.2e} {b:16.2e} {b / a:7.2f}   cycle {c}')
    # same-density repeat: a lower bound on the non-systematic (non-reproducible) part
    print('# ratio << 1  => error is systematic, defect correction removes it')
    print('# ratio ~ 1   => error decorrelates with D (rounding noise), correction does not help')


if __name__ == '__main__':
    main()
