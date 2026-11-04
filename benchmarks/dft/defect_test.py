#!/usr/bin/env python
'''
Go/no-go experiment for a defect-corrected dual-grid SCF: is the grid defect  Delta(D) = V_fine(D) - V_coarse(D)  (and the
corresponding energy defect) smooth in the density, i.e.  |Delta(D2) - Delta(D1)| << |Delta(D1)|  for SCF iterates D1, D2?
Prints, for pairs of iterates, the relative change of the defect for the XC matrix, the NLC matrix and their energies.
usage: defect_test.py [--mol 033_Bisphenol_A] [--basis def2-tzvp]
'''
import argparse, copy, os
import cupy
from pyscf import gto


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mol', default='033_Bisphenol_A')
    p.add_argument('--basis', default='def2-tzvp')
    p.add_argument('--coarse', type=int, nargs=2, default=(50, 194))
    p.add_argument('--coarse_nlc', type=int, nargs=2, default=(30, 110))
    args = p.parse_args()
    from gpu4pyscf import dft
    from gpu4pyscf.dft import numint
    from gpu4pyscf.df import df_jk
    here = os.path.dirname(os.path.abspath(__file__))
    mol = gto.M(atom=os.path.join(here, '..', 'molecules', 'organic', args.mol + '.xyz'), basis=args.basis, verbose=0,
                max_memory=14000)
    mf = dft.RKS(mol, xc='wb97m-v').density_fit()
    mf.grids.atom_grid = (99, 590)
    mf.nlcgrids.atom_grid = (50, 194)
    numint.XC_PRECISION = df_jk.K_PRECISION = 'crt'
    numint.VV10_PRECISION = 'fp32'
    mf.conv_tol = 1e-9
    dms = {}
    def cb(env):
        dms[env['cycle'] + 1] = (cupy.asarray(env['dm']).copy() if 'dm' in env else None)
    mf.callback = cb
    mf.max_cycle = 12
    mf.kernel()
    numint.XC_PRECISION = df_jk.K_PRECISION = 'fp64'
    numint.VV10_PRECISION = 'fp64'
    ni = mf._numint
    gf, gc = mf.grids, copy.copy(mf.grids)
    gc.reset(); gc.atom_grid = tuple(args.coarse)
    nf, nc = mf.nlcgrids, copy.copy(mf.nlcgrids)
    nc.reset(); nc.atom_grid = tuple(args.coarse_nlc)
    for g in (gf, gc, nf, nc):
        if g.coords is None:
            g.build(with_non0tab=True)

    def defect(dm):
        _, ef, vf = ni.nr_rks(mol, gf, mf.xc, dm)
        _, ec, vc = ni.nr_rks(mol, gc, mf.xc, dm)
        _, ef2, vf2 = ni.nr_nlc_vxc(mol, nf, mf.xc, dm)
        _, ec2, vc2 = ni.nr_nlc_vxc(mol, nc, mf.xc, dm)
        return (vf - vc, ef - ec, vf2 - vc2, ef2 - ec2)

    cyc = sorted(k for k, v in dms.items() if v is not None)
    print('# cycles with densities:', cyc)
    pairs = [(3, 5), (5, 7), (7, 9), (3, 9)]
    cache = {}
    for a, b in pairs:
        for k in (a, b):
            if k not in cache:
                cache[k] = defect(dms[k])
        da, db = cache[a], cache[b]
        n1, n2 = float(cupy.linalg.norm(da[0])), float(cupy.linalg.norm(da[2]))
        r1 = float(cupy.linalg.norm(db[0] - da[0])) / n1
        r2 = float(cupy.linalg.norm(db[2] - da[2])) / n2
        print(f'pair ({a},{b}): XC matrix defect |D|={n1:.2e} rel change {r1:.3f}; E_xc defect {da[1]:+.3e} -> {db[1]:+.3e}; '
              f'NLC matrix defect |D|={n2:.2e} rel change {r2:.3f}; E_nlc defect {da[3]:+.3e} -> {db[3]:+.3e}', flush=True)
    vref = float(cupy.linalg.norm(ni.nr_rks(mol, gf, mf.xc, dms[9])[2]))
    print(f'# |V_xc| = {vref:.3e}  (defect norm relative: {float(cupy.linalg.norm(cache[9][0])) / vref:.2e})')


if __name__ == '__main__':
    main()
