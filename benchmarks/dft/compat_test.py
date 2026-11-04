#!/usr/bin/env python
'''
Compatibility / accuracy matrix of the 'crt' mixed-precision scheme against the plain FP64 SCF (same grids, same DF basis):
functionals (LDA/GGA/MGGA, hybrids, range-separated, +VV10), restricted / unrestricted, ECPs, charged systems, diffuse and
large basis sets, Cartesian AOs.  Prints one line per case: energies, difference, wall times, gradient difference (if --grad).
usage: compat_test.py [--cases name,name,...] [--grad] [--list]
'''
import argparse
import time
import traceback
import cupy
from pyscf import gto

WATER = 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692'
CH3I = 'C 0 0 0; I 0 0 2.14; H 1.03 0 -0.36; H -0.515 0.892 -0.36; H -0.515 -0.892 -0.36'
AGCL = 'Ag 0 0 0; Cl 0 0 2.28'
PTCL = 'Pt 0 0 0; Cl 2.3 0 0; Cl -2.3 0 0; Cl 0 2.3 0; Cl 0 -2.3 0'
ACETATE = 'C 0 0 0; C 1.52 0 0; O 2.15 1.08 0; O 2.15 -1.08 0; H -0.38 1.02 0; H -0.38 -0.51 0.88; H -0.38 -0.51 -0.88'
CH3 = 'C 0 0 0; H 1.08 0 0; H -0.54 0.935 0; H -0.54 -0.935 0'
OH = 'O 0 0 0; H 0 0 0.97'
NO2 = 'N 0 0 0; O 1.1 0.55 0; O -1.1 0.55 0'
ACOH2 = ('C 0.0 0.0 0.0; C 1.5 0.0 0.0; O -0.6 1.05 0.0; O -0.6 -1.05 0.0; H 1.9 1.0 0.0; H 1.9 -0.5 0.88; H 1.9 -0.5 -0.88;'
         'H -1.55 1.0 0.0')

# name: (atom, basis, xc, charge, spin, cart, grid)
CASES = {
    'wb97mv_svp':        (WATER, 'def2-svp', 'wb97m-v', 0, 0, False, (50, 194)),
    'pbe_tzvp':          (ACOH2, 'def2-tzvp', 'pbe', 0, 0, False, (50, 194)),
    'lda_tzvp':          (ACOH2, 'def2-tzvp', 'lda,vwn', 0, 0, False, (50, 194)),
    'b3lyp_tzvp':        (ACOH2, 'def2-tzvp', 'b3lyp', 0, 0, False, (50, 194)),
    'r2scan_tzvp':       (ACOH2, 'def2-tzvp', 'r2scan', 0, 0, False, (50, 194)),
    'tpssh_tzvp':        (ACOH2, 'def2-tzvp', 'tpssh', 0, 0, False, (50, 194)),
    'wb97xv_tzvp':       (ACOH2, 'def2-tzvp', 'wb97x-v', 0, 0, False, (50, 194)),
    'b97mv_tzvp':        (ACOH2, 'def2-tzvp', 'b97m-v', 0, 0, False, (50, 194)),
    'wb97mv_cart':       (ACOH2, 'def2-tzvp', 'wb97m-v', 0, 0, True, (50, 194)),
    'uks_wb97mv_ch3':    (CH3, 'def2-tzvp', 'wb97m-v', 0, 1, False, (50, 194)),
    'uks_pbe0_no2':      (NO2, 'def2-tzvp', 'pbe0', 0, 1, False, (50, 194)),
    'ecp_ch3i':          (CH3I, 'def2-tzvp', 'wb97m-v', 0, 0, False, (50, 194)),
    'ecp_agcl':          (AGCL, 'def2-tzvp', 'wb97m-v', 0, 0, False, (50, 194)),
    'ecp_ptcl4':         (PTCL, 'def2-tzvp', 'pbe0', -2, 0, False, (50, 194)),
    'anion_tzvppd':      (ACETATE, 'def2-tzvppd', 'wb97m-v', -1, 0, False, (50, 194)),
    'water_qzvpp':       (WATER, 'def2-qzvpp', 'wb97m-v', 0, 0, False, (50, 194)),
    'water_qzvppd':      (WATER, 'def2-qzvppd', 'wb97m-v', 0, 0, False, (50, 194)),
    'acoh_qzvppd':       (ACOH2, 'def2-qzvppd', 'wb97m-v', 0, 0, False, (50, 194)),
}


def run_case(name, grad, nolog=False):
    atom, basis, xc, charge, spin, cart, grid = CASES[name]
    from gpu4pyscf import dft
    mol = gto.M(atom=atom, basis=basis, charge=charge, spin=spin, cart=cart, verbose=0, max_memory=14000)
    res = {}
    for label, mp in (('fp64', False), ('crt', {'scheme': 'crt'})):
        mf = (dft.UKS(mol, xc=xc) if spin else dft.RKS(mol, xc=xc)).density_fit()
        mf.grids.atom_grid = grid
        if mf._numint.libxc.is_nlc(mf.xc):
            mf.nlcgrids.atom_grid = (30, 110)
        mf.conv_tol = 1e-9
        mf.max_cycle = 60
        if mp:
            mf.mixed_precision = mp
        cupy.cuda.Stream.null.synchronize()
        t = time.perf_counter()
        e = mf.kernel()
        cupy.cuda.Stream.null.synchronize()
        ts = time.perf_counter() - t
        g = None
        tg = 0
        if grad:
            t = time.perf_counter()
            try:
                g = mf.nuc_grad_method().kernel()
            except Exception as ex:
                g = f'grad failed: {type(ex).__name__}: {str(ex)[:80]}'
            tg = time.perf_counter() - t
        res[label] = (e, mf.converged, ts, g, tg, mol.nao)
        del mf
    e0, c0, t0, g0, tg0, nao = res['fp64']
    e1, c1, t1, g1, tg1, _ = res['crt']
    dg = ''
    if grad and not isinstance(g0, str) and not isinstance(g1, str) and g0 is not None:
        import numpy as np
        dg = f' max|dgrad|={float(np.abs(g1 - g0).max()):.1e}'
    elif grad:
        dg = f' grad: {g0 if isinstance(g0, str) else ""} {g1 if isinstance(g1, str) else ""}'
    print(f'{name:18s} nao={nao:4d} E_fp64={e0:.9f} E_crt={e1:.9f} dE={e1 - e0:+.1e} conv={c0}/{c1} '
          f't_fp64={t0:6.1f}s t_crt={t1:6.1f}s{dg}', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cases', default='')
    p.add_argument('--grad', action='store_true')
    p.add_argument('--list', action='store_true')
    args = p.parse_args()
    if args.list:
        print(' '.join(CASES))
        return
    names = args.cases.split(',') if args.cases else list(CASES)
    # JIT warm-up
    from gpu4pyscf import dft
    from gpu4pyscf.dft import numint
    for name in names:
        try:
            run_case(name, args.grad)
        except Exception as ex:
            print(f'{name:18s} FAILED: {type(ex).__name__}: {str(ex)[:160]}', flush=True)
            traceback.print_exc(limit=3)
        cupy.get_default_memory_pool().free_all_blocks()


if __name__ == '__main__':
    main()
