#!/usr/bin/env python
'''
Benchmark of the implicit-solvent part of a Kohn-Sham SCF + nuclear gradient (PCM family / SMD), with per-component
synchronising timers, to calibrate the static model of solvent_cost_model.py and to measure the effect of
mf.mixed_precision (--scheme crt) on solvated calculations.

Default job: Bisphenol A in water, wB97M-V/def2-TZVP, DF (def2-universal-jkfit), IEF-PCM with 302 points per atom.

    python bench_solvent.py                                    # FP64 pipeline + PCM
    python bench_solvent.py --scheme crt --grad                # crt mixed precision + PCM, SCF + gradient
    python bench_solvent.py --scheme crt --coarse_solvent_lebedev 17 --grad
    python bench_solvent.py --solvent none --scheme crt        # same job in vacuum: the solvent overhead is the difference
    python bench_solvent.py --solvent smd --grad               # SMD (590 points per atom, dense matrices: watch the memory)
    python bench_solvent.py --method C-PCM --lowmem            # matrix-free (GMRES/MINRES) solves

Printed: SCF/gradient wall time, E, E_solv, E_cds, the stage table (solvent components first) and the number of
PCM._get_vind calls made by the gradient (0 after the solvent-response cache fix).  Compare --out json files with --ref.
'''
import os
import sys
import json
import time
import argparse
import numpy as np
import cupy
from pyscf import gto

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
import bench_wb97mv as B   # noqa: E402  (timers: B.timed, B.STAGES, B.install_timers)


def install_solvent_timers(solvent):
    from gpu4pyscf.solvent import pcm, smd
    from gpu4pyscf.solvent import _attach_solvent
    from gpu4pyscf.solvent.grad import pcm as gpcm
    B.timed('solvent: get_veff wrapper', _attach_solvent.SCFWithSolvent, 'get_veff')
    for cls, tag in ((pcm.PCM, 'PCM'), (smd.SMD, 'SMD')):
        B.timed(f'  kernel (v_grids+solve+vmat) [{tag}]', cls, 'kernel')
        B.timed(f'    _get_v  int1e_grids(dm)->v_g [{tag}]', cls, '_get_v')
        B.timed(f'    _get_vmat int1e_grids(q)->Fock [{tag}]', cls, '_get_vmat')
        B.timed(f'    left_solve_K (LU/GMRES) [{tag}]', cls, 'left_solve_K')
        B.timed(f'    left_multiply_R [{tag}]', cls, 'left_multiply_R')
        B.timed(f'  _get_vind (SCF + gradient re-solves) [{tag}]', cls, '_get_vind')
    B.timed('  PCM.build (surface, S/D, DA*S GEMM, LU, nuclear potential)', pcm.PCM, 'build')
    B.timed('  SMD.build', smd.SMD, 'build')
    B.timed('  get_D_S kernel', pcm, 'get_D_S')
    for name in ('grad_qv', 'grad_solver', 'grad_nuc'):
        B.timed(f'solvent grad: {name}', gpcm, name)
    B.timed('  int1e_grids_ip1', gpcm, 'int1e_grids_ip1')
    B.timed('  int1e_grids_ip2', gpcm, 'int1e_grids_ip2')
    B.timed('  get_dD_dS', gpcm, 'get_dD_dS')
    B.timed('  get_dF_dA', gpcm, 'get_dF_dA')
    B.timed('solvent grad: PCM.grad total', pcm.PCM, 'grad')
    B.timed('solvent grad: SMD.grad total', smd.SMD, 'grad')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mol', default='033_Bisphenol_A')
    p.add_argument('--basis', default='def2-tzvp')
    p.add_argument('--auxbasis', default='def2-universal-jkfit')
    p.add_argument('--xc', default='wb97m-v')
    p.add_argument('--grid', type=int, nargs=2, default=(99, 590))
    p.add_argument('--nlcgrid', type=int, nargs=2, default=(50, 194))
    p.add_argument('--charge', type=int, default=0)
    p.add_argument('--spin', type=int, default=0)
    p.add_argument('--solvent', default='pcm', choices=['pcm', 'smd', 'none'])
    p.add_argument('--solvent_name', default='water')
    p.add_argument('--method', default='IEF-PCM', choices=['C-PCM', 'COSMO', 'IEF-PCM', 'SS(V)PE'])
    p.add_argument('--eps', type=float, default=None)
    p.add_argument('--lebedev', type=int, default=29, help='PCM surface: Lebedev order (29 -> 302 points per atom)')
    p.add_argument('--lowmem', action='store_true', help='PCM lowmem_intermediate_storage: matrix-free iterative solves')
    p.add_argument('--scheme', default=None, choices=['crt', 'classic', 'auto'], help='mf.mixed_precision scheme; default: FP64')
    p.add_argument('--coarse_solvent_lebedev', type=int, default=None,
                   help='opt-in: Lebedev order of the solvent surface in the coarse stage (e.g. 17)')
    p.add_argument('--conv_tol', type=float, default=1e-9)
    p.add_argument('--max_cycle', type=int, default=50)
    p.add_argument('--grad', action='store_true')
    p.add_argument('--warmup', action='store_true', help='run a tiny solvated SCF first (JIT, library loading)')
    p.add_argument('--verbose', type=int, default=4)
    p.add_argument('--max_memory', type=int, default=40000)
    p.add_argument('--out', default=None)
    p.add_argument('--ref', default=None)
    p.add_argument('--tol', type=float, default=1e-7)
    args = p.parse_args()

    from gpu4pyscf import dft
    from gpu4pyscf.solvent import pcm as gpcm_mod

    def make(mol, xc, auxbasis, grid, nlcgrid):
        mf = dft.UKS(mol, xc=xc) if mol.spin != 0 else dft.RKS(mol, xc=xc)
        mf = mf.density_fit(auxbasis=auxbasis)
        mf.grids.atom_grid = tuple(grid)
        if mf.do_nlc():
            mf.nlcgrids.atom_grid = tuple(nlcgrid)
        return mf

    def add_solvent(mf):
        if args.solvent == 'pcm':
            mf = mf.PCM()
            sol = mf.with_solvent
            sol.method = args.method
            sol.lebedev_order = args.lebedev
            if args.solvent_name:
                sol.solvent = args.solvent_name
            if args.eps is not None:
                sol.eps = args.eps
            sol.lowmem_intermediate_storage = args.lowmem
        elif args.solvent == 'smd':
            mf = mf.SMD(solvent=args.solvent_name)
        return mf

    if args.warmup:
        w = gto.M(atom='O 0 0 0; H 0 0.75 0.6; H 0 -0.75 0.6', basis='def2-svp', verbose=0)
        wm = add_solvent(make(w, args.xc, args.auxbasis, (20, 50), (10, 26)))
        wm.kernel()
        if args.grad:
            wm.Gradients().kernel()

    B.STAGES.clear()
    B.install_timers()
    if args.solvent != 'none':
        install_solvent_timers(args.solvent)

    path = args.mol if os.path.exists(args.mol) else os.path.join(here, '..', 'molecules', 'organic', args.mol + '.xyz')
    mol = gto.M(atom=path, basis=args.basis, charge=args.charge, spin=args.spin, verbose=args.verbose,
                max_memory=args.max_memory)
    mf = add_solvent(make(mol, args.xc, args.auxbasis, args.grid, args.nlcgrid))
    mf.conv_tol = args.conv_tol
    mf.max_cycle = args.max_cycle
    if args.scheme:
        mf.mixed_precision = {'scheme': args.scheme}
        if args.coarse_solvent_lebedev:
            mf.mixed_precision['coarse_solvent_lebedev'] = args.coarse_solvent_lebedev
    print(f'# {args.mol} natm={mol.natm} nao={mol.nao} basis={args.basis} xc={args.xc} solvent={args.solvent}/{args.method} '
          f'lebedev={args.lebedev} lowmem={args.lowmem} scheme={args.scheme} coarse_solvent_lebedev={args.coarse_solvent_lebedev}')
    print(f'# GPU: {cupy.cuda.runtime.getDeviceProperties(0)["name"].decode()}')

    sync = B._sync
    sync()
    t0 = time.perf_counter()
    e = mf.kernel()
    sync()
    t_scf = time.perf_counter() - t0
    res = {'energy': float(e), 'converged': bool(mf.converged), 'scf_time': t_scf, 'nao': int(mol.nao),
           'cycles': int(getattr(mf, 'cycles', -1)), 'cycles_coarse': int(getattr(mf, 'cycles_coarse', 0))}
    sol = getattr(mf, 'with_solvent', None)
    if sol is not None:
        res['nsurf'] = int(sol.surface['grid_coords'].shape[0]) if sol.surface else 0
        res['e_solvent'] = float(mf.scf_summary.get('e_solvent', 0.0))
        res['e_cds'] = float(mf.scf_summary.get('e_cds', 0.0))
        mem = 0
        for v in (sol._intermediates or {}).values():
            if isinstance(v, cupy.ndarray):
                mem += v.nbytes
        res['solvent_intermediates_GB'] = mem / 1e9
    if args.grad:
        n_vind = {'n': 0}
        if sol is not None:
            from gpu4pyscf.solvent import pcm, smd
            for cls in (pcm.PCM, smd.SMD):
                orig = cls._get_vind

                def counted(self, *a, _orig=orig, **k):
                    n_vind['n'] += 1
                    return _orig(self, *a, **k)
                cls._get_vind = counted
        sync()
        t0 = time.perf_counter()
        g = mf.Gradients().kernel()
        sync()
        res['grad_time'] = time.perf_counter() - t0
        res['grad'] = np.asarray(g).tolist()
        res['grad_get_vind_calls'] = n_vind['n']

    print(f'E = {e:.10f} converged={mf.converged} cycles(final stage)={res["cycles"]} coarse={res["cycles_coarse"]}')
    if sol is not None:
        print(f'E_solvent = {res["e_solvent"]:.8f}  E_cds = {res["e_cds"]:.8f}  nsurf = {res["nsurf"]}  '
              f'solvent intermediates on GPU = {res["solvent_intermediates_GB"]:.2f} GB')
    print(f'SCF wall time: {t_scf:.2f} s')
    print(f'{"stage":62s} {"total/s":>9s} {"calls":>6s} {"per-call/s":>11s}')
    for k, (sec, n) in B.STAGES.items():
        if n or not k.startswith(' '):
            print(f'{k:62s} {sec:9.3f} {n:6d} {sec / max(n, 1):11.3f}')
    if args.grad:
        print(f'gradient time: {res["grad_time"]:.2f} s   PCM._get_vind calls during the gradient: {res["grad_get_vind_calls"]}')
        print(f'|grad|_max = {np.abs(g).max():.6e}')
    res['stages'] = {k: {'sec': v[0], 'calls': v[1]} for k, v in B.STAGES.items()}

    status = 0
    if args.ref:
        ref = json.load(open(args.ref))
        dE = res['energy'] - ref['energy']
        print(f'dE vs ref = {dE:.3e} Eh  (tol {args.tol:.1e})')
        if abs(dE) > args.tol:
            print('REGRESSION CHECK FAILED (energy)')
            status = 1
        if args.grad and 'grad' in ref:
            dg = np.abs(np.asarray(res['grad']) - np.asarray(ref['grad'])).max()
            print(f'max |dgrad| vs ref = {dg:.3e}')
            if dg > 10 * args.tol:
                print('REGRESSION CHECK FAILED (gradient)')
                status = 1
    if args.out:
        json.dump(res, open(args.out, 'w'), indent=1)
    sys.exit(status)


if __name__ == '__main__':
    main()
