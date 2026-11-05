#!/usr/bin/env python
'''
Stage-resolved benchmark / regression driver for RKS + density fitting + (VV10) NLC.

Example:
    python bench_wb97mv.py --mol 042_Penicillin_V --basis def2-tzvp --xc wb97m-v

Every GPU-side stage is wrapped with a synchronising timer, so the printed table
shows where the wall-clock time goes (grid build, DF tensor build, J, K, XC, VV10 ...).
The final energy is printed with 10 digits so that it can be compared to a reference
(see --ref/--tol) when checking that an optimisation does not change the answer.
'''
import os
import sys
import json
import time
import argparse
import collections
import functools
import numpy as np
import cupy
import pyscf
from pyscf import gto

STAGES = collections.OrderedDict()


def _sync():
    cupy.cuda.get_current_stream().synchronize()


def timed(name, obj, attr):
    '''Replace obj.attr by a version that accumulates synchronised wall time under `name`.'''
    f = getattr(obj, attr)
    STAGES.setdefault(name, [0.0, 0])  # register now so a timer that never fires is reported

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        _sync()
        t0 = time.perf_counter()
        try:
            return f(*args, **kwargs)
        finally:
            _sync()
            dt = time.perf_counter() - t0
            rec = STAGES.setdefault(name, [0.0, 0])
            rec[0] += dt
            rec[1] += 1
    setattr(obj, attr, wrapper)


def install_timers():
    from gpu4pyscf.dft import numint, gen_grid, rks
    from gpu4pyscf.df import df, df_jk
    timed('grids.build', gen_grid.Grids, 'build')
    # get_veff calls these through the NumInt instance, so patch the class attributes
    timed('XC main (nr_rks)', numint.NumInt, 'nr_rks')
    timed('NLC total (nr_nlc_vxc)', numint.NumInt, 'nr_nlc_vxc')
    timed('  VV10 kernels (_vv10nlc)', numint, '_vv10nlc')
    timed('  DF.build (cderi)', df.DF, 'build')  # nested inside get_jk
    timed('df_jk.get_jk', df_jk, 'get_jk')


def install_fine_timers():
    '''Finer-grained (synchronising, so slightly pessimistic) timers inside the XC pipeline.'''
    from gpu4pyscf.dft import numint
    timed('  . eval_ao', numint, 'eval_ao')
    timed('  . eval_rho (_eval_rho2/eval_rho)', numint, '_eval_rho2')
    timed('  . eval_rho dm path', numint, 'eval_rho')
    timed('  . eval_xc_eff (libxc)', numint.NumInt, 'eval_xc_eff')
    timed('  . _scale_ao', numint, '_scale_ao')
    timed('  . _tau_dot', numint, '_tau_dot')
    timed('  . add_sparse', numint, 'add_sparse')
    timed('  . contract (cutensor)', numint, 'contract')


def build_mol(args):
    path = args.mol
    if not os.path.exists(path):
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, '..', 'molecules', 'organic', args.mol + '.xyz')
    mol = gto.M(atom=path, basis=args.basis, verbose=args.verbose, max_memory=args.max_memory)
    return mol


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mol', default='042_Penicillin_V')
    p.add_argument('--basis', default='def2-tzvp')
    p.add_argument('--auxbasis', default='def2-universal-jkfit')
    p.add_argument('--xc', default='wb97m-v')
    p.add_argument('--grid', type=int, nargs=2, default=(99, 590))
    p.add_argument('--nlcgrid', type=int, nargs=2, default=(50, 194))
    p.add_argument('--max_cycle', type=int, default=50)
    p.add_argument('--conv_tol', type=float, default=1e-9)
    p.add_argument('--verbose', type=int, default=4)
    p.add_argument('--max_memory', type=int, default=40000)
    p.add_argument('--vv10', default='fp64', choices=['fp64', 'fp32'], help='precision of the VV10 pair kernel')
    p.add_argument('--kprec', default='fp64', choices=['fp64', 'fp32', 'crt'], help='precision of the DF-K GEMMs (all iterations)')
    p.add_argument('--mixed', default=None, choices=['fp32', 'tf32x3', 'auto'], help='staged mixed-precision SCF (K precision of stage 1)')
    p.add_argument('--xcprec', default='fp64', choices=['fp64', 'fp32', 'tf32x3', 'crt'], help='precision of XC grid GEMMs (all iterations)')
    p.add_argument('--crt_n', type=int, default=11, help='number of INT8 moduli for xcprec/kprec crt')
    p.add_argument('--final_grad_factor', type=float, default=None, help='stage-2 gradient-threshold factor of the staged scheme')
    p.add_argument('--scheme', default=None, choices=['auto', 'crt', 'classic'], help='mixed-precision scheme (see scf/mixed_precision.py)')
    p.add_argument('--no1b', action='store_true', help='disable stage 1b (FP64 K + reduced XC polish)')
    p.add_argument('--grad', action='store_true')
    p.add_argument('--profile', action='store_true', help='add fine-grained (synchronising) timers inside the XC pipeline')
    p.add_argument('--warmup', action='store_true', help='run a tiny SCF first to exclude JIT costs')
    p.add_argument('--out', default=None, help='write a json summary here')
    p.add_argument('--ref', default=None, help='json summary of a reference run to compare against')
    p.add_argument('--tol', type=float, default=1e-7, help='|E - E_ref| tolerance in Eh')
    p.add_argument('--set', nargs='*', default=[], help='extra mf attributes, e.g. --set nlcgrids.level=1')
    p.add_argument('--multifidelity', default=None,
                   help="multi-fidelity SCF (scf/multifidelity.py): '{}' or a dict literal of options")
    args = p.parse_args()

    import gpu4pyscf
    from gpu4pyscf import dft
    if args.warmup:
        w = gto.M(atom='O 0 0 0; H 0 0.75 0.6; H 0 -0.75 0.6', basis='def2-svp', verbose=0)
        wm = dft.RKS(w, xc=args.xc).density_fit()
        wm.grids.atom_grid = (20, 50)
        wm.nlcgrids.atom_grid = (10, 26)
        if args.multifidelity is not None:      # JIT of the multi-fidelity kernels
            wm.multifidelity = eval(args.multifidelity) or True
        wm.kernel()
        wm.nuc_grad_method().kernel() if args.grad else None
    from gpu4pyscf.dft import numint as _numint
    from gpu4pyscf.df import df_jk as _df_jk
    _numint.VV10_PRECISION = args.vv10
    _df_jk.K_PRECISION = args.kprec
    _numint.XC_PRECISION = args.xcprec
    _numint.XC_CRT_MODULI = args.crt_n
    _df_jk.K_CRT_MODULI = args.crt_n
    install_timers()
    if args.profile:
        install_fine_timers()

    mol = build_mol(args)
    mf = dft.RKS(mol, xc=args.xc).density_fit(auxbasis=args.auxbasis)
    mf.grids.atom_grid = tuple(args.grid)
    mf.nlcgrids.atom_grid = tuple(args.nlcgrid)
    mf.conv_tol = args.conv_tol
    if args.mixed:
        if args.mixed == 'auto':   # K/XC precision chosen by GPU generation (FP64 on sm_80+)
            mf.mixed_precision = {'vv10_precision': 'fp32'}
        else:
            mf.mixed_precision = {'k_precision': args.mixed, 'xc_precision': args.mixed}
        if args.final_grad_factor is not None:
            mf.mixed_precision['final_grad_factor'] = args.final_grad_factor
        if args.no1b:
            mf.mixed_precision['stage1b'] = False
        if args.scheme:
            mf.mixed_precision['scheme'] = args.scheme
    mf.max_cycle = args.max_cycle
    if args.multifidelity is not None:
        mf.multifidelity = eval(args.multifidelity) or True
    for s in args.set:
        k, v = s.split('=')
        obj = mf
        parts = k.split('.')
        for q in parts[:-1]:
            obj = getattr(obj, q)
        setattr(obj, parts[-1], eval(v))

    print(f'# {args.mol} natm={mol.natm} nao={mol.nao} basis={args.basis} aux={args.auxbasis} '
          f'xc={args.xc} grid={args.grid} nlcgrid={args.nlcgrid}')
    print(f'# GPU: {cupy.cuda.runtime.getDeviceProperties(0)["name"].decode()}')
    _sync()
    t0 = time.perf_counter()
    e = mf.kernel()
    _sync()
    t_scf = time.perf_counter() - t0
    res = {'energy': float(e), 'converged': bool(mf.converged), 'scf_time': t_scf,
           'ngrids': int(mf.grids.size), 'nlc_ngrids': int(mf.nlcgrids.size) if mf.do_nlc() else 0,
           'nao': int(mol.nao), 'stages': {k: {'sec': v[0], 'calls': v[1]} for k, v in STAGES.items()}}
    res['cycles'] = int(getattr(mf, 'cycles', -1))
    res['cycles_coarse'] = int(getattr(mf, 'cycles_coarse', 0))
    res['cycles_stage1'] = int(getattr(mf, 'cycles_stage1', 0))
    res['cycles_stage1b'] = int(getattr(mf, 'cycles_stage1b', 0))
    try:
        dfobj = mf.with_df
        nbytes = 0
        for obj in [dfobj] + list(getattr(dfobj, '_rsh_df', {}).values()):
            c = getattr(obj, '_cderi', None)
            if c is not None:
                nbytes += sum(int(x.nbytes) for x in (c.values() if isinstance(c, dict) else c))
        res['cderi_GB'] = nbytes / 1e9
        print(f'# DF tensors held: {nbytes / 1e9:.2f} GB; GPU mem pool used: '
              f'{cupy.get_default_memory_pool().total_bytes() / 1e9:.2f} GB')
    except Exception as ex:  # reporting only
        print('# (could not size DF tensors:', ex, ')')
    if args.grad:
        _sync()
        t0 = time.perf_counter()
        g = mf.nuc_grad_method().kernel()
        _sync()
        res['grad_time'] = time.perf_counter() - t0
        res['grad'] = np.asarray(g).tolist()

    print(f'E = {e:.10f}  converged={mf.converged}  cycles(stage2/FP64)={res["cycles"]} coarse={res["cycles_coarse"]} stage1={res["cycles_stage1"]} stage1b={res["cycles_stage1b"]}  '
          f'ngrids={res["ngrids"]} nlc_ngrids={res["nlc_ngrids"]}')
    print(f'SCF wall time: {t_scf:.2f} s   ({t_scf / max(res["cycles"], 1):.2f} s/cycle incl. setup)')
    print(f'{"stage":32s} {"total/s":>9s} {"calls":>6s} {"per-call/s":>11s}')
    for k, (sec, n) in STAGES.items():
        print(f'{k:32s} {sec:9.3f} {n:6d} {sec / max(n, 1):11.3f}')
    # nested timers (indented names) are not part of the top-level sum
    top = sum(v[0] for k, v in STAGES.items() if not k.startswith(' '))
    print(f'{"unaccounted (diag, DIIS, misc)":32s} {t_scf - top:9.3f}')
    res['unaccounted'] = t_scf - top
    dead = [k for k, v in STAGES.items() if v[1] == 0]
    if dead:
        print('WARNING: timers that never fired:', dead)
    if args.grad:
        print(f'gradient time: {res["grad_time"]:.2f} s')
    st = getattr(mf, 'multifidelity_stats', None)
    if st:
        print('multi-fidelity cycles', st['cycles'], 'conv_tol_grad %.1e' % st['conv_tol_grad'])
        for g_, e_ in st['outer_hist']:
            print(f'  outer |g| = {g_:.3e}  E = {e_:.12f}')
        for k_, (sec_, n_) in sorted(st['timers'].items(), key=lambda kv: -kv[1][0]):
            print(f'  {k_:20s} {sec_:8.3f} s {n_:5d} calls {sec_ / max(n_, 1):8.4f} s/call')
        res['multifidelity'] = {'cycles': st['cycles'], 'outer_hist': st['outer_hist'],
                                'timers': {k_: list(v_) for k_, v_ in st['timers'].items()}, 'wall': st['wall']}

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
