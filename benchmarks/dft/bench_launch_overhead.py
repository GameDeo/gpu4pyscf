#!/usr/bin/env python
'''
Launch-overhead benchmark of the XC grid loops: original chains of small ops (numint.XC_FUSED = False)
versus the fused kernels / merged GEMMs (XC_FUSED = True), on one fixed density.

Reports, for nr_rks (FP64 and FP32 contractions), nr_nlc_vxc and the XC gradient (grad.rks.get_exc):
  * wall time (best of --repeat, stream synchronised)
  * Python function calls (cProfile) and CUDA kernel launches / GPU busy time (torch profiler, if available)
  * max |dV| / |V| between the two variants

usage: bench_launch_overhead.py [--mol 033_Bisphenol_A] [--basis def2-svp] [--xc wb97m-v] [--repeat 3]
'''
import argparse
import cProfile
import json
import os
import pstats
import time
import numpy as np
import cupy
from pyscf import gto


def sync():
    cupy.cuda.Stream.null.synchronize()


def timed(fn, repeat):
    fn()   # warm up (JIT, plan caches, AO caches)
    sync()
    best = 1e30
    for _ in range(repeat):
        t0 = time.perf_counter()
        res = fn()
        sync()
        best = min(best, time.perf_counter() - t0)
    return best, res


def count_calls(fn):
    sync()
    pr = cProfile.Profile()
    pr.enable()
    fn()
    sync()
    pr.disable()
    st = pstats.Stats(pr)
    return st.total_calls


def kernel_stats(fn):
    '''(number of CUDA kernels, GPU busy time in s) from the torch profiler; (None, None) if unavailable'''
    try:
        from torch.profiler import profile, ProfilerActivity
    except Exception:
        return None, None
    sync()
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
        fn()
        sync()
    n = 0
    t = 0.
    for e in prof.key_averages():
        dt = getattr(e, 'device_time_total', None)
        if dt is None:
            dt = getattr(e, 'cuda_time_total', 0)
        if dt and e.device_type is not None and 'CUDA' in str(e.device_type):
            n += e.count
            t += dt / 1e6
    return n, t


def rel(a, b):
    a = cupy.asarray(a)
    b = cupy.asarray(b)
    return float(cupy.abs(a - b).max()) / max(float(cupy.abs(b).max()), 1e-300)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mol', default='033_Bisphenol_A')
    p.add_argument('--basis', default='def2-svp')
    p.add_argument('--xc', default='wb97m-v')
    p.add_argument('--repeat', type=int, default=3)
    p.add_argument('--cycles', type=int, default=3)
    p.add_argument('--blksize', type=int, default=0, help='unused (grid block size is a build-time config)')
    p.add_argument('--no-grad', action='store_true')
    p.add_argument('--out', default=None)
    args = p.parse_args()

    from gpu4pyscf import dft
    from gpu4pyscf.dft import numint
    from gpu4pyscf.grad import rks as rks_grad
    here = os.path.dirname(os.path.abspath(__file__))
    mol = gto.M(atom=os.path.join(here, '..', 'molecules', 'organic', args.mol + '.xyz'),
                basis=args.basis, verbose=0, max_memory=14000)
    mf = dft.RKS(mol, xc=args.xc).density_fit()
    mf.grids.atom_grid = (99, 590)
    mf.nlcgrids.atom_grid = (50, 194)
    mf.max_cycle = args.cycles
    mf.conv_tol = 1e-3
    mf.kernel()
    dm = mf.make_rdm1()
    mf.grids.build()
    ni = mf._numint
    xc = mf.xc
    xc_base = xc.split('__')[0]
    print(f'# {args.mol} {args.basis} {xc}: nao={mol.nao} ngrids={mf.grids.coords.shape[0]}  '
          f'GPU={cupy.cuda.runtime.getDeviceProperties(0)["name"].decode()}')

    results = {}

    def run(label, fn, check=None):
        row = {}
        for fused in (False, True):
            numint.XC_FUSED = fused
            try:
                wall, res = timed(fn, args.repeat)
                ncalls = count_calls(fn)
                nk, busy = kernel_stats(fn)
            finally:
                numint.XC_FUSED = True
            row['fused' if fused else 'legacy'] = dict(wall=wall, py_calls=ncalls, kernels=nk, gpu_busy=busy)
            row['fused' if fused else 'legacy']['res'] = res
        r0, r1 = row['legacy'], row['fused']
        err = check(r1.pop('res'), r0.pop('res')) if check else None
        row['rel_diff'] = err
        results[label] = row
        print(f'{label:28s} wall {r0["wall"]:.3f} -> {r1["wall"]:.3f} s ({r0["wall"]/r1["wall"]:.2f}x)  '
              f'py calls {r0["py_calls"]} -> {r1["py_calls"]}  '
              f'kernels {r0["kernels"]} -> {r1["kernels"]}  '
              f'GPU busy {r0["gpu_busy"]} -> {r1["gpu_busy"]}  rel diff {err}')

    chk = lambda a, b: rel(a[2], b[2])
    prec_old = numint.XC_PRECISION
    for prec in ('fp64', 'fp32'):
        numint.XC_PRECISION = prec
        run(f'nr_rks {prec}', lambda: ni.nr_rks(mol, mf.grids, xc, dm), chk)
    numint.XC_PRECISION = prec_old
    if mf.nlcgrids is not None and mf.do_nlc():
        mf.nlcgrids.build()
        run('nr_nlc_vxc fp64', lambda: ni.nr_nlc_vxc(mol, mf.nlcgrids, xc, dm), chk)
    if not args.no_grad:
        def exc_grad():
            return rks_grad.get_exc(ni, mol, mf.grids, xc, dm)
        run('grad.get_exc', exc_grad, lambda a, b: rel(a, b))
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(results, f, indent=1, default=str)


if __name__ == '__main__':
    main()
