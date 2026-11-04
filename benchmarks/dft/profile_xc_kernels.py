#!/usr/bin/env python
'''
Kernel-level breakdown of ONE nr_rks (XC) evaluation (and optionally one NLC + one K build).

Purpose: decide which share of the XC time is GEMM (the only part INT8/TF32 emulation can speed up),
AO evaluation (gint), libxc, and elementwise glue.  Uses the torch/CUPTI profiler, which sees every
CUDA kernel regardless of whether it was launched by cupy, cuBLAS, cuTENSOR or libgdft.

usage: profile_xc_kernels.py [--mol 033_Bisphenol_A] [--xcprec fp64|fp32] [--cycles 4]
'''
import argparse
import os
import re
import time
import numpy as np
import cupy
from pyscf import gto


def classify(name):
    n = name.lower()
    if re.search(r'gemm|cutlass|sgemm|dgemm|cublas|gemv|syrk|ampere|turing_|volta_|sm75_xmma|xmma', n):
        return 'GEMM (cuBLAS/cutlass)'
    if 'cutensor' in n or 'contraction' in n or 'tensor' in n and 'elementwise' not in n:
        return 'GEMM (cuTENSOR)'
    if re.search(r'gto|eval_ao|gdft|ao_|_ao|sph|cart|contract_rho|vxc|nr_rks|add_sparse|take|scatter', n):
        return 'AO / gdft kernels'
    if re.search(r'xc|lda|gga|mgga|libxc|work_', n):
        return 'libxc'
    if re.search(r'elementwise|cupy_|reduce|copy|transpose|fill|cub|multiply|add|sum|astype|cast', n):
        return 'elementwise / glue'
    return 'other'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mol', default='033_Bisphenol_A')
    p.add_argument('--basis', default='def2-tzvp')
    p.add_argument('--xcprec', default='fp64')
    p.add_argument('--cycles', type=int, default=4, help='SCF cycles used to obtain a realistic density')
    p.add_argument('--top', type=int, default=25)
    args = p.parse_args()

    import torch
    from torch.profiler import profile, ProfilerActivity
    from gpu4pyscf import dft
    from gpu4pyscf.dft import numint
    here = os.path.dirname(os.path.abspath(__file__))
    mol = gto.M(atom=os.path.join(here, '..', 'molecules', 'organic', args.mol + '.xyz'),
                basis=args.basis, verbose=0, max_memory=14000)
    mf = dft.RKS(mol, xc='wb97m-v').density_fit()
    mf.grids.atom_grid = (99, 590)
    mf.nlcgrids.atom_grid = (50, 194)
    mf.max_cycle = args.cycles
    mf.conv_tol = 1e-3
    mf.kernel()
    dm = mf.make_rdm1()
    mf.grids.build()
    numint.XC_PRECISION = args.xcprec

    ni = mf._numint
    ni.nr_rks(mol, mf.grids, mf.xc, dm)  # warm up (AO/grid caches, JIT)
    cupy.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    ni.nr_rks(mol, mf.grids, mf.xc, dm)
    cupy.cuda.Stream.null.synchronize()
    wall = time.perf_counter() - t0
    print(f'# nr_rks wall (xc_precision={args.xcprec}): {wall:.3f} s   ngrids={mf.grids.coords.shape[0]}')

    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
        ni.nr_rks(mol, mf.grids, mf.xc, dm)
        cupy.cuda.Stream.null.synchronize()
    ka = prof.key_averages()
    rows = []
    for e in ka:
        t = getattr(e, 'device_time_total', None)
        if t is None:
            t = getattr(e, 'cuda_time_total', 0)
        if t and e.device_type is not None and 'CUDA' in str(e.device_type):
            rows.append((e.key, t / 1e6, e.count))
    total = sum(r[1] for r in rows)
    cats = {}
    for k, t, c in rows:
        cats[classify(k)] = cats.get(classify(k), 0.0) + t
    print(f'# total GPU kernel time {total:.3f} s of {wall:.3f} s wall (rest = idle/launch gaps)')
    for c, t in sorted(cats.items(), key=lambda x: -x[1]):
        print(f'  {c:28s} {t:8.3f} s  {100 * t / total:5.1f} %')
    print('# top kernels')
    for k, t, c in sorted(rows, key=lambda r: -r[1])[:args.top]:
        print(f'  {t:8.3f} s  {c:6d}x  {k[:110]}')


if __name__ == '__main__':
    main()
