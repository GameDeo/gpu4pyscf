#!/usr/bin/env python
'''Kernel-level profile of one DF get_jk (J + K) call for a given K precision.
usage: profile_k.py [--kprec fp64|crt] [--crt_n 11] [--mol 033_Bisphenol_A]'''
import argparse, os, re, time
import cupy
from pyscf import gto


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mol', default='033_Bisphenol_A')
    p.add_argument('--basis', default='def2-tzvp')
    p.add_argument('--kprec', default='crt')
    p.add_argument('--crt_n', type=int, default=11)
    p.add_argument('--top', type=int, default=16)
    args = p.parse_args()
    import torch
    from torch.profiler import profile, ProfilerActivity
    from gpu4pyscf import dft
    from gpu4pyscf.df import df_jk
    here = os.path.dirname(os.path.abspath(__file__))
    mol = gto.M(atom=os.path.join(here, '..', 'molecules', 'organic', args.mol + '.xyz'),
                basis=args.basis, verbose=0, max_memory=14000)
    mf = dft.RKS(mol, xc='wb97m-v').density_fit()
    mf.grids.atom_grid = (30, 110)
    mf.nlcgrids.atom_grid = (20, 50)
    mf.max_cycle = 3
    mf.kernel()
    dm = mf.make_rdm1()
    df_jk.K_PRECISION = 'fp64'
    mf.get_jk(mol, dm)
    cupy.cuda.Stream.null.synchronize()
    t = time.perf_counter(); vj0, vk0 = mf.get_jk(mol, dm); cupy.cuda.Stream.null.synchronize()
    print(f'# get_jk FP64: {time.perf_counter() - t:.3f} s')
    df_jk.K_PRECISION = args.kprec
    df_jk.K_CRT_MODULI = args.crt_n
    mf.get_jk(mol, dm)
    cupy.cuda.Stream.null.synchronize()
    t = time.perf_counter(); vj, vk = mf.get_jk(mol, dm); cupy.cuda.Stream.null.synchronize()
    print(f'# get_jk {args.kprec}: {time.perf_counter() - t:.3f} s   max|dK|={float(cupy.abs(vk - vk0).max()):.2e} '
          f'max|dJ|={float(cupy.abs(vj - vj0).max()):.2e}')
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
        mf.get_jk(mol, dm)
        cupy.cuda.Stream.null.synchronize()
    rows = []
    for e in prof.key_averages():
        tt = getattr(e, 'device_time_total', None) or getattr(e, 'cuda_time_total', 0)
        if tt and e.device_type is not None and 'CUDA' in str(e.device_type):
            rows.append((e.key, tt / 1e6, e.count))
    print(f'# total GPU kernel time {sum(r[1] for r in rows):.3f} s')
    for k, tt, c in sorted(rows, key=lambda r: -r[1])[:args.top]:
        print(f'  {tt:8.3f} s  {c:6d}x  {k[:110]}')


if __name__ == '__main__':
    main()
