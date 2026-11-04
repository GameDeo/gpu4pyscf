#!/usr/bin/env python
'''Kernel-level and timer-level breakdown of the nuclear gradient (RKS + DF + VV10).'''
import argparse, os, re, time
import cupy
from pyscf import gto


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mol', default='033_Bisphenol_A')
    p.add_argument('--basis', default='def2-tzvp')
    p.add_argument('--top', type=int, default=22)
    args = p.parse_args()
    import torch
    from torch.profiler import profile, ProfilerActivity
    from gpu4pyscf import dft
    from gpu4pyscf.dft import numint
    numint.VV10_PRECISION = 'fp32'
    here = os.path.dirname(os.path.abspath(__file__))
    mol = gto.M(atom=os.path.join(here, '..', 'molecules', 'organic', args.mol + '.xyz'),
                basis=args.basis, verbose=0, max_memory=14000)
    mf = dft.RKS(mol, xc='wb97m-v').density_fit()
    mf.grids.atom_grid = (99, 590)
    mf.nlcgrids.atom_grid = (50, 194)
    mf.mixed_precision = {'scheme': 'crt'}
    mf.conv_tol = 1e-9
    mf.kernel()
    g = mf.nuc_grad_method()
    g.verbose = 5
    g.kernel()  # warm
    cupy.cuda.Stream.null.synchronize()
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
        t0 = time.perf_counter()
        g.verbose = 5
        g.kernel()
        cupy.cuda.Stream.null.synchronize()
        wall = time.perf_counter() - t0
    rows = []
    for e in prof.key_averages():
        t = getattr(e, 'device_time_total', None) or getattr(e, 'cuda_time_total', 0)
        if t and 'CUDA' in str(e.device_type):
            rows.append((e.key, t / 1e6, e.count))
    tot = sum(r[1] for r in rows)
    print(f'# gradient wall {wall:.2f} s, GPU kernel time {tot:.2f} s')
    for k, t, c in sorted(rows, key=lambda r: -r[1])[:args.top]:
        print(f'  {t:8.3f} s {c:6d}x {k[:120]}')


if __name__ == '__main__':
    main()
