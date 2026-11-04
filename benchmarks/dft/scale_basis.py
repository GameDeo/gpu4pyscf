#!/usr/bin/env python
'''
Basis-set scaling of the 'crt' scheme: one molecule, increasing basis (def2-SVP ... def2-QZVPPD), wB97M-V + DF.
Reports nao, naux, DF-tensor size / location, SCF wall time, cycles, and the energy difference to a run with more INT8
moduli (crt_moduli_final=14, 16 for the DF build) as an internal accuracy check.
usage: scale_basis.py --mol 020_Vitamin_C --bases def2-svp,def2-tzvp,def2-qzvpp [--ref] [--fp64]
'''
import argparse, os, time
import cupy
from pyscf import gto


def run(mol, scheme, grid, nlcgrid, tag, verbose=0):
    from gpu4pyscf import dft
    mf = dft.RKS(mol, xc='wb97m-v').density_fit()
    mf.grids.atom_grid = grid
    mf.nlcgrids.atom_grid = nlcgrid
    mf.conv_tol = 1e-9
    mf.max_cycle = 60
    if scheme:
        mf.mixed_precision = scheme
    cupy.cuda.Stream.null.synchronize()
    t = time.perf_counter()
    e = mf.kernel()
    cupy.cuda.Stream.null.synchronize()
    ts = time.perf_counter() - t
    df = mf.with_df
    c = df._cderi
    on_gpu = isinstance(c[0], cupy.ndarray) if isinstance(c, (list, tuple)) else isinstance(c, cupy.ndarray)
    sz = sum(x.size * 8 for x in c) / 1e9 if isinstance(c, (list, tuple)) else c.size * 8 / 1e9
    print(f'  {tag:10s} E={e:.10f} conv={mf.converged} cycles={mf.cycles} coarse={getattr(mf, "cycles_coarse", "-")} '
          f'time={ts:7.1f} s  naux={df.naux}  DF tensor {sz:.2f} GB ({"GPU" if on_gpu else "host"})  '
          f'mem pool {cupy.get_default_memory_pool().total_bytes() / 1e9:.1f} GB', flush=True)
    return mf, e, ts


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mol', default='020_Vitamin_C')
    p.add_argument('--bases', default='def2-svp,def2-tzvp,def2-tzvpd,def2-qzvpp')
    p.add_argument('--grid', type=int, nargs=2, default=(75, 302))
    p.add_argument('--nlcgrid', type=int, nargs=2, default=(50, 194))
    p.add_argument('--ref', action='store_true', help='also run with more moduli and report the difference')
    p.add_argument('--fp64', action='store_true', help='also run the plain FP64 SCF')
    p.add_argument('--grad', action='store_true')
    args = p.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    for basis in args.bases.split(','):
        mol = gto.M(atom=os.path.join(here, '..', 'molecules', 'organic', args.mol + '.xyz'), basis=basis,
                    verbose=0, max_memory=14000)
        print(f'# {args.mol} {basis}: nao={mol.nao} natm={mol.natm}', flush=True)
        try:
            mf, e, t = run(mol, {'scheme': 'crt'}, tuple(args.grid), tuple(args.nlcgrid), 'crt')
            if args.grad:
                t0 = time.perf_counter(); g = mf.nuc_grad_method().kernel(); cupy.cuda.Stream.null.synchronize()
                print(f'  gradient {time.perf_counter() - t0:.1f} s  |g|max={abs(g).max():.2e}', flush=True)
            del mf
            cupy.get_default_memory_pool().free_all_blocks()
            if args.ref:
                mf2, e2, t2 = run(mol, {'scheme': 'crt', 'crt_moduli_final': 14, 'crt_moduli_cderi': 17},
                                  tuple(args.grid), tuple(args.nlcgrid), 'crt-14')
                print(f'  dE(crt11 - crt14) = {e - e2:+.2e}', flush=True)
                del mf2
                cupy.get_default_memory_pool().free_all_blocks()
            if args.fp64:
                mf3, e3, t3 = run(mol, False, tuple(args.grid), tuple(args.nlcgrid), 'fp64')
                print(f'  dE(crt - fp64) = {e - e3:+.2e}  speedup {t3 / t:.1f}x', flush=True)
                del mf3
                cupy.get_default_memory_pool().free_all_blocks()
        except Exception as ex:
            import traceback
            print(f'  FAILED: {type(ex).__name__}: {str(ex)[:200]}', flush=True)
            traceback.print_exc(limit=4)
            cupy.get_default_memory_pool().free_all_blocks()


if __name__ == '__main__':
    main()
