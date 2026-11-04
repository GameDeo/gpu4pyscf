#!/usr/bin/env python
'''
CPU-only static cost model of the implicit-solvent part of a GPU4PySCF Fock build / gradient.

No GPU, no cupy: it re-implements only the *counting* of
  * the PCM surface points (SWIG switching, same radii / Lebedev grids as gpu4pyscf.solvent.pcm.gen_surface),
  * the primitive shell-pair x surface-point work of the two int1e_grids calls per SCF cycle
    (density-contracted potential at the surface, charge-contracted Fock contribution),
  * the dense linear algebra of PCM.build (S, D, DA S GEMM, LU) and of one gradient,
and converts them to seconds for a few GPUs with a stated efficiency.  The absolute numbers carry a factor-2 uncertainty
(the per-primitive FLOP count is a model, not a measurement); the scaling with natm / nsurf / nao is what matters.

    python solvent_cost_model.py --mol 033_Bisphenol_A --basis def2-tzvp --ng 302
'''
import os
import argparse
import numpy as np
from pyscf import gto
from pyscf.data import radii
from pyscf.dft.LebedevGrid import MakeAngularGrid

BOHR = radii.BOHR
GPUS = {  # FP64 non-tensor-core peak (TFLOP/s), FP64 tensor-core peak, HBM/GDDR bandwidth (GB/s), memory (GB)
    'T4':   dict(fp64=0.25, fp64_tc=0.25, bw=300, mem=15),
    'A100': dict(fp64=9.7, fp64_tc=19.5, bw=1555, mem=40),
    'H100': dict(fp64=34., fp64_tc=67., bw=3350, mem=80),
}
KERNEL_EFF = 0.35   # fraction of FP64 peak reached by the Rys-type point-charge kernels (transcendental heavy)
DENSE_EFF = 0.75    # fraction of peak reached by cuBLAS DGEMM / cuSOLVER getrf at these sizes


def swig_surface_count(mol, ng=302, vdw_scale=1.2):
    '''Number of surface points that survive the SWIG switching (replica of pcm.gen_surface, NumPy).'''
    ang = MakeAngularGrid(ng)
    mb = radii.VDW.copy()
    mb[1] = 1.1 / BOHR
    rad = vdw_scale * mb
    coords = mol.atom_coords()
    Z = [gto.charge(e) for e in mol.elements]
    R = np.array([rad[z] for z in Z])
    R_sw = R * (14.0 / ng) ** 0.5
    alpha = 0.5 + R / R_sw - ((R / R_sw) ** 2 - 1.0 / 28) ** 0.5
    R_in = R - alpha * R_sw
    n = 0
    for ia in range(mol.natm):
        pts = R[ia] * ang[:, :3] + coords[ia]
        w = ang[:, 3] * 4 * np.pi
        r = np.linalg.norm(pts[:, None, :] - coords[None], axis=-1)
        d = (r - R_in) / R_sw
        d[:, ia] = 1.0
        d[d < 1e-8] = 0.0
        x = np.clip(d, 0, 1)
        f = x ** 3 * (10 - 15 * x + 6 * x ** 2)
        sw = f.prod(axis=1)
        n += int((w * sw > 1e-12).sum())
    return n


def shell_pair_primitives(mol, thr=1e-13):
    '''Histogram {(li, lj): [n_contracted_pairs, n_primitive_pairs]} for segmented shells, pairs screened by max|S_ij| > thr
    (proxy of the int1e_grids q_cond screening at 1e-14).'''
    S = mol.intor('int1e_ovlp')
    ao_loc = mol.ao_loc_nr()
    nbas = mol.nbas
    ls = mol._bas[:, 1]
    nprim = mol._bas[:, 2]
    nctr = mol._bas[:, 3]
    hist = {}
    for i in range(nbas):
        for j in range(i + 1):
            blk = S[ao_loc[i]:ao_loc[i + 1], ao_loc[j]:ao_loc[j + 1]]
            if np.abs(blk).max() < thr:
                continue
            # a shell with nctr contracted functions sharing primitives is replicated nctr times by basis_seg_contraction
            ncp = nctr[i] * nctr[j]
            key = (int(ls[i]), int(ls[j]))
            h = hist.setdefault(key, [0, 0])
            h[0] += ncp
            h[1] += ncp * int(nprim[i]) * int(nprim[j])
    return hist


def flops_per_prim_grid(li, lj, charge_contracted):
    '''Modelled FP64 instruction count of one (primitive pair, surface point) evaluation.'''
    L = li + lj
    nr = L // 2 + 1
    nherm = (L + 1) * (L + 2) * (L + 3) // 6
    c = 40.0 + nr * (40.0 + 9.0 * (L + 1)) + nherm * (2 * nr + 2)
    if charge_contracted:
        ncart = (li + 1) * (li + 2) // 2 * (lj + 1) * (lj + 2) // 2
        c += ncart * (2 * nr + 2) + 3 * nr * (L + 1) * 2   # form all AO components, apply the HRR
    return c


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mol', default='033_Bisphenol_A')
    p.add_argument('--basis', default='def2-tzvp')
    p.add_argument('--ng', type=int, default=302, help='Lebedev points per atom (PCM lebedev_order 29 -> 302; SMD sasa_ng 590)')
    p.add_argument('--gpus', nargs='*', default=['T4', 'A100', 'H100'])
    p.add_argument('--method', default='IEF-PCM', choices=['C-PCM', 'IEF-PCM', 'SS(V)PE'])
    args = p.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    path = args.mol if os.path.exists(args.mol) else os.path.join(here, '..', 'molecules', 'organic', args.mol + '.xyz')
    mol = gto.M(atom=path, basis=args.basis, verbose=0)
    n = swig_surface_count(mol, args.ng)
    nao = mol.nao
    hist = shell_pair_primitives(mol)
    prim_pairs = sum(v[1] for v in hist.values())
    cpairs = sum(v[0] for v in hist.values())
    w_pot = sum(v[1] * flops_per_prim_grid(l[0], l[1], False) for l, v in hist.items())  # per surface point
    w_fock = sum(v[1] * flops_per_prim_grid(l[0], l[1], True) for l, v in hist.items())
    print(f'{args.mol}: natm={mol.natm} nao={nao} basis={args.basis} Lebedev/atom={args.ng} method={args.method}')
    print(f'  nsurf = {n}  ({n / mol.natm:.0f} per atom, {n / (mol.natm * args.ng):.2f} of the full spheres)')
    print(f'  contracted shell pairs kept = {cpairs}, primitive pairs = {prim_pairs}  (avg {prim_pairs / cpairs:.1f} per pair)')
    print(f'  per SCF cycle: int1e_grids(dm)  : {float(n) * float(prim_pairs):.3e} prim-pair x point evaluations, {n * w_pot / 1e9:.1f} GFLOP(model)')
    print(f'                 int1e_grids(q)   : {float(n) * float(prim_pairs):.3e}                              , {n * w_fock / 1e9:.1f} GFLOP(model)')
    # dense linear algebra
    cpcm = args.method == 'C-PCM'
    mem_final = (1 if cpcm else 4) * n * n * 8 / 1e9        # K_LU (C-PCM) | S, D, K_LU (+A vector)
    mem_peak = (2 if cpcm else 6) * n * n * 8 / 1e9         # build temporaries: DA, DAS, K next to S, D
    fl_gemm = 0.0 if cpcm else 2.0 * n ** 3
    fl_lu = 2.0 / 3.0 * n ** 3
    fl_R = 0.0 if cpcm else 4.0 * n * n                      # DA matvecs in left_multiply_R (2 per cycle)
    print(f'  build: DA*S GEMM {fl_gemm / 1e12:.2f} TFLOP, LU {fl_lu / 1e12:.2f} TFLOP; resident {mem_final:.2f} GB, peak {mem_peak:.2f} GB')
    print(f'  per-cycle dense: 2 LU solves (read 2 x {n * n * 8 / 1e9:.2f} GB), '
          f'R matvecs {fl_R / 1e9:.2f} GFLOP + {0 if cpcm else 2 * n * n * 8 / 1e9:.2f} GB temporaries (DA = D*A)')
    print(f'  gradient: 3 cache-missing re-evaluations of the SCF solvent step (grad_qv, grad_solver, grad_nuc), '
          f'dD/dS/dA are [3,n,n] = {3 * n * n * 8 / 1e9:.2f} GB each (IEF-PCM dense path)')
    print()
    print(f'  {"GPU":6s} {"int(dm) s":>10s} {"int(q) s":>10s} {"solvent/cycle s":>16s} {"build s":>9s} {"fits":>5s}')
    for g in args.gpus:
        spec = GPUS[g]
        t_pot = n * w_pot / (spec['fp64'] * 1e12 * KERNEL_EFF)
        t_fock = n * w_fock / (spec['fp64'] * 1e12 * KERNEL_EFF)
        t_dense = (fl_gemm + fl_lu) / (spec['fp64_tc'] * 1e12 * DENSE_EFF)
        t_solve = 2 * n * n * 8 * 2 / (spec['bw'] * 1e9) + fl_R / (spec['fp64'] * 1e12) + 6 * n * n * 8 / (spec['bw'] * 1e9)
        print(f'  {g:6s} {t_pot:10.2f} {t_fock:10.2f} {t_pot + t_fock + t_solve:16.2f} {t_dense:9.1f} '
              f'{"yes" if mem_peak < spec["mem"] * 0.8 else "NO":>5s}')


if __name__ == '__main__':
    main()
