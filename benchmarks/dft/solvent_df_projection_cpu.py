#!/usr/bin/env python
'''
CPU-only (PySCF + NumPy) validation of the "solvent through the DF tensor" idea for the early SCF stages.

Exact PCM needs, per SCF cycle, two passes of point-charge integrals over (shell pair x surface point)
    v_g  = sum_mn D_mn (mn|g)            (potential of the density at the surface points)
    V_mn = - sum_g q_g (mn|g)            (reaction-field Fock contribution)
In FP64 these are the expensive part of the solvent on GPUs with weak FP64 (see solvent_cost_model.py).
With the density-fitting tensor B_P,mn = (mn|Q) L^-T_QP that the Fock build already holds

    c_P   = (L^-1 W)^T-free form:  rho_P = sum_mn B_P,mn D_mn      (the J-build contraction, free)
    v_g  ~= sum_P Wt_P,g rho_P,      Wt = L^-1 (P|g)               (static, naux x nsurf)
    V_mn ~= - sum_P B_P,mn (Wt q)_P                               (rides the J-build back-contraction)

i.e. both integral passes become O(naux x nsurf) GEMVs plus the existing cderi contractions.  The result is the DF
approximation of the solvent integrals (not exact), so it is only meant for the intermediate SCF stages; the final
stage keeps the exact integrals and fixes the answer.

This script measures, on small molecules, (a) the error of the projected potential / Fock term / solvation energy at the
converged density and (b) a staged SCF: DF-projected solvent until |g| < coarse tolerance, then exact solvent; it prints
the number of exact-solvent cycles needed afterwards and checks that the final energy is identical to a plain exact run.

    python solvent_df_projection_cpu.py --mol glycine --basis def2-tzvp
'''
import argparse
import copy
import numpy as np
from pyscf import gto, scf, df, lib
from pyscf.solvent import pcm as cpu_pcm

MOLS = {
    'water': 'O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587',
    'glycine': '''N -1.25 0.00 0.00; C 0.00 0.60 0.00; C 1.20 -0.30 0.10; O 1.20 -1.50 0.20; O 2.35 0.35 0.05;
                  H -1.30 -1.00 0.00; H -2.00 0.50 0.20; H 0.05 1.25 0.88; H 0.05 1.25 -0.88; H 3.05 -0.25 0.15''',
    'formamide_dimer': '''C 0 0 0; O 1.23 0 0; N -0.67 1.17 0; H -0.15 2.03 0; H -1.68 1.20 0; H -0.55 -0.93 0;
                  C 3.9 2.6 0; O 2.67 2.6 0; N 4.57 1.43 0; H 4.05 0.57 0; H 5.58 1.40 0; H 4.45 3.53 0''',
}


class DFProjectedSolvent:
    '''Replaces PCM._get_v / PCM._get_vmat of a PySCF CPU PCM object by the DF-projected versions.'''

    def __init__(self, pcmobj, auxbasis):
        self.pcm = pcmobj
        mol = pcmobj.mol
        if not pcmobj._intermediates:
            pcmobj.build()
        auxmol = df.make_auxmol(mol, auxbasis)
        j2c = auxmol.intor('int2c2e')
        self.L = np.linalg.cholesky(j2c)                 # j2c = L L^T
        g = pcmobj.surface['grid_coords']
        xi = pcmobj.surface['charge_exp']
        fake = gto.fakemol_for_charges(g, expnt=xi ** 2)
        W = gto.mole.intor_cross('int2c2e', auxmol, fake)     # (P|g)  [naux, nsurf]
        self.Wt = np.linalg.solve(self.L, W)                  # L^-1 W
        self.j3c = df.incore.aux_e2(mol, auxmol, intor='int3c2e', aosym='s1')    # (mn|P) [nao,nao,naux]
        self.B = np.linalg.solve(self.L, self.j3c.reshape(-1, auxmol.nao).T).reshape(auxmol.nao, mol.nao, mol.nao)  # L^-1 (P|mn)
        self.orig = (pcmobj._get_v, pcmobj._get_vmat)

        self.dv = 0.0          # defect corrections, set by refresh()
        self.dV = 0.0

    def get_v(self, dms):
        rho = np.einsum('Pmn,imn->iP', self.B, dms)       # J-build contraction
        return rho @ self.Wt + self.dv                      # [nset, nsurf]

    def get_vmat(self, q):
        q = q.reshape(-1, self.Wt.shape[1])
        x = q @ self.Wt.T                                   # [nset, naux]
        return -np.einsum('Pmn,iP->imn', self.B, x) + self.dV

    def refresh(self, dm):
        '''Defect correction: one exact pass pair at the current density fixes the (smooth, nearly D-independent) DF error,
        v_g(D) ~ v_DF(D) + [v_ex(D0) - v_DF(D0)],  V(q) ~ V_DF(q) + [V_ex(q0) - V_DF(q0)].'''
        self.dv = self.dV = 0.0
        pc = self.pcm
        d = dm.reshape(1, *dm.shape)
        v_ex = self.orig[0](d)
        self.dv = v_ex - self.get_v(d)
        # charges belonging to the corrected potential
        self.on()
        pc._get_vind(dm)
        q0 = pc._intermediates['q_sym'].reshape(1, -1)
        self.dV = self.orig[1](q0) - self.get_vmat(q0)

    def on(self):
        self.pcm._get_v, self.pcm._get_vmat = self.get_v, self.get_vmat

    def off(self):
        self.pcm._get_v, self.pcm._get_vmat = self.orig


def run(args):
    mol = gto.M(atom=MOLS[args.mol], basis=args.basis, verbose=0)
    print(f'# {args.mol} nao={mol.nao} basis={args.basis} aux={args.auxbasis} method={args.method}')

    def make():
        mf = scf.RHF(mol).PCM()
        mf.with_solvent.method = args.method
        mf.with_solvent.lebedev_order = args.lebedev
        mf.conv_tol = 1e-10
        mf.conv_tol_grad = 1e-7
        return mf

    # exact reference
    mf = make()
    e_ref = mf.kernel()
    sol = mf.with_solvent
    dm = mf.make_rdm1()
    print(f'exact PCM SCF: E = {e_ref:.10f}   E_solv = {mf.scf_summary.get("e_solvent", float("nan")):.8f}  nsurf = {sol.surface["grid_coords"].shape[0]}')

    # (a) fixed-density comparison
    proj = DFProjectedSolvent(sol, args.auxbasis)
    v_ex = sol._get_v(dm.reshape(1, *dm.shape))
    v_df = proj.get_v(dm.reshape(1, *dm.shape))
    e_ex, V_ex = sol._get_vind(dm)
    proj.on()
    e_df, V_df = sol._get_vind(dm)
    proj.off()
    print(f'(a) fixed density: max|dv_g| = {abs(v_ex - v_df).max():.3e} (|v_g| ~ {abs(v_ex).max():.2f}),  dE_solv = {e_df - e_ex:.3e} Eh,  '
          f'max|dV| = {abs(V_df - V_ex).max():.3e},  |dV|_F = {np.linalg.norm(V_df - V_ex):.3e}')
    # first-order energy change of the SCF energy caused by the Fock error (variational: energy error is 2nd order in dD)
    # (b) staged SCF
    mf2 = make()
    mf2.max_cycle = 100
    proj2 = copy.copy(proj)          # same static tensors (same molecule, same deterministic surface), other PCM object
    mf2.with_solvent.build()
    proj2.pcm = mf2.with_solvent
    proj2.orig = (proj2.pcm._get_v, proj2.pcm._get_vmat)
    proj2.on()
    mf2.conv_tol = 1e-4
    mf2.conv_tol_grad = args.coarse_tol
    cyc2 = []
    mf2.callback = lambda envs: cyc2.append(envs['cycle'])
    if args.defect is not None:
        # DF-only stage to |g| < args.defect, one exact refresh, then continue with the corrected DF solvent
        mf2.conv_tol, mf2.conv_tol_grad = 1e-2, args.defect
        mf2.kernel()
        dm1 = mf2.make_rdm1()
        proj2.refresh(dm1)
        mf2.conv_tol, mf2.conv_tol_grad = 1e-4, args.coarse_tol
        mf2.kernel(dm0=dm1)
        print(f'    defect-corrected: DF cycles before refresh {len(cyc2)}')
    else:
        mf2.kernel()
    e_coarse = mf2.e_tot
    proj2.off()
    dm_c = mf2.make_rdm1()
    mf3 = make()
    vhf = mf3.get_veff(mol, dm_c)
    g0 = np.linalg.norm(mf3.get_grad(mf2.mo_coeff, mf2.mo_occ, mf3.get_fock(dm=dm_c, vhf=vhf)))
    print(f'    exact-solvent |g| at the DF-solvent coarse density: {g0:.3e}   (coarse stage stopped at {args.coarse_tol:g})')
    cyc = []
    mf3.callback = lambda envs: cyc.append(envs['cycle'])
    mf3.kernel(dm0=dm_c)
    mf4 = make()
    cyc4 = []
    mf4.callback = lambda envs: cyc4.append(envs['cycle'])
    mf4.kernel()
    # plain exact SCF from the same starting point quality (core/minao guess) for the cycle count
    print(f'(b) DF-projected coarse stage: {len(cyc2)} cycles to |g| < {args.coarse_tol:g},  E_coarse(DF-solvent) - E_ref = {e_coarse - e_ref:.3e}')
    print(f'    exact-solvent cycles from the coarse density: {len(cyc)}   vs   from scratch: {len(cyc4)}')
    print(f'    final E (staged) - E_ref = {mf3.e_tot - e_ref:.3e}  ({"identical" if abs(mf3.e_tot - e_ref) < 1e-8 else "DIFFERENT"})')
    return 0


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mol', default='glycine', choices=list(MOLS))
    p.add_argument('--basis', default='def2-tzvp')
    p.add_argument('--auxbasis', default='def2-universal-jkfit')
    p.add_argument('--method', default='IEF-PCM')
    p.add_argument('--lebedev', type=int, default=29)
    p.add_argument('--coarse_tol', type=float, default=1e-3)
    p.add_argument('--defect', type=float, default=None, help='defect-correct the DF solvent once the DF stage reaches this |g|')
    run(p.parse_args())
