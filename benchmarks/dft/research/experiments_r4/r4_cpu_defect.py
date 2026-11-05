'''CPU emulation of round-4 multi-fidelity SCF variants (counts of expensive evaluations; no hardware timings).

Outer step (fine, "exact"): F_true(D_ref) = h + J + V_xc,fine + V_nlc,fine - 1/2 K_exact.
Inner surrogate:           F_s(D) = h + J(D) + V_xc,coarse(D) [semilocal only, NLC frozen] - 1/2 K_inner(D) + Delta
modes
  full   : Delta = full matrix (F_true - F_cheap)(D_ref)                     [round-3 dual grid, frozen NLC]
  occ    : Delta~ built from W = (F_true - F_cheap)(D_ref) C_o only            [fine evaluations as F*C_o]
           Delta~ = W C^T S + S C W^T - S C (C^T W) C^T S  (exact on the occupied space at D_ref)
  occK   : occ + K_inner from a smaller aux basis (cc-pvdz-jkfit), its defect folded into W
  lrK:t  : full + K_inner(D) = K(D_ref) + K(trunc_t(D - D_ref)), eigen-truncated at relative tolerance t
  occlrK:t : occ, inner K = K_cheap(D_ref) + K_cheap(trunc(dD))... (not used)
Also logs the numerical rank of dD = D - D_ref inside the inner loops.
usage: python r4_cpu_defect.py XYZ BASIS MODE [coarse_xc coarse_nlc]
'''
import sys
import numpy as np
from pyscf import gto, dft, lib
from pyscf.scf import diis as scfdiis

lib.num_threads(4)
xyz, basis, mode = sys.argv[1], sys.argv[2], sys.argv[3]
cxc = eval(sys.argv[4]) if len(sys.argv) > 4 else (20, 50)
cnl = eval(sys.argv[5]) if len(sys.argv) > 5 else (10, 26)
xc = 'wb97m_v'
mol = gto.M(atom=xyz, basis=basis, verbose=0)
mf = dft.RKS(mol).density_fit(auxbasis='def2-universal-jkfit')
mf.xc = xc
mf.nlc = 'VV10'
ni = mf._numint
omega, alpha, hyb = ni.rsh_and_hybrid_coeff(xc, spin=mol.spin)
nocc = mol.nelectron // 2


def grid(ag):
    g = dft.gen_grid.Grids(mol)
    g.atom_grid = ag
    g.build()
    return g


gf, nf, gc, nc = grid((99, 590)), grid((50, 194)), grid(cxc), grid(cnl)
s1e = mf.get_ovlp()
h1e = mf.get_hcore()
w, U = np.linalg.eigh(s1e)
Sh = (U * np.sqrt(w)) @ U.T
mfc = dft.RKS(mol).density_fit(auxbasis='cc-pvdz-jkfit')   # cheap K surrogate (smaller aux basis)
print('nao', mol.nao, 'naux', mf.with_df.auxmol.nao if mf.with_df.auxmol else None, flush=True)
cnt = dict(fineXC=0, coarseXC=0, nlc_fine=0, nlc_coarse=0, Kexact=0, Kcheap=0, K_lr_rank_sum=0)


def Kgen(m, dm):
    k = hyb * m.get_k(mol, dm) if hyb != 0 else 0
    if omega != 0:
        k = k + (alpha - hyb) * m.get_k(mol, dm, omega=omega)
    return k


def Kexact(dm):
    cnt['Kexact'] += 1
    return Kgen(mf, dm)


def Kcheap(dm):
    cnt['Kcheap'] += 1
    return Kgen(mfc, dm)


def vxc(dm, fine, nlc):
    g, ng = (gf, nf) if fine else (gc, nc)
    cnt['fineXC' if fine else 'coarseXC'] += 1
    v = ni.nr_rks(mol, g, xc, dm)[2]
    if nlc:
        cnt['nlc_fine' if fine else 'nlc_coarse'] += 1
        v = v + ni.nr_nlc_vxc(mol, ng, xc, dm)[2]
    return v


def cnorm(f, dm):
    x = f @ dm @ s1e
    return np.linalg.norm(x - x.T)


def solve(f):
    e, mo = mf.eig(f, s1e)
    occ = mf.get_occ(e, mo)
    return mf.make_rdm1(mo, occ), mo, occ


def occ_orbitals(D):
    # S-orthonormal occupied orbitals spanning D (D = 2 C C^T)
    ev, V = np.linalg.eigh(Sh @ D @ Sh)
    X = V[:, -nocc:]
    return np.linalg.solve(Sh, X)


def rank_stats(dD):
    ev = np.linalg.eigvalsh(Sh @ dD @ Sh)
    a = np.sort(abs(ev))[::-1]
    if a[0] == 0:
        return '0'
    return 'max %.1e r(1e-1)=%d r(1e-2)=%d r(1e-3)=%d r(1e-4)=%d' % (
        a[0], (a > 1e-1 * a[0]).sum(), (a > 1e-2 * a[0]).sum(), (a > 1e-3 * a[0]).sum(), (a > 1e-4 * a[0]).sum())


def trunc(dD, t):
    ev, V = np.linalg.eigh(Sh @ dD @ Sh)
    keep = abs(ev) > t * abs(ev).max()
    Vk = np.linalg.solve(Sh, V[:, keep])
    return (Vk * ev[keep]) @ Vk.T, int(keep.sum())


# ---------------- stage 1: everything cheap (coarse XC incl. coarse NLC, cheap K for occK) to |g| < 1e-3
cheapK = mode.startswith('occK')
dm = mf.get_init_guess(key='minao')
d = scfdiis.CDIIS(mf)
for c in range(60):
    f = h1e + mf.get_j(mol, dm) + vxc(dm, False, True) - 0.5 * (Kcheap(dm) if cheapK else Kexact(dm))
    g = cnorm(f, dm)
    if g < 1e-3:
        break
    f = d.update(s1e, dm, f)
    dm = solve(f)[0]
print('stage1 cycles', c + 1, 'g', '%.1e' % g, flush=True)
st1 = dict(cnt)
Dref = dm
lr_t = float(mode.split(':')[1]) if ':' in mode else None
hist = []
for outer in range(12):
    Kr = Kexact(Dref)
    Vf = vxc(Dref, True, True)
    Ftrue = h1e + mf.get_j(mol, Dref) + Vf - 0.5 * Kr
    gt = cnorm(Ftrue, Dref)
    hist.append(gt)
    print(f'OUTER {outer}: |g_true|={gt:.2e}', flush=True)
    if gt < 1e-7:
        break
    Vc = vxc(Dref, False, False)
    if cheapK:
        Kc = Kcheap(Dref)
        Fcheap = h1e + mf.get_j(mol, Dref) + Vc - 0.5 * Kc
    else:
        Fcheap = h1e + mf.get_j(mol, Dref) + Vc - 0.5 * Kr
    Dfull = Ftrue - Fcheap
    if mode.startswith('occ'):
        C = occ_orbitals(Dref)
        W = Dfull @ C
        SC = s1e @ C
        Delta = W @ SC.T + SC @ W.T - SC @ (C.T @ W) @ SC.T
    else:
        Delta = Dfull
    dd = scfdiis.CDIIS(mf)
    dmi, f = Dref, Ftrue
    inner_tol = max(gt * 1e-2, 1e-9)
    for it in range(60):
        f = dd.update(s1e, dmi, f)
        dmi = solve(f)[0]
        dD = dmi - Dref
        if lr_t is not None:
            dDt, r = trunc(dD, lr_t)
            cnt['K_lr_rank_sum'] += r
            Kin = Kr + Kgen(mf, dDt)
        elif cheapK:
            Kin = Kcheap(dmi)
        else:
            Kin = Kexact(dmi)
        f = h1e + mf.get_j(mol, dmi) + vxc(dmi, False, False) - 0.5 * Kin + Delta
        gs = cnorm(f, dmi)
        if it in (0, 2) or gs < inner_tol:
            print(f'    it {it} |g_s|={gs:.1e} dD {rank_stats(dD)}', flush=True)
        if gs < inner_tol:
            break
    print(f'    inner cycles {it + 1}', flush=True)
    Dref = dmi
e = mf.energy_tot(Dref, h1e, None) if False else None
print('RESULT', mode, 'outer_hist', ['%.1e' % x for x in hist], 'counts', cnt, 'stage1', st1, flush=True)
