'''THC (ISDF / LS-THC) exchange as the inner surrogate of the defect-corrected SCF, with the exact DF-K applied only
as K*C_o at the outer steps (occupied-projected defect).  CPU emulation: counts and accuracy, no hardware timings.

K_THC(D) = X [ M o (X^T D X) ] X^T,  X[mu,P] = phi_mu(r_P),  M = hyb Z_FR Z_FR^T + (alpha-hyb) Z_LR Z_LR^T
(one N_P x N_P kernel for full-range + long-range together), Z = S^-1 E, S = (X^T X)o(X^T X), E[P,L] = sum X X B^L.
modes:  thc:c        inner K = K_THC(D)   (N_P = c * nao)
        thc16:c      inner K = K_THC(D_ref) + K_THC^{fp16}(D - D_ref)  (increment with FP16-rounded operands, FP32 accumulate)
        thc16abs:c   inner K = K_THC^{fp16}(D)  (no increment form: shows why the increment form matters)
usage: python r4_cpu_thc.py XYZ BASIS MODE [cand_grid]
'''
import sys, time
import numpy as np
import scipy.linalg
from pyscf import gto, dft, lib
from pyscf.scf import diis as scfdiis

lib.num_threads(4)
xyz, basis, mode = sys.argv[1], sys.argv[2], sys.argv[3]
cand = eval(sys.argv[4]) if len(sys.argv) > 4 else (40, 146)
xc = 'wb97m_v'
mol = gto.M(atom=xyz, basis=basis, verbose=0)
mf = dft.RKS(mol).density_fit(auxbasis='def2-universal-jkfit')
mf.xc = xc
mf.nlc = 'VV10'
ni = mf._numint
omega, alpha, hyb = ni.rsh_and_hybrid_coeff(xc, spin=mol.spin)
nocc = mol.nelectron // 2
nao = mol.nao


def grid(ag):
    g = dft.gen_grid.Grids(mol)
    g.atom_grid = ag
    g.build()
    return g


gf, nf, gc, nc = grid((99, 590)), grid((50, 194)), grid((20, 50)), grid((10, 26))
s1e = mf.get_ovlp()
h1e = mf.get_hcore()
w, U = np.linalg.eigh(s1e)
Sh = (U * np.sqrt(w)) @ U.T

# ---------------- THC construction
t0 = time.time()
mf.with_df.build()
B_fr = lib.unpack_tril(mf.with_df._cderi)
with mf.with_df.range_coulomb(omega) as dflr:
    dflr.build()
    B_lr = lib.unpack_tril(dflr._cderi)
naux = B_fr.shape[0]
c_np = float(mode.split(':')[1])
NP = int(round(c_np * nao))
gcand = grid(cand)
phi = ni.eval_ao(mol, gcand.coords) * gcand.weights[:, None] ** 0.25      # ng x nao (weighted)
rng = np.random.default_rng(0)
k = NP + 20
G = (phi @ rng.standard_normal((nao, k))) * (phi @ rng.standard_normal((nao, k)))   # sketch of pair products
_, _, piv = scipy.linalg.qr(G.T, pivoting=True, mode='economic')
pts = np.sort(piv[:NP])
X = phi[pts].T.copy()                                                      # nao x NP
S = (X.T @ X) ** 2


def fitZ(B):
    T = np.einsum('lmn,np->lmp', B, X)
    E = np.einsum('lmp,mp->pl', T, X)
    return scipy.linalg.lstsq(S, E, cond=1e-12)[0]                         # NP x naux


Zf, Zl = fitZ(B_fr), fitZ(B_lr)
M = hyb * Zf @ Zf.T + (alpha - hyb) * Zl @ Zl.T
t_thc = time.time() - t0
print(f'nao {nao} naux {naux} NP {NP} (c={c_np}) cand {gcand.size} THC build {t_thc:.1f}s', flush=True)
Xh, Mh = X.astype(np.float16).astype(np.float32), None


def K_thc(D, fp16=False):
    if not fp16:
        return X @ (M * (X.T @ D @ X)) @ X.T
    s = abs(D).max() or 1.0
    Dh = (D / s).astype(np.float16).astype(np.float32)
    sm = abs(M).max()
    Mh = (M / sm).astype(np.float16).astype(np.float32)
    Y = (Xh.T @ Dh @ Xh)
    Y = Y.astype(np.float16).astype(np.float32)
    return (Xh @ (Mh * Y) @ Xh.T).astype(np.float64) * (s * sm)


cnt = dict(fineXC=0, coarseXC=0, Kexact_full=0, Kexact_occ=0, Kthc=0)


def Kgen(dm):
    k = hyb * mf.get_k(mol, dm) if hyb != 0 else 0
    if omega != 0:
        k = k + (alpha - hyb) * mf.get_k(mol, dm, omega=omega)
    return k


def vxc(dm, fine, nlc):
    g, ng = (gf, nf) if fine else (gc, nc)
    cnt['fineXC' if fine else 'coarseXC'] += 1
    v = ni.nr_rks(mol, g, xc, dm)[2]
    if nlc:
        v = v + ni.nr_nlc_vxc(mol, ng, xc, dm)[2]
    return v


def cnorm(f, dm):
    x = f @ dm @ s1e
    return np.linalg.norm(x - x.T)


def solve(f):
    e, mo = mf.eig(f, s1e)
    return mf.make_rdm1(mo, mf.get_occ(e, mo))


def occ_orbitals(D):
    ev, V = np.linalg.eigh(Sh @ D @ Sh)
    return np.linalg.solve(Sh, V[:, -nocc:])


fp16 = mode.startswith('thc16')
incr = mode.startswith('thc16:')
# ---------------- stage 1: coarse XC + coarse NLC + THC K
dm = mf.get_init_guess(key='minao')
d = scfdiis.CDIIS(mf)
for c in range(80):
    cnt['Kthc'] += 1
    f = h1e + mf.get_j(mol, dm) + vxc(dm, False, True) - 0.5 * K_thc(dm, fp16 and not incr)
    g = cnorm(f, dm)
    if g < 1e-3:
        break
    f = d.update(s1e, dm, f)
    dm = solve(f)
print('stage1 cycles', c + 1, flush=True)
Dref = dm
hist = []
for outer in range(15):
    Kr = Kgen(Dref)           # emulates the exact K*C_o (only Kr @ C is used below)
    cnt['Kexact_occ'] += 1
    Vf = vxc(Dref, True, True)
    Ftrue = h1e + mf.get_j(mol, Dref) + Vf - 0.5 * Kr
    gt = cnorm(Ftrue, Dref)
    hist.append(gt)
    if outer == 0:
        kt = K_thc(Dref)
        print('  THC K rel err %.2e  dE_K %.2e' % (np.linalg.norm(kt - Kr) / np.linalg.norm(Kr),
                                                 -0.25 * np.einsum('ij,ji', Dref, kt - Kr)), flush=True)
    print(f'OUTER {outer}: |g_true|={gt:.2e}', flush=True)
    if gt < 1e-7:
        break
    Vc = vxc(Dref, False, False)
    Kt_ref = K_thc(Dref)
    Fcheap = h1e + mf.get_j(mol, Dref) + Vc - 0.5 * Kt_ref
    C = occ_orbitals(Dref)
    W = (Ftrue - Fcheap) @ C
    SC = s1e @ C
    Delta = W @ SC.T + SC @ W.T - SC @ (C.T @ W) @ SC.T
    dd = scfdiis.CDIIS(mf)
    dmi, f = Dref, Ftrue
    inner_tol = max(gt * 1e-2, 1e-9)
    for it in range(60):
        f = dd.update(s1e, dmi, f)
        dmi = solve(f)
        cnt['Kthc'] += 1
        if incr:
            Kin = Kt_ref + K_thc(dmi - Dref, fp16=True)
        else:
            Kin = K_thc(dmi, fp16=fp16)
        f = h1e + mf.get_j(mol, dmi) + vxc(dmi, False, False) - 0.5 * Kin + Delta
        gs = cnorm(f, dmi)
        if gs < inner_tol:
            break
    print(f'    inner cycles {it + 1}  |g_s|={gs:.1e}', flush=True)
    Dref = dmi
print('RESULT', mode, 'NP', NP, 'outer_hist', ['%.1e' % x for x in hist], 'counts', cnt, flush=True)
