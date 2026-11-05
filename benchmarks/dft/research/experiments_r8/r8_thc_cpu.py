'''Round 8, CPU only: does an ISDF/THC surrogate (fitted from the exact jkfit DF tensor) keep the multi-fidelity
contraction rate, without stored O(npairs naux) surrogate tensors?

HF/def2-TZVP on BPA (pyscf, CPU).  Exact operator: DF J and K with def2-universal-jkfit.  For each surrogate we emulate
the multi-fidelity outer loop of gpu4pyscf/scf/multifidelity.py exactly (exact J at the checkpoint as reference,
surrogate J increments, surrogate K, occupied-projected defect Delta~), starting from a common density D0, and report the
exact orbital gradient at each checkpoint.  The ratio |g_k| / |g_k+1| is the contraction per exact evaluation.

Surrogates:
  svpri+exJ   K: DF def2-svp-ri, J: exact (the round-7 dense-FP32-K default; ~400x on wB97M-V)
  svpri+jfit  K: DF def2-svp-ri, J increments: DF def2-universal-jfit (the f32 host path; ~100x on wB97M-V)
  thcN        K and J increments from LS-THC of the exact jkfit tensor, N interpolation points
  thcN+exJ    THC K, exact J
Only ratios between surrogates are meaningful (HF weights exchange differently from wB97M-V).

usage: python r8_thc_cpu.py [ng_factors, e.g. 4,8,12] [molecule xyz]
'''
import os, sys, time
import numpy as np
import scipy.linalg
from pyscf import gto, scf, df, lib, dft
from pyscf.scf import diis as pdiis

here = os.path.dirname(os.path.abspath(__file__))
ngf = [float(x) for x in (sys.argv[1] if len(sys.argv) > 1 else '4,8,12').split(',')]
xyz = sys.argv[2] if len(sys.argv) > 2 else os.path.join(here, '..', '..', '..', 'molecules', 'organic',
                                                         '033_Bisphenol_A.xyz')
mol = gto.M(atom=xyz, basis='def2-tzvp', verbose=0, max_memory=12000)
nao = mol.nao
nocc = mol.nelectron // 2
T0 = time.time()
THC_RCOND = float(os.environ.get('THC_RCOND', '1e-14'))


def log(*a):
    print('[%7.1f s]' % (time.time() - T0), *a, flush=True)


log('nao', nao, 'nocc', nocc)
S = mol.intor('int1e_ovlp')
h1 = scf.hf.get_hcore(mol)
w_s = np.linalg.eigvalsh(S)
log('lambda_min(S) %.2e, %d below 1e-3' % (w_s[0], (w_s < 1e-3).sum()))


def dfobj(aux):
    d = df.DF(mol, auxbasis=aux)
    d.build()
    return d


def df_jk(d):
    def j(D):
        return df.df_jk.get_jk(d, D, hermi=1, with_k=False)[0]

    def k(D):
        return df.df_jk.get_jk(d, D, hermi=1, with_j=False)[1]
    return j, k


dfx = dfobj('def2-universal-jkfit')
log('exact jkfit naux', dfx.get_naoaux())
J_ex, K_ex = df_jk(dfx)
J_jf, _ = df_jk(dfobj('def2-universal-jfit'))
_, K_svp = df_jk(dfobj('def2-svp-ri'))


def fock(D, J, K):
    return h1 + J(D) - 0.5 * K(D)


def solve(F):
    e, C = scipy.linalg.eigh(F, S)
    return e, C


def dens(C):
    Co = C[:, :nocc]
    return 2 * Co @ Co.T, Co


def scf_loop(fockfn, D, Co, tol, maxc=80):
    '''DIIS SCF from (D, Co); the gradient is taken with the orbitals that produced D'''
    d = pdiis.CDIIS()
    d.space = 8
    for it in range(maxc):
        F = fockfn(D)
        if Co is not None:
            g = 2 * np.linalg.norm((np.eye(nao) - D @ S / 2).T @ F @ Co)
            if g < tol:
                break
        e, C = solve(d.update(S, D, F))
        D, Co = dens(C)
    return D, Co, it + 1, g


# ---------------------------------------------------------------- LS-THC of the exact jkfit tensor
def thc_build(ng):
    '''interpolation points by weighted K-means (Dong, Hu, Lin, JCTC 14, 1311 (2018)) on a DFT grid, weight
    sum_m phi_m(r)^2; X = E G^-1 with E_Lg = sum_mn B_Lmn phi_m(g) phi_n(g) and G = (P P^T)^2 (least squares in the
    Coulomb metric, since B is the metric-orthonormalised jkfit tensor); V = X^T X'''
    from scipy.spatial import cKDTree
    t = time.time()
    g = dft.gen_grid.Grids(mol)
    g.atom_grid = (40, 194)
    g.prune = dft.gen_grid.nwchem_prune
    g.build()
    pts = g.coords
    phi = dft.numint.eval_ao(mol, pts)
    w = np.einsum('gi,gi->g', phi, phi)
    rng = np.random.default_rng(1)
    cen = pts[rng.choice(len(pts), ng, replace=False, p=w / w.sum())]
    for it in range(20):
        lab = cKDTree(cen).query(pts)[1]
        ws = np.bincount(lab, w, ng)
        new = np.stack([np.bincount(lab, w * pts[:, x], ng) for x in range(3)], 1)
        ok = ws > 0
        cen[ok] = new[ok] / ws[ok, None]
    piv = np.unique(cKDTree(pts).query(cen)[1])
    ng = len(piv)
    log('  THC: %d K-means points from %d grid points, %.1f s' % (ng, len(pts), time.time() - t))
    P = phi[piv]                                      # (ng, nao)
    # E_Lg from the packed exact tensor (lower triangle, off-diagonal x2)
    tri = np.tril_indices(nao)
    fac = np.where(tri[0] == tri[1], 1.0, 2.0)
    E = np.zeros((dfx.get_naoaux(), ng))
    for g0, g1 in lib.prange(0, ng, 256):
        pp = (P[g0:g1, tri[0]] * P[g0:g1, tri[1]]) * fac
        p0 = 0
        for B in dfx.loop():
            E[p0:p0 + B.shape[0], g0:g1] = B @ pp.T
            p0 += B.shape[0]
    G = (P @ P.T) ** 2
    # unit-diagonal scaling (core points have pair products ~1e6 larger), then a pseudo-inverse
    sc = 1 / np.sqrt(np.diag(G))
    w, U = np.linalg.eigh(G * sc[:, None] * sc)
    keep = w > w[-1] * THC_RCOND
    Ginv = sc[:, None] * ((U[:, keep] / w[keep]) @ U[:, keep].T) * sc
    X = E @ Ginv
    res = np.linalg.norm(E - X @ G) / np.linalg.norm(E)
    log('  THC fit: |E - X G|/|E| = %.1e' % res)
    V = X.T @ X
    log('  THC ng=%d built in %.1f s (kept %d of %d Gram eigenvalues)' % (ng, time.time() - t, keep.sum(), ng))

    def j(D):
        rho = np.einsum('gi,ij,gj->g', P, D, P)
        return (P.T * (V @ rho)) @ P

    def k(D):
        M = V * (P @ D @ P.T)
        return P.T @ M @ P
    return j, k


# ---------------------------------------------------------------- multi-fidelity outer loop emulation
def outer_loop(name, J_s, K_s, D0, Co0, nouter=3, inner_factor=1e-3):
    Co = Co0
    gs = []
    for outer in range(nouter + 1):
        Dref = 2 * Co @ Co.T
        Jx = J_ex(Dref)
        Fx = h1 + Jx - 0.5 * K_ex(Dref)
        # exact orbital gradient at Dref (virtuals: S-orthogonal complement)
        gx = 2 * np.linalg.norm((np.eye(nao) - Dref @ S / 2).T @ Fx @ Co)
        gs.append(gx)
        if outer == nouter:
            break
        Jsr = J_s(Dref) if J_s is not J_ex else None
        W = Fx @ Co - (h1 + Jx - 0.5 * K_s(Dref)) @ Co
        SC = S @ Co
        Delta = W @ SC.T + SC @ W.T - SC @ (Co.T @ W) @ SC.T

        def ffn(D):
            Jd = Jx + (J_s(D) - Jsr) if Jsr is not None else J_ex(D)
            return h1 + Jd - 0.5 * K_s(D) + Delta
        D1, Co, nit, g1 = scf_loop(ffn, Dref, Co, inner_factor * gx)
        log('   %s outer %d: inner %d cycles' % (name, outer, nit))
    log('%-14s outer |g|: %s  ratios: %s' % (name, ' '.join('%.2e' % x for x in gs),
                                             ' '.join('%.0f' % (gs[i] / gs[i + 1]) for i in range(len(gs) - 1))))
    return gs


# common starting point: converged with the cheapest surrogate (SVP-RI K + jfit J), from minao
D0 = scf.hf.init_guess_by_minao(mol)
D0, Co0, nit, g = scf_loop(lambda D: fock(D, J_jf, K_svp), D0, None, 1e-3)
log('D0: surrogate SCF %d cycles, |g_s| %.1e' % (nit, g))

if not os.environ.get("SKIPREF"): outer_loop("svpri+exJ", J_ex, K_svp, D0, Co0)
if not os.environ.get("SKIPREF"): outer_loop("svpri+jfit", J_jf, K_svp, D0, Co0)
for f in ngf:
    ng = int(f * nao)
    Jt, Kt = thc_build(ng)
    if os.environ.get('DIAG'):
        rng = np.random.default_rng(0)
        Cv = scipy.linalg.null_space((S @ Co0).T)          # S-orthogonal virtual complement
        Cv = Cv / np.sqrt(np.einsum('ia,ij,ja->a', Cv, S, Cv))
        A = rng.standard_normal((Cv.shape[1], nocc))
        dD = Cv @ A @ Co0.T
        dD = dD + dD.T
        for nm, X in (('D0', D0), ('dD ov', dD)):
            for op, fx, ft in (('K', K_ex, Kt), ('K svp', K_ex, K_svp), ('J', J_ex, Jt), ('J jfit', J_ex, J_jf)):
                a, b = fx(X), ft(X)
                # orbital-space errors (what drives the SCF): occupied-virtual block
                eo = np.linalg.norm(Cv.T @ (a - b) @ Co0) / np.linalg.norm(Cv.T @ a @ Co0)
                print('   %-6s %-7s rel err  full %.1e   ov %.1e' % (nm, op, np.linalg.norm(a - b) / np.linalg.norm(a), eo))
        continue
    outer_loop('thc%d' % ng, Jt, Kt, D0, Co0)
    outer_loop('thc%d+exJ' % ng, J_ex, Kt, D0, Co0)
log('done')
