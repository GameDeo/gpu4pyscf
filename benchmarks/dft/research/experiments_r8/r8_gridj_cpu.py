'''Round 8, CPU only: memory-light surrogates that keep the operator structure (no stored npairs x naux tensor).

1. Grid-fitted J increment ("grid-DF J"): on a coarse DFT grid,
       r_P = sum_g w_g drho(g) V_P(g),  c = M^-1 r,  v(g) = sum_P c_P V_P(g),  J_mn = sum_g w_g phi_m phi_n v(g)
   with V_P(g) the potential of aux function P at grid point g (analytic, computed on the fly on a GPU).
   Memory O(naux) + one grid batch.  It is of the form Phi^T diag Phi, so it is small along near-null AO combinations.
   On the GPU it would add w v to the coarse XC potential and share the XC increment GEMMs.
2. Natural-auxiliary-function (NAF) compression of the K surrogate tensor: B -> Q^T B, Q = leading eigenvectors of B B^T.
   A left (aux-side) projection, so it is also safe along near-null AO combinations.

Reports operator errors on the occupied-virtual block, then the multi-fidelity outer-loop contraction (same emulation
as r8_thc_cpu.py).
usage: python r8_gridj_cpu.py [xyz] [grid radial,angular]'''
import os, sys, time
import numpy as np
import scipy.linalg
from pyscf import gto, scf, df, lib, dft
from pyscf.scf import diis as pdiis

here = os.path.dirname(os.path.abspath(__file__))
xyz = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, '..', '..', '..', 'molecules', 'organic',
                                                         '033_Bisphenol_A.xyz')
grid = tuple(int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else '50,194').split(','))
mol = gto.M(atom=xyz, basis='def2-tzvp', verbose=0, max_memory=12000)
nao, nocc = mol.nao, mol.nelectron // 2
T0 = time.time()


def log(*a):
    print('[%7.1f s]' % (time.time() - T0), *a, flush=True)


S = mol.intor('int1e_ovlp')
h1 = scf.hf.get_hcore(mol)
log('nao', nao, 'nocc', nocc, 'lambda_min(S) %.1e' % np.linalg.eigvalsh(S)[0])


def dfobj(aux):
    d = df.DF(mol, auxbasis=aux)
    d.build()
    return d


def df_jk(d):
    return (lambda D: df.df_jk.get_jk(d, D, hermi=1, with_k=False)[0],
            lambda D: df.df_jk.get_jk(d, D, hermi=1, with_j=False)[1])


dfx = dfobj('def2-universal-jkfit')
J_ex, K_ex = df_jk(dfx)
J_jf, _ = df_jk(dfobj('def2-universal-jfit'))
dsvp = dfobj('def2-svp-ri')
_, K_svp = df_jk(dsvp)


# ---------------------------------------------------------------- grid-DF J
def gridj_build(auxbasis, atom_grid):
    t = time.time()
    g = dft.gen_grid.Grids(mol)
    g.atom_grid = atom_grid
    g.prune = dft.gen_grid.nwchem_prune
    g.build()
    phi = dft.numint.eval_ao(mol, g.coords)
    w = g.weights
    auxmol = df.addons.make_auxmol(mol, auxbasis)
    fake = gto.fakemol_for_charges(g.coords)
    V = gto.mole.intor_cross('int2c2e', auxmol, fake)          # (naux, Ng): potential of P at the points
    M = auxmol.intor('int2c2e')
    Mc = scipy.linalg.cho_factor(M)
    log('  grid-J %s %s: %d points, naux %d, %.1f s' % (auxbasis, atom_grid, len(w), auxmol.nao, time.time() - t))

    def j(D):
        rho = np.einsum('gi,ij,gj->g', phi, D, phi)
        c = scipy.linalg.cho_solve(Mc, V @ (w * rho))
        v = V.T @ c
        return (phi.T * (w * v)) @ phi
    return j


# ---------------------------------------------------------------- seminumerical (COSX) K surrogate
def sgx_k(level):
    from pyscf import sgx
    from pyscf.sgx import sgx_jk
    m = sgx.sgx_fit(scf.RHF(mol), pjs=False)
    m.with_df.grids_level_f = level
    m.with_df.grids_level_i = level
    m.with_df.build()
    wdf = m.with_df

    def k(D):
        Kn = sgx_jk.get_jk_favork(wdf, D, hermi=1, with_j=False)[1] if hasattr(sgx_jk, 'get_jk_favork') \
            else sgx_jk.get_jk(wdf, D, hermi=1, with_j=False)[1]
        return 0.5 * (Kn + Kn.T)
    return k


# ---------------------------------------------------------------- NAF-compressed K surrogate
def naf_k(d, eps):
    B = np.vstack(list(d.loop()))              # (naux, npair_tril)
    tri = np.tril_indices(nao)
    fac = np.where(tri[0] == tri[1], 1.0, np.sqrt(2.0))
    W = (B * fac) @ (B * fac).T                # Gram over the full (symmetric) pair space
    e, U = np.linalg.eigh(W)
    keep = e > eps * e[-1]
    Bq = U[:, keep].T @ B
    r = keep.sum()

    def k(D):
        out = np.zeros((nao, nao))
        for p0, p1 in lib.prange(0, r, 128):
            b = lib.unpack_tril(Bq[p0:p1])
            t = np.einsum('pij,jk->pik', b, D)
            out += np.einsum('pik,plk->il', t, b)
        return out
    log('  NAF eps %.0e: %d of %d aux kept' % (eps, r, B.shape[0]))
    return k


def solve(F):
    return scipy.linalg.eigh(F, S)


def dens(C):
    Co = C[:, :nocc]
    return 2 * Co @ Co.T, Co


def scf_loop(fockfn, D, Co, tol, maxc=80):
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


def outer_loop(name, J_s, K_s, Co, nouter=3, inner_factor=float(os.environ.get('INNER', '1e-3'))):
    gs = []
    for outer in range(nouter + 1):
        Dref = 2 * Co @ Co.T
        Jx = J_ex(Dref)
        Fx = h1 + Jx - 0.5 * K_ex(Dref)
        gx = 2 * np.linalg.norm((np.eye(nao) - Dref @ S / 2).T @ Fx @ Co)
        gs.append(gx)
        if outer == nouter:
            break
        if hasattr(K_s, 'set_ref'):
            K_s.set_ref(Dref)
        Jsr = J_s(Dref) if J_s is not J_ex else None
        W = Fx @ Co - (h1 + Jx - 0.5 * K_s(Dref)) @ Co
        SC = S @ Co
        Delta = W @ SC.T + SC @ W.T - SC @ (Co.T @ W) @ SC.T

        def ffn(D):
            Jd = Jx + (J_s(D) - Jsr) if Jsr is not None else J_ex(D)
            return h1 + Jd - 0.5 * K_s(D) + Delta
        _, Co, nit, _ = scf_loop(ffn, Dref, Co, inner_factor * gx)
    log('%-22s outer |g|: %s  ratios: %s' % (name, ' '.join('%.2e' % x for x in gs),
                                             ' '.join('%.0f' % (gs[i] / gs[i + 1]) for i in range(len(gs) - 1))))


D0 = scf.hf.init_guess_by_minao(mol)
D0, Co0, nit, g = scf_loop(lambda D: h1 + J_jf(D) - 0.5 * K_svp(D), D0, None, 1e-3)
log('D0: surrogate SCF %d cycles' % nit)

Cv = scipy.linalg.null_space((S @ Co0).T)
Cv = Cv / np.sqrt(np.einsum('ia,ij,ja->a', Cv, S, Cv))
rng = np.random.default_rng(0)
dD = Cv @ rng.standard_normal((Cv.shape[1], nocc)) @ Co0.T
dD = dD + dD.T


def err(fx, ft, X):
    a, b = fx(X), ft(X)
    return np.linalg.norm(Cv.T @ (a - b) @ Co0) / np.linalg.norm(Cv.T @ a @ Co0)


ISDF_NV = float(os.environ.get('ISDF_NV', '0'))
ISDF_FIX = os.environ.get('ISDF_FIX') == '1'
ISDF_TIK = float(os.environ.get('ISDF_TIK', '0'))
# ---------------------------------------------------------------- ISDF of occupied-pair products + ACE
def isdf_ace_k(ng, Co_sel, auxbasis='def2-universal-jkfit', atom_grid=None, weighted=True):
    """K(D) for D = 2 C C^T, memory-light:
      psi_i(r) psi_j(r) ~= sum_g zeta_g(r) psi_i(r_g) psi_j(r_g)        (ISDF, points fixed, zeta from current orbitals)
      V_g(r) = Coulomb potential of zeta_g via the grid-fitted aux basis (same machinery as grid-DF J)
      (K psi_j)(r) = sum_g V_g(r) Q(r, g) psi_j(r_g),  Q = Psi Psi_g^T     (all dense GEMMs)
      W = K(D) C = 2 Phi^T diag(w) (K psi)  ->  ACE  K_s = W (C^T W)^-1 W^T
    Points: QR-pivoted on the (sqrt(w)-weighted) occupied pair products of Co_sel, then kept fixed."""
    t = time.time()
    g = dft.gen_grid.Grids(mol)
    g.atom_grid = atom_grid or grid
    g.prune = dft.gen_grid.nwchem_prune
    g.build()
    phi = dft.numint.eval_ao(mol, g.coords)
    w = g.weights
    auxmol = df.addons.make_auxmol(mol, auxbasis)
    V = gto.mole.intor_cross('int2c2e', auxmol, gto.fakemol_for_charges(g.coords))     # (naux, Nr)
    Mc = scipy.linalg.cho_factor(auxmol.intor('int2c2e'))
    Psi = phi @ Co_sel
    iu = np.triu_indices(nocc)
    Z = Psi[:, iu[0]] * Psi[:, iu[1]]
    if ISDF_NV:
        # response-adapted points: also occupied x low-virtual products psi_i chi_a (what the inner-loop
        # response is made of), virtuals from the surrogate Fock at D0
        Dsel = 2 * Co_sel @ Co_sel.T
        _, Cf = scipy.linalg.eigh(h1 + J_jf(Dsel) - 0.5 * K_svp(Dsel), S)
        Chi = phi @ Cf[:, nocc:nocc + int(ISDF_NV * nocc)]
        Z = np.hstack([Z, (Psi[:, :, None] * Chi[:, None, :]).reshape(len(w), -1)])
    if weighted:
        Z = Z * np.sqrt(w)[:, None]
    _, _, piv = scipy.linalg.qr(Z.T, mode='economic', pivoting=True)
    piv = np.sort(piv[:ng])
    log('  ISDF-ACE ng=%d (%.1f x nocc) on %d grid points, setup %.1f s' % (ng, ng / nocc, len(w), time.time() - t))

    def occ(D):
        # S-orthonormal C with D = 2 C C^T
        e, U = scipy.linalg.eigh(S @ D @ S, S)
        return U[:, -nocc:] * np.sqrt(np.maximum(e[-nocc:], 0) / 2)

    ref = {}

    def set_ref(D):
        ref.clear()
        if ISDF_FIX:
            ref['Vg'] = _vg(occ(D))

    def _vg(C):
        Psi = phi @ C
        Pg = Psi[piv]
        G = (Pg @ Pg.T) ** 2
        sc = 1 / np.sqrt(np.diag(G))
        ev, U = np.linalg.eigh(G * sc[:, None] * sc)
        keep = ev > ev[-1] * 1e-12
        Gi = sc[:, None] * ((U[:, keep] / ev[keep]) @ U[:, keep].T) * sc
        zeta = ((Psi @ Pg.T) ** 2) @ Gi
        return V.T @ scipy.linalg.cho_solve(Mc, V @ (w[:, None] * zeta))

    def k(D):
        C = occ(D)
        if 'Vg' in ref:
            # frozen zeta and potentials: per call only (Nr x ng x nocc) GEMMs, smooth in the orbitals
            Psi = phi @ C
            Pg = Psi[piv]
            Kpsi = (ref['Vg'] * (Psi @ Pg.T)) @ Pg
            W = 2 * phi.T @ (w[:, None] * Kpsi)
            M = C.T @ W
            M = 0.5 * (M + M.T)
            return W @ np.linalg.solve(M, W.T)
        Psi = phi @ C
        Pg = Psi[piv]
        G = (Pg @ Pg.T) ** 2
        sc = 1 / np.sqrt(np.diag(G))
        ev, U = np.linalg.eigh(G * sc[:, None] * sc)
        if ISDF_TIK:
            # Tikhonov: smooth in the orbitals (a hard eigenvalue cut changes rank as the orbitals move)
            lam = ISDF_TIK * ev[-1]
            Gi = sc[:, None] * ((U * (ev / (ev ** 2 + lam ** 2))) @ U.T) * sc
        else:
            keep = ev > ev[-1] * 1e-12
            Gi = sc[:, None] * ((U[:, keep] / ev[keep]) @ U[:, keep].T) * sc
        Q = Psi @ Pg.T                                   # (Nr, ng) = sum_i psi_i(r) psi_i(r_g)
        zeta = (Q ** 2) @ Gi                             # (Nr, ng)
        c = scipy.linalg.cho_solve(Mc, V @ (w[:, None] * zeta))
        Vg = V.T @ c                                     # (Nr, ng)
        Kpsi = (Vg * Q) @ Pg                             # (Nr, nocc)
        W = 2 * phi.T @ (w[:, None] * Kpsi)
        M = C.T @ W
        M = 0.5 * (M + M.T)
        return W @ np.linalg.solve(M, W.T)
    k.set_ref = set_ref
    return k


if os.environ.get('ISDF'):
    gj = gridj_build('def2-universal-jkfit', grid)
    for f in [float(x) for x in os.environ['ISDF'].split(',')]:
        Ki = isdf_ace_k(int(f * nocc), Co0)
        # ACE is exact only on the occupied space of D: compare W = K(D) C (ov block) at D0, and the response
        e0 = err(K_ex, Ki, D0)
        log('  K ISDF-ACE %.0fx nocc  ov rel err at D0 %.1e' % (f, e0))
        outer_loop('isdf%.0fx K + grid J' % f, gj, Ki, Co0)
    sys.exit()

if os.environ.get('SGX'):
    gj = gridj_build('def2-universal-jkfit', grid)
    for lv in [int(x) for x in os.environ['SGX'].split(',')]:
        Ks = sgx_k(lv)
        log('  K sgx level %d  ov rel err: dD %.1e  D0 %.1e' % (lv, err(K_ex, Ks, dD), err(K_ex, Ks, D0)))
        outer_loop('sgx%d K + grid J' % lv, gj, Ks, Co0)
    sys.exit()
cands_j = [('jfit DF', J_jf)]
for aux in ('def2-universal-jkfit', 'def2-universal-jfit'):
    for ag in (grid, (75, 302)):
        cands_j.append(('grid %s %s' % (aux.split('-')[-1], ag), gridj_build(aux, ag)))
for nm, Jt in cands_j:
    log('  J %-28s ov rel err: dD %.1e  D0 %.1e' % (nm, err(J_ex, Jt, dD), err(J_ex, Jt, D0)))
cands_k = [('svp-ri', K_svp)] + [('svp-ri NAF %.0e' % e, naf_k(dsvp, e)) for e in (1e-6, 1e-5, 1e-4)]
for nm, Kt in cands_k:
    log('  K %-28s ov rel err: dD %.1e  D0 %.1e' % (nm, err(K_ex, Kt, dD), err(K_ex, Kt, D0)))

outer_loop('svpri + exact J', J_ex, K_svp, Co0)
outer_loop('svpri + jfit J', J_jf, K_svp, Co0)
for nm, Jt in cands_j[1:]:
    outer_loop('svpri + ' + nm, Jt, K_svp, Co0)
for nm, Kt in cands_k[1:]:
    outer_loop(nm + ' + exact J', J_ex, Kt, Co0)
log('done')
