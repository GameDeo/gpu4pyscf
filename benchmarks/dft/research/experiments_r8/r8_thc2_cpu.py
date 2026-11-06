'''Round 8b, CPU: is THC/ISDF viable once the hard parts are removed?
Variables tested on the occupied-virtual response error of K (what sets the multi-fidelity contraction rate):
  operator:  full-range 1/r  vs  long-range erf(w r)/r (w = 0.3, the 85 % part of wB97M-V exchange)
  points:    weighted K-means  vs  randomized QR-pivoted (Lu-Ying sketch of the pair products)
  rank:      ng = f * nao
Reference: DF-K with def2-universal-jkfit for the same operator.  Baseline surrogate: DF-K with def2-svp-ri.
usage: python r8_thc2_cpu.py xyz [factors]'''
import os, sys, time
import numpy as np
import scipy.linalg
from pyscf import gto, scf, df, lib, dft
from scipy.spatial import cKDTree

xyz = sys.argv[1]
fac = [float(x) for x in (sys.argv[2] if len(sys.argv) > 2 else '2,4,8').split(',')]
T0 = time.time()


def log(*a):
    print('[%6.0f s]' % (time.time() - T0), *a, flush=True)


def build(omega):
    mol = gto.M(atom=xyz, basis='def2-tzvp', verbose=0, max_memory=12000)
    mol.omega = omega
    return mol


mol = build(0)
nao, nocc = mol.nao, mol.nelectron // 2
mf = scf.RHF(mol).density_fit(auxbasis='def2-universal-jkfit')
mf.kernel()
S = mol.intor('int1e_ovlp')
Co = mf.mo_coeff[:, :nocc]
Cv = mf.mo_coeff[:, nocc:]
rng = np.random.default_rng(0)
dD = Cv @ rng.standard_normal((Cv.shape[1], nocc)) @ Co.T
dD = dD + dD.T
log('nao', nao, 'nocc', nocc)

# candidate grid and AO values (shared by all operators)
g = dft.gen_grid.Grids(mol)
g.atom_grid = (40, 194)
g.prune = dft.gen_grid.nwchem_prune
g.build()
pts = g.coords
phi = dft.numint.eval_ao(mol, pts)
log('candidate points', len(pts))


def kmeans_pts(ng):
    w = np.einsum('gi,gi->g', phi, phi)
    r = np.random.default_rng(1)
    cen = pts[r.choice(len(pts), ng, replace=False, p=w / w.sum())]
    for _ in range(20):
        lab = cKDTree(cen).query(pts)[1]
        ws = np.bincount(lab, w, ng)
        new = np.stack([np.bincount(lab, w * pts[:, x], ng) for x in range(3)], 1)
        ok = ws > 0
        cen[ok] = new[ok] / ws[ok, None]
    return np.unique(cKDTree(pts).query(cen)[1])


def qrcp_pts(ng):
    # randomized sketch of the pair-product matrix: Z_g,(ab) = (phi G1)_ga (phi G2)_gb, r^2 >= 1.2 ng columns
    r = int(np.ceil(np.sqrt(1.2 * ng)))
    r_ = np.random.default_rng(2)
    A = phi @ r_.standard_normal((nao, r))
    B = phi @ r_.standard_normal((nao, r))
    Z = (A[:, :, None] * B[:, None, :]).reshape(len(pts), -1)
    _, _, piv = scipy.linalg.qr(Z.T, mode='economic', pivoting=True)
    return np.sort(piv[:ng])


def thc_k(dfo, piv):
    P = phi[piv]
    ng = len(piv)
    tri = np.tril_indices(nao)
    f2 = np.where(tri[0] == tri[1], 1.0, 2.0)
    naux = dfo.get_naoaux()
    E = np.zeros((naux, ng))
    for g0, g1 in lib.prange(0, ng, 256):
        pp = (P[g0:g1, tri[0]] * P[g0:g1, tri[1]]) * f2
        p0 = 0
        for Bk in dfo.loop():
            E[p0:p0 + Bk.shape[0], g0:g1] = Bk @ pp.T
            p0 += Bk.shape[0]
    G = (P @ P.T) ** 2
    sc = 1 / np.sqrt(np.diag(G))
    w, U = np.linalg.eigh(G * sc[:, None] * sc)
    keep = w > w[-1] * 1e-14
    X = E @ (sc[:, None] * ((U[:, keep] / w[keep]) @ U[:, keep].T) * sc)
    V = X.T @ X
    return lambda D: P.T @ (V * (P @ D @ P.T)) @ P


def ov_err(Kx, Kt, X):
    a, b = Kx(X), Kt(X)
    return np.linalg.norm(Cv.T @ (a - b) @ Co) / np.linalg.norm(Cv.T @ a @ Co)


# spatial extent of each AO: <(r - R_A)^2>, from <r^2> - |<r>|^2
r2 = np.diag(mol.intor('int1e_r2'))
rv = np.array([np.diag(x) for x in mol.intor('int1e_r')])
ext = np.sqrt(np.maximum(r2 - (rv ** 2).sum(0), 0))


def split_k(dfo, piv, rcut):
    """core-split THC: pairs with a tight AO (extent < rcut bohr) exact (stored, local); smooth-smooth pairs by THC
    fitted on the smooth pair products only"""
    tight = ext < rcut
    Bfull = lib.unpack_tril(np.vstack(list(dfo.loop())))          # (naux, nao, nao)
    mask_t = tight[:, None] | tight[None, :]
    Bt = Bfull * mask_t
    sm = ~tight
    P = phi[piv][:, sm]
    Bss = Bfull[:, sm][:, :, sm]
    E = np.einsum('Lmn,gm,gn->Lg', Bss, P, P, optimize=True)
    G = (P @ P.T) ** 2
    sc = 1 / np.sqrt(np.diag(G))
    w, U = np.linalg.eigh(G * sc[:, None] * sc)
    keep = w > w[-1] * 1e-14
    X = E @ (sc[:, None] * ((U[:, keep] / w[keep]) @ U[:, keep].T) * sc)
    # B~ = Bt + embed(sum_g X_Lg phi_m(g) phi_n(g)) on smooth pairs; K = sum_L B~ D B~
    Pfull = np.zeros((len(piv), nao)); Pfull[:, sm] = P
    nst = int(mask_t.sum())
    tri_frac = nst / nao ** 2

    def k(D):
        out = np.zeros((nao, nao))
        for L0, L1 in lib.prange(0, X.shape[0], 64):
            Bt_L = Bt[L0:L1]
            Bs_L = np.einsum('Lg,gm,gn->Lmn', X[L0:L1], Pfull, Pfull, optimize=True)
            BL = Bt_L + Bs_L
            out += np.einsum('Lmk,kl,Lln->mn', BL, D, BL, optimize=True)
        return out
    return k, tight.sum(), tri_frac


for omega, label in ((0, 'full-range'), (0.3, 'LR erf(0.3)')):
    if os.environ.get('SPLIT'):
        m = build(omega)
        dx = df.DF(m, auxbasis='def2-universal-jkfit'); dx.build()
        Kx = lambda D, d=dx: df.df_jk.get_jk(d, D, hermi=1, with_j=False)[1]
        for rcut in [float(x) for x in os.environ['SPLIT'].split(',')]:
            for f in fac:
                piv = qrcp_pts(int(f * nao)) if os.environ.get('QR') else kmeans_pts(int(f * nao))
                Kt, ntight, frac_t = split_k(dx, piv, rcut)
                log('%-12s split THC rcut %.1f: %d tight AOs, exact pairs %.0f %%, ng %d (%.0fx nao)  ov err %.1e' % (
                    label, rcut, ntight, 100 * frac_t, len(piv), f, ov_err(Kx, Kt, dD)))
        continue
    m = build(omega)
    dx = df.DF(m, auxbasis='def2-universal-jkfit'); dx.build()
    ds = df.DF(m, auxbasis='def2-svp-ri'); ds.build()
    Kx = lambda D, d=dx: df.df_jk.get_jk(d, D, hermi=1, with_j=False)[1]
    Ks = lambda D, d=ds: df.df_jk.get_jk(d, D, hermi=1, with_j=False)[1]
    log('%-12s SVP-RI DF K (baseline)          ov err %.1e' % (label, ov_err(Kx, Ks, dD)))
    for f in fac:
        ng = int(f * nao)
        for sel, fn in (('kmeans', kmeans_pts), ('qrcp', qrcp_pts)):
            t = time.time()
            piv = fn(ng)
            Kt = thc_k(dx, piv)
            log('%-12s THC %-6s ng=%5d (%.0fx nao)  ov err %.1e   (%.0f s)' % (
                label, sel, len(piv), f, ov_err(Kx, Kt, dD), time.time() - t))
log('done')
