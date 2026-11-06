'''Round 9 agent, CPU emulation (HF, DF): incremental exact K at the multi-fidelity checkpoints.

Checkpoint k >= 1 does not recompute X = B C from scratch; it keeps X_store = B C_store from the previous checkpoint
and adds dX = B dC_r, where dC = C_k (Procrustes-aligned to C_store) - C_store, truncated to rank r in the S metric
(singular values > tau), and evaluated in a lower precision (fp64 | fp32 | fp16 inputs with fp32 accumulation).
    K(D_k) C_k ~= 2 sum_P X_P (X_P^T C_k)
This is what a GPU checkpoint would do with X held in (host) memory: no second integral/half-transform pass, and the
increment GEMM has r << nocc columns at fewer CRT moduli.  Also records the model-energy prediction of the next
checkpoint (to judge whether the certifying evaluation can be skipped).
usage: python a1_incremental_x.py [xyz] [variants: tau:prec,...]'''
import os, sys, time
import numpy as np
import scipy.linalg
from pyscf import gto, scf, df, lib
from pyscf.scf import diis as pdiis

here = os.path.dirname(os.path.abspath(__file__))
xyz = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, 'phenol.xyz')
variants = [(float(v.split(':')[0]), v.split(':')[1]) for v in
            (sys.argv[2] if len(sys.argv) > 2 else '0:fp64,1e-10:fp64,1e-8:fp64,1e-6:fp64,0:fp32,0:fp16').split(',')]
NOUTER = int(os.environ.get('NOUTER', '3'))
INNER = float(os.environ.get('INNER', '1e-3'))
mol = gto.M(atom=xyz, basis='def2-tzvp', verbose=0, max_memory=12000)
nao, nocc = mol.nao, mol.nelectron // 2
T0 = time.time()


def log(*a):
    print('[%7.1f s]' % (time.time() - T0), *a, flush=True)


S = mol.intor('int1e_ovlp')
Lc = np.linalg.cholesky(S)
h1 = scf.hf.get_hcore(mol)
enuc = mol.energy_nuc()
dfx = df.DF(mol, auxbasis='def2-universal-jkfit'); dfx.build()
dsvp = df.DF(mol, auxbasis=os.environ.get('KSURR', 'def2-svp-ri')); dsvp.build()
B = np.vstack([lib.unpack_tril(b) for b in dfx.loop()])      # (naux, nao, nao) Cholesky-fitted
naux = len(B)
B2 = B.reshape(naux * nao, nao)
log('nao', nao, 'nocc', nocc, 'naux', naux)


def J_ex(D):
    return df.df_jk.get_jk(dfx, D, hermi=1, with_k=False)[0]


def K_s(D):
    return df.df_jk.get_jk(dsvp, D, hermi=1, with_j=False)[1]


def x_of(C, prec='fp64'):
    if prec == 'fp64':
        return (B2 @ C).reshape(naux, nao, -1)
    if prec == 'fp32':
        return (B2.astype(np.float32) @ C.astype(np.float32)).astype(np.float64).reshape(naux, nao, -1)
    if prec == 'fp16':     # fp16 inputs (per-tensor scaled), fp32 accumulation: tensor-core HMMA
        sb, sc = np.abs(B2).max(), np.abs(C).max()
        b = (B2 / sb).astype(np.float16).astype(np.float32)
        c = (C / sc).astype(np.float16).astype(np.float32)
        return ((b @ c).astype(np.float64) * (sb * sc)).reshape(naux, nao, -1)
    raise ValueError(prec)


def kc_from_x(X, C):
    T = np.einsum('Pmi,mj->Pij', X, C)
    return 2 * np.einsum('Pmi,Pij->mj', X, T)


def energy(D, J, Co, WK):
    return np.einsum('ij,ji', D, h1) + 0.5 * np.einsum('ij,ji', D, J) - 0.5 * np.einsum('mi,mi', Co, WK) + enuc


def esurr(D):
    return np.einsum('ij,ji', D, h1) + 0.5 * np.einsum('ij,ji', D, J_ex(D)) - 0.25 * np.einsum('ij,ji', D, K_s(D))


def pgrad(F, Co, D):
    return 2 * np.linalg.norm((np.eye(nao) - D @ S / 2).T @ F @ Co)


def solve(F):
    return scipy.linalg.eigh(F, S)


def scf_loop(fockfn, D, Co, tol, maxc=80):
    d = pdiis.CDIIS(); d.space = 8
    g = np.inf
    for it in range(maxc):
        F = fockfn(D)
        if Co is not None:
            g = pgrad(F, Co, D)
            if g < tol:
                break
        e, C = solve(d.update(S, D, F))
        Co = C[:, :nocc]; D = 2 * Co @ Co.T
    return D, Co, it + 1, g


# exact reference
mf = scf.RHF(mol).density_fit(); mf.with_df = dfx; mf.conv_tol = 1e-12; mf.conv_tol_grad = 1e-8
Estar = mf.kernel()
log('E* (exact DF-HF) %.12f converged %s' % (Estar, mf.converged))

D0 = scf.hf.init_guess_by_minao(mol)
D0, Co0, nit, _ = scf_loop(lambda D: h1 + J_ex(D) - 0.5 * K_s(D), D0, None, 1e-3)
D0, Co0, nit, g = scf_loop(lambda D: h1 + J_ex(D) - 0.5 * K_s(D), D0, Co0, 1e-3)
log('D0: surrogate SCF, |g_s| %.1e' % g)


def outer(tau, prec):
    Co = Co0.copy()
    Xs = Cs = None
    rows = []
    Epred = None
    for k in range(NOUTER + 1):
        D = 2 * Co @ Co.T
        info = ''
        if Xs is None or tau < 0:
            Xs = x_of(Co); Cs = Co.copy(); r = nocc
        else:
            U, _, Vt = np.linalg.svd(Cs.T @ S @ Co)
            Co = Co @ (U @ Vt).T                       # aligned: Co ~ Cs, same D
            dC = Co - Cs
            u, s, vt = np.linalg.svd(Lc.T @ dC, full_matrices=False)
            r = int((s > tau).sum()) if tau > 0 else len(s)
            dCr = scipy.linalg.solve_triangular(Lc.T, u[:, :r] * s[:r], lower=False) @ vt[:r]
            if r:
                Xs = Xs + x_of(dCr, prec)
                Cs = Cs + dCr
            info = '|dC|_S %.1e  sv>1e-6:%d >1e-8:%d >1e-10:%d of %d  kept %d' % (
                s[0], (s > 1e-6).sum(), (s > 1e-8).sum(), (s > 1e-10).sum(), len(s), r)
        J = J_ex(D)
        WK = kc_from_x(Xs, Co)
        WKt = kc_from_x(x_of(Co), Co)
        Fh = h1 + J
        Wt = Fh @ Co - 0.5 * WK
        g_used = 2 * np.linalg.norm((np.eye(nao) - D @ S / 2).T @ Wt)
        g_true = 2 * np.linalg.norm((np.eye(nao) - D @ S / 2).T @ (Fh @ Co - 0.5 * WKt))
        E_used, E_true = energy(D, J, Co, WK), energy(D, J, Co, WKt)
        rows.append((k, g_used, g_true, E_used - Estar, E_true - Estar, (Epred - E_true) if Epred is not None else np.nan, info))
        log('  k=%d |g| used %.2e true %.2e  E-E*: used %.1e true %.1e  model-pred err %.1e  %s' % rows[-1])
        if k == NOUTER:
            break
        Ksr = K_s(D)
        W = Wt - (Fh - 0.5 * Ksr) @ Co
        SC = S @ Co
        Delta = W @ SC.T + SC @ W.T - SC @ (Co.T @ W) @ SC.T
        Dref, Eref, Esref = D, E_used, esurr(D)
        D1, Co, nit, _ = scf_loop(lambda D: h1 + J_ex(D) - 0.5 * K_s(D) + Delta, D, Co, INNER * g_used)
        Epred = Eref + esurr(D1) - Esref + np.einsum('ij,ji', Delta, D1 - Dref)
    return rows


for tau, prec in variants:
    log('variant tau=%g prec=%s' % (tau, prec))
    outer(tau, prec)
log('done')
