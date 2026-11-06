'''Round 9 agent, CPU emulation of the integral-direct checkpoint K with a stored raw half-transform.
Direct DF-K (as gpu4pyscf/df/direct_k.py):  X_P = (P|mu nu) C,  T_P = C^T X_P,  Z = M^-1 T,  K C = 2 sum_P X_P Z_P.
Proposal: keep X (raw, rounded to `store` precision) from the previous checkpoint; at checkpoint k
    dC = C_k(aligned) - C_s,  dX = (P|mu nu) dC   (one integral pass, `inc` precision)
    T  += dC^T X_s + X_s^T dC + dC^T dX           (FP64 accumulation; cross terms see the stored X only times dC)
    X_s = round(X_s + dX),  KC = 2 sum X_s Z
so the M^-1-amplified T never sees the storage rounding of X, and there is no second pass.
Variants: store:inc[:naive]  (naive = T recomputed from the stored X: shows the M^-1 amplification).
usage: python a3_direct_raw.py xyz fp64:fp64,fp32:fp32,fp32:fp16,fp32:fp32:naive'''
import os, sys, time
import numpy as np
import scipy.linalg
from pyscf import gto, scf, df, lib
from pyscf.scf import diis as pdiis

xyz = sys.argv[1]
variants = [v.split(':') for v in (sys.argv[2] if len(sys.argv) > 2 else
                                   'fp64:fp64,fp32:fp32,fp32:fp16,fp16:fp16,fp32:fp32:naive').split(',')]
NOUTER = int(os.environ.get('NOUTER', '3'))
mol = gto.M(atom=xyz, basis='def2-tzvp', verbose=0, max_memory=12000)
nao, nocc = mol.nao, mol.nelectron // 2
T0 = time.time()
log = lambda *a: print('[%7.1f s]' % (time.time() - T0), *a, flush=True)
S = mol.intor('int1e_ovlp'); h1 = scf.hf.get_hcore(mol); enuc = mol.energy_nuc()
auxmol = df.addons.make_auxmol(mol, 'def2-universal-jkfit')
R = df.incore.aux_e2(mol, auxmol, 'int3c2e', aosym='s1').reshape(nao, nao, -1).transpose(2, 0, 1).copy()
naux = len(R); R2 = R.reshape(naux * nao, nao)
M = auxmol.intor('int2c2e'); Mc = scipy.linalg.cho_factor(M)
log('nao %d nocc %d naux %d cond(M) %.1e' % (nao, nocc, naux, np.linalg.cond(M)))
dfx = df.DF(mol, auxbasis='def2-universal-jkfit'); dfx.build()
dsvp = df.DF(mol, auxbasis=os.environ.get('KSURR', 'def2-svp-ri')); dsvp.build()
J_ex = lambda D: df.df_jk.get_jk(dfx, D, hermi=1, with_k=False)[0]
K_s = lambda D: df.df_jk.get_jk(dsvp, D, hermi=1, with_j=False)[1]


def rnd(A, p):
    if p == 'fp64':
        return A
    if p == 'fp32':
        return A.astype(np.float32).astype(np.float64)
    s = np.abs(A).max()
    return (A / s).astype(np.float16).astype(np.float64) * s


def gemm(A, Bm, p):
    if p == 'fp64':
        return A @ Bm
    return (rnd(A, p).astype(np.float32) @ rnd(Bm, p).astype(np.float32)).astype(np.float64)


def x_of(C, p='fp64'):
    return gemm(R2, C, p).reshape(naux, nao, -1)


def t_of(X, C):
    return np.einsum('Pmi,mj->Pij', X, C)


def kc_of(X, T):
    Z = scipy.linalg.cho_solve(Mc, T.reshape(naux, -1)).reshape(T.shape)
    return 2 * np.einsum('Pmi,Pij->mj', X, Z)


def pg(F, Co, D):
    return 2 * np.linalg.norm((np.eye(nao) - D @ S / 2).T @ F @ Co)


def scf_loop(fockfn, D, Co, tol, maxc=80):
    d = pdiis.CDIIS(); d.space = 8
    for it in range(maxc):
        F = fockfn(D)
        if Co is not None and pg(F, Co, D) < tol:
            break
        e, C = scipy.linalg.eigh(d.update(S, D, F), S)
        Co = C[:, :nocc]; D = 2 * Co @ Co.T
    return D, Co


mf = scf.RHF(mol).density_fit(); mf.with_df = dfx; mf.conv_tol = 1e-12; mf.conv_tol_grad = 1e-8
Estar = mf.kernel()
D0, Co0 = scf_loop(lambda D: h1 + J_ex(D) - 0.5 * K_s(D), scf.hf.init_guess_by_minao(mol), None, 1e-3)
D0, Co0 = scf_loop(lambda D: h1 + J_ex(D) - 0.5 * K_s(D), D0, Co0, 1e-3)
log('E* %.12f' % Estar)

for v in variants:
    store, inc = v[0], v[1]
    naive = len(v) > 2
    log('variant store=%s inc=%s %s' % (store, inc, 'naive (T from stored X)' if naive else ''))
    Co = Co0.copy(); Xs = None
    for k in range(NOUTER + 1):
        D = 2 * Co @ Co.T
        if Xs is None:
            X = x_of(Co); T = t_of(X, Co); Xs = rnd(X, store); Cs = Co.copy()
            if naive:
                T = t_of(Xs, Co)
        else:
            U, _, Vt = np.linalg.svd(Cs.T @ S @ Co); Co = Co @ (U @ Vt).T
            dC = Co - Cs
            dX = x_of(dC, inc)
            if naive:
                Xs = rnd(Xs + dX, store); T = t_of(Xs, Co)
            else:
                T = T + t_of(Xs, dC) + t_of(Xs, dC).transpose(0, 2, 1) + t_of(dX, dC)
                Xs = rnd(Xs + dX, store)
            Cs = Co.copy()
        WK = kc_of(Xs, T)
        X = x_of(Co); WKt = kc_of(X, t_of(X, Co))
        J = J_ex(D); Fh = h1 + J
        Wt = Fh @ Co - 0.5 * WK
        P = (np.eye(nao) - D @ S / 2).T
        gu, gt = 2 * np.linalg.norm(P @ Wt), 2 * np.linalg.norm(P @ (Fh @ Co - 0.5 * WKt))
        E0 = np.einsum('ij,ji', D, h1) + 0.5 * np.einsum('ij,ji', D, J) + enuc
        Eu, Et = E0 - 0.5 * np.einsum('mi,mi', Co, WK), E0 - 0.5 * np.einsum('mi,mi', Co, WKt)
        log('  k=%d |g| used %.3e true %.3e  |g_used-g_true| %.1e  E_used-E_true %.1e  E_true-E* %.1e  rel|dKC| %.1e' % (
            k, gu, gt, abs(gu - gt), Eu - Et, Et - Estar, np.linalg.norm(WK - WKt) / np.linalg.norm(WKt)))
        if k == NOUTER:
            break
        W = Wt - (Fh - 0.5 * K_s(D)) @ Co
        SC = S @ Co
        Delta = W @ SC.T + SC @ W.T - SC @ (Co.T @ W) @ SC.T
        _, Co = scf_loop(lambda D: h1 + J_ex(D) - 0.5 * K_s(D) + Delta, D, Co, 1e-3 * gu)
log('done')
