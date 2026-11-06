'''Round 9 agent: sparsity of the direct DF-K half transform X_{P mu i} = sum_nu (P|mu nu) L_{nu i} with localized
(Pipek-Mezey) occupied orbitals, BPA HF/def2-TZVP, raw jkfit integrals (what direct_k.py contracts).
Screening at (shell pair MN, orbital i) granularity: drop the block if q_MN * l_{N i} < thr, with
q_MN = max_P |(P|MN)|, l_{N i} = max_{nu in N} |L_{nu i}| (and symmetric). Measures the kept fraction of the
half-transform flops relative to Schwarz-only screening, and the error of K C / E_K / orbital gradient.
Also dumps distance statistics for a geometric extrapolation to Taxol.'''
import os, sys, time
import numpy as np, scipy.linalg
from pyscf import gto, scf, df, lo, lib

xyz = sys.argv[1]
T0 = time.time()
log = lambda *a: print('[%7.1f s]' % (time.time() - T0), *a, flush=True)
mol = gto.M(atom=xyz, basis='def2-tzvp', verbose=0, max_memory=13000)
nao, nocc = mol.nao, mol.nelectron // 2
mf = scf.RHF(mol).density_fit(auxbasis='def2-universal-jkfit'); mf.conv_tol = 1e-9
mf.kernel()
log('HF E %.10f conv %s nao %d nocc %d' % (mf.e_tot, mf.converged, nao, nocc))
Co = mf.mo_coeff[:, :nocc]
L = lo.PM(mol, Co).kernel()
log('PM localized')
auxmol = df.addons.make_auxmol(mol, 'def2-universal-jkfit'); naux = auxmol.nao
Mc = scipy.linalg.cho_factor(auxmol.intor('int2c2e'))
R = df.incore.aux_e2(mol, auxmol, 'int3c2e', aosym='s1').reshape(nao, nao, naux)   # [mu, nu, P]
log('raw int3c2e', R.shape)
ao_loc = mol.ao_loc_nr(); nsh = mol.nbas
sl = [slice(ao_loc[i], ao_loc[i + 1]) for i in range(nsh)]
q = np.array([[np.abs(R[sl[a], sl[b]]).max() for b in range(nsh)] for a in range(nsh)])
lN = np.array([np.abs(L[sl[b]]).max(axis=0) for b in range(nsh)])        # (nsh, nocc)
nf = np.diff(ao_loc)
# centroids and pair centres for the extrapolation model
r = mol.intor('int1e_r')
cen = np.einsum('xmn,mi,ni->ix', r, L, L)
shc = np.array([mol.bas_coord(i) for i in range(nsh)])


def kc(X):
    T = np.einsum('Pmi,mj->Pij', X, L, optimize=True)
    Z = scipy.linalg.cho_solve(Mc, T.reshape(naux, -1)).reshape(T.shape)
    return 2 * np.einsum('Pmi,Pij->mj', X, Z, optimize=True), T


shell_of = np.repeat(np.arange(nsh), nf)


def build_x(thr):
    X = np.empty((naux, nao, nocc))
    kept = tot = 0.0
    for a in range(nsh):
        sigb = q[a] >= 1e-13                                    # (nsh,)
        m = (q[a][:, None] * lN >= thr) & sigb[:, None] if thr > 0 else np.repeat(sigb[:, None], nocc, 1)
        tot += nf[a] * (nf * sigb).sum() * nocc
        kept += nf[a] * (nf[:, None] * m).sum()
        Lm = L * m[shell_of]
        X[:, sl[a]] = np.einsum('mnP,ni->Pmi', R[sl[a]], Lm, optimize=True)
    return X, kept / tot


np.savez('logs/a2_stats.npz', q=q, lN=lN, d=np.linalg.norm(shc[:, None, :] - cen[None, :, :], axis=2), nf=nf,
         shc=shc, cen=cen)
Xe, _ = build_x(0)
KCe, Te = kc(Xe)
P = np.eye(nao) - 0.5 * (2 * Co @ Co.T) @ mol.intor('int1e_ovlp')
EK = -0.5 * np.einsum('mi,mi', L, KCe)
log('E_K (exact DF) %.10f  |P^T KC| %.3e' % (EK, np.linalg.norm(P.T @ KCe)))
# (i, j) pair sparsity of T
tn = np.sqrt(np.einsum('Pij,Pij->ij', Te, Te))
for thr in (1e-9, 1e-8, 1e-7, 1e-6):
    X, frac = build_x(thr)
    K, T = kc(X)
    dE = -0.5 * np.einsum('mi,mi', L, K) - EK
    dg = np.linalg.norm(P.T @ (K - KCe)) * 0.5     # Fock = ... - 1/2 K
    log('thr %.0e  kept flop fraction %.3f  dE_K %.1e  |dg_K| %.1e  rel|dKC| %.1e  (ij) pairs |T|>thr: %.3f' % (
        thr, frac, dE, dg, np.linalg.norm(K - KCe) / np.linalg.norm(KCe), (tn > thr).mean()))
# survival vs distance (centroid - shell centre of b, the transformed index) at thr 1e-8
d = np.linalg.norm(shc[:, None, :] - cen[None, :, :], axis=2)              # (nsh, nocc)
np.savez('logs/a2_stats.npz', q=q, lN=lN, d=d, nf=nf, shc=shc, cen=cen)
log('done')
