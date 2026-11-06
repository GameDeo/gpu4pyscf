'''Round 9 agent: geometric extrapolation of the localized half-transform sparsity (a2) to Taxol.
Model: block (shell b, orbital i) of X is kept with probability p(d_bi) measured on BPA (d = distance between the
shell centre and the PM centroid of orbital i, TZVP orbitals), weighted by W_b = sum_a nf_a nf_b over significant
pairs (a, b).  Taxol centroids come from PM-localized HF/STO-3G orbitals (centroids are basis-insensitive); the
model is validated by applying it to BPA with its own STO-3G centroids.  Pair significance: max|S_ab| >= s_cut,
s_cut calibrated so that BPA's TZVP pair weight matches the q_MN > 1e-13 count of a2.'''
import sys, numpy as np
from pyscf import gto, scf, lo

st = np.load('logs/a2_stats.npz')
q, lN, d, nf = st['q'], st['lN'], st['d'], st['nf']
bpa = sys.argv[1]; tax = sys.argv[2]
thrs = [1e-9, 1e-8, 1e-7]
bins = np.linspace(0, 30, 61)
sig = q >= 1e-13
Wb_bpa = (nf[:, None] * nf[None, :] * sig).sum(0)          # weight of transformed shell b
prof = {}
for thr in thrs:
    keep = (q.max(0)[:, None] * lN >= thr)                  # upper bound proxy per (b, i): any a
    # exact kept weight per (b, i): sum over significant a with q_ab*l_bi >= thr
    kw = np.einsum('ab,a,b,abi->bi', sig, nf, nf, q[:, :, None] * lN[None] >= thr)
    frac_bi = kw / np.maximum(Wb_bpa[:, None], 1)
    idx = np.digitize(d, bins)
    p = np.array([frac_bi[idx == k].mean() if (idx == k).any() else 0 for k in range(len(bins) + 1)])
    prof[thr] = p
    print('BPA TZVP measured kept fraction thr %.0e: %.3f' % (thr, (kw.sum() / (Wb_bpa.sum() * lN.shape[1]))))


def model(xyz, label):
    m3 = gto.M(atom=xyz, basis='sto-3g', verbose=0)
    mf = scf.RHF(m3).density_fit(); mf.conv_tol = 1e-7; mf.kernel()
    no = m3.nelectron // 2
    L = lo.PM(m3, mf.mo_coeff[:, :no]).kernel()
    cen = np.einsum('xmn,mi,ni->ix', m3.intor('int1e_r'), L, L)
    mt = gto.M(atom=xyz, basis='def2-tzvp', verbose=0)
    S = mt.intor('int1e_ovlp'); al = mt.ao_loc_nr(); nsh = mt.nbas
    smax = np.array([[np.abs(S[al[a]:al[a + 1], al[b]:al[b + 1]]).max() for b in range(nsh)] for a in range(nsh)])
    nft = np.diff(al)
    shc = np.array([mt.bas_coord(i) for i in range(nsh)])
    dd = np.linalg.norm(shc[:, None] - cen[None], axis=2)
    return smax, nft, dd, no


sm_b, nf_b, dd_b, _ = model(bpa, 'BPA')
target = (nf[:, None] * nf[None, :] * sig).sum()
cuts = np.logspace(-14, -4, 101)
w = [(nf_b[:, None] * nf_b[None, :] * (sm_b >= c)).sum() for c in cuts]
s_cut = cuts[np.argmin(np.abs(np.log(np.array(w) / target)))]
print('s_cut %.1e (BPA pair weight %d vs q-based %d)' % (s_cut, (nf_b[:, None] * nf_b[None, :] * (sm_b >= s_cut)).sum(), target))
sm_t, nf_t, dd_t, no_t = model(tax, 'Taxol')
for thr in thrs:
    for nm, sm, nft, dd in (('BPA (model, STO-3G centroids)', sm_b, nf_b, dd_b), ('Taxol (model)', sm_t, nf_t, dd_t)):
        Wb = (nft[:, None] * nft[None, :] * (sm >= s_cut)).sum(0)
        p = prof[thr][np.digitize(dd, bins)]
        print('thr %.0e  %-30s kept fraction %.3f' % (thr, nm, (Wb[:, None] * p).sum() / (Wb.sum() * dd.shape[1])))
