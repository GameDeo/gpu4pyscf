'''Inner-loop exchange surrogates built from explicit DF tensors (CPU emulation, counts/accuracy only).
Outer: exact F*C_o (fine XC + fine NLC + exact DF-K with def2-universal-jkfit); occupied-projected defect.
Inner K options (comma-separated spec):
  aux=NAME        aux basis of the surrogate: 'full' (def2-universal-jkfit), 'lmaxN' (drop l>N aux shells),
                  'emaxE' (drop primitives with exponent > E), 'svpri' (def2-svp-ri), 'etbB' (even-tempered, beta B)
  mix             one tensor for FR+LR: K = B'^T B', B' = (hyb I + (alpha-hyb) G)^1/2 B, G = L^-1 J_erf L^-T
  fp16inc         K(D) = K(D_ref) + K_fp16(D - D_ref)   (operands rounded to fp16, fp32 accumulate)
  fp16abs         K(D) = K_fp16(D)
usage: python r4_cpu_auxK.py XYZ BASIS SPEC
'''
import sys
import numpy as np
import scipy.linalg
from pyscf import gto, dft, lib, df
from pyscf.scf import diis as scfdiis

lib.num_threads(4)
xyz, basis, spec = sys.argv[1], sys.argv[2], sys.argv[3]
opts = {}
for x in spec.split(','):
    if '=' in x:
        k_, v_ = x.split('=')
        opts[k_] = v_
    elif x in ('mix', 'fp16inc', 'fp16abs', 'ref'):
        opts[x] = True
    else:
        opts['aux'] = x            # bare aux-basis name
xc = 'wb97m_v'
mol = gto.M(atom=xyz, basis=basis, verbose=0)
mf = dft.RKS(mol).density_fit(auxbasis='def2-universal-jkfit')
mf.xc = xc
mf.nlc = 'VV10'
mf.grids.atom_grid = (99, 590)
mf.nlcgrids.atom_grid = (50, 194)
ni = mf._numint
omega, alpha, hyb = ni.rsh_and_hybrid_coeff(xc, spin=mol.spin)
nocc = mol.nelectron // 2
nao = mol.nao


def aux_basis(name):
    if name == 'full':
        return 'def2-universal-jkfit'
    if name == 'svpri':
        return 'def2-svp-ri'
    if name.startswith('etb'):
        return df.aug_etb(mol, beta=float(name[3:]))
    out = {}
    for el in set(mol.elements):
        shells = gto.load('def2-universal-jkfit', el)
        keep = []
        for sh in shells:
            l = sh[0]
            prims = [p for p in sh[1:]]
            if name.startswith('lmax') and l > int(name[4:]):
                continue
            if name.startswith('emax'):
                prims = [p for p in prims if p[0] <= float(name[4:])]
                if not prims:
                    continue
            keep.append([l] + prims)
        out[el] = keep
    return out


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

if spec == 'ref':
    mf.conv_tol = 1e-11
    mf.conv_tol_grad = 1e-7
    e = mf.kernel()
    print('RESULT ref E %.12f converged %s' % (e, mf.converged), flush=True)
    sys.exit()
# ---------------- surrogate DF tensors
auxmol = df.make_auxmol(mol, aux_basis(opts.get('aux', 'full')))
naux = auxmol.nao
int3c = df.incore.aux_e2(mol, auxmol, 'int3c2e', aosym='s1').reshape(nao * nao, naux)
J2 = auxmol.intor('int2c2e')
ev, V = np.linalg.eigh(J2)
keep = ev > 1e-10 * ev.max()
Linv = (V[:, keep] / np.sqrt(ev[keep])).T              # robust J^-1/2 (rows: orthonormal aux functions)
B = (Linv @ int3c.T).reshape(-1, nao, nao)
with auxmol.with_range_coulomb(omega):
    J2lr = auxmol.intor('int2c2e')
G = Linv @ J2lr @ Linv.T
if 'mix' in opts:
    Mk = hyb * np.eye(G.shape[0]) + (alpha - hyb) * G
    lam, Um = np.linalg.eigh(Mk)
    lam = np.clip(lam, 0, None)
    tensors = [((np.sqrt(lam)[:, None] * Um.T) @ B.reshape(len(lam), -1)).reshape(-1, nao, nao)]
    coefs = [1.0]
else:
    with mol.with_range_coulomb(omega):
        int3lr = df.incore.aux_e2(mol, auxmol, 'int3c2e', aosym='s1').reshape(nao * nao, naux)
    lam, Ug = np.linalg.eigh(J2lr)
    kl = lam > 1e-10 * lam.max()
    Linv_lr = (Ug[:, kl] / np.sqrt(lam[kl])).T
    Blr = (Linv_lr @ int3lr.T).reshape(-1, nao, nao)
    tensors, coefs = [B, Blr], [hyb, alpha - hyb]
naux_eff = sum(t.shape[0] for t in tensors)
print(f'spec {spec} nao {nao} naux {naux} tensor rows {naux_eff} (exact: 2 x {df.make_auxmol(mol, "def2-universal-jkfit").nao})',
      flush=True)


def r16(x):
    s = abs(x).max() or 1.0
    return (x / s).astype(np.float16).astype(np.float32), s


T16 = [r16(t) for t in tensors]


def K_sur(D, fp16=False):
    # factorised: D = V diag(lam) V^T (indefinite for increments), K = sum_L (B^L V) diag(lam) (B^L V)^T
    lam, Q = np.linalg.eigh(Sh @ D @ Sh)
    kk = abs(lam) > 1e-13 * max(abs(lam).max(), 1e-300)
    Vf = np.linalg.solve(Sh, Q[:, kk])
    lam = lam[kk]
    out = np.zeros((nao, nao))
    for t, c, (th, sc) in zip(tensors, coefs, T16):
        if fp16:
            Vh, sv = r16(Vf)
            Y = np.einsum('lmn,ns->lms', th, Vh)                 # fp16 operands, fp32 accumulation
            Y, sy = r16(Y)
            out += c * np.einsum('lms,s,lns->mn', Y, lam.astype(np.float32), Y).astype(np.float64) * (sc * sv * sy) ** 2
        else:
            Y = np.einsum('lmn,ns->lms', t, Vf)
            out += c * np.einsum('lms,s,lns->mn', Y, lam, Y)
    return out


def Kgen(dm):
    return hyb * mf.get_k(mol, dm) + (alpha - hyb) * mf.get_k(mol, dm, omega=omega)


def vxc(dm, fine, nlc):
    g, ng = (gf, nf) if fine else (gc, nc)
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


inc, f16abs = 'fp16inc' in opts, 'fp16abs' in opts
nK = 0
dm = mf.get_init_guess(key='minao')
d = scfdiis.CDIIS(mf)
for c in range(80):
    f = h1e + mf.get_j(mol, dm) + vxc(dm, False, True) - 0.5 * K_sur(dm, f16abs)
    nK += 1
    g = cnorm(f, dm)
    if g < 1e-3:
        break
    f = d.update(s1e, dm, f)
    dm = solve(f)
st1 = c + 1
Dref = dm
hist, inner = [], []
for outer in range(15):
    Kr = Kgen(Dref)
    Ftrue = h1e + mf.get_j(mol, Dref) + vxc(Dref, True, True) - 0.5 * Kr
    C = occ_orbitals(Dref)
    SC = s1e @ C
    Wt = Ftrue @ C                      # only F*C_o of the fine operator is used from here on
    X = 2 * Wt @ SC.T
    gt = np.linalg.norm(X - X.T)        # == ||F D S - S D F|| computed from F*C_o only
    hist.append(gt)
    if outer == 0:
        ks = K_sur(Dref)
        print('  surrogate K rel err %.2e' % (np.linalg.norm(ks - Kr) / np.linalg.norm(Kr)), flush=True)
    if gt < 1e-7:
        break
    Ks_ref = K_sur(Dref)
    Fcheap = h1e + mf.get_j(mol, Dref) + vxc(Dref, False, False) - 0.5 * Ks_ref
    W = Wt - Fcheap @ C
    Delta = W @ SC.T + SC @ W.T - SC @ (C.T @ W) @ SC.T
    dd = scfdiis.CDIIS(mf)
    dmi, f = Dref, Fcheap + Delta       # surrogate at D_ref (no full fine matrix)
    inner_tol = max(gt * 1e-2, 1e-9)
    for it in range(60):
        f = dd.update(s1e, dmi, f)
        dmi = solve(f)
        nK += 1
        Kin = Ks_ref + K_sur(dmi - Dref, fp16=True) if inc else K_sur(dmi, f16abs)
        f = h1e + mf.get_j(mol, dmi) + vxc(dmi, False, False) - 0.5 * Kin + Delta
        gs = cnorm(f, dmi)
        if gs < inner_tol:
            break
    inner.append(it + 1)
    Dref = dmi
Efin = mf.energy_tot(Dref, h1e, mf.get_veff(mol, Dref))
print('ENERGY %.12f' % Efin, flush=True)
print('RESULT', spec, 'naux_rows', naux_eff, 'stage1', st1, 'outer', ['%.1e' % x for x in hist], 'inner', inner,
      'surrogateK', nK, flush=True)
