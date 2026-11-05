# CPU emulation: defect-corrected dual-grid XC + frozen ACE exchange (refreshed each outer) -> count exact refreshes
import numpy as np, sys, time
from pyscf import gto, dft, lib
from pyscf.scf import diis as scfdiis
lib.num_threads(4)
xyz, basis, xc, coarse = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
mol = gto.M(atom=xyz, basis=basis, verbose=0)
mf = dft.RKS(mol).density_fit(auxbasis='def2-universal-jkfit'); mf.xc = xc
isvv = 'wb97m_v' in xc
if isvv: mf.nlc='VV10'
ni = mf._numint
omega, alpha, hyb = ni.rsh_and_hybrid_coeff(xc, spin=mol.spin)
print('omega alpha hyb', omega, alpha, hyb)
def grid(ag): 
    g = dft.gen_grid.Grids(mol); g.atom_grid=ag; g.build(); return g
gf, nf = grid((99,590)), (grid((50,194)) if isvv else None)
cg = eval(coarse)
gc, nc = grid(cg[0]), (grid(cg[1]) if isvv else None)
s1e = mf.get_ovlp(); h1e = mf.get_hcore()
small = sys.argv[6] if len(sys.argv)>6 else 'def2-svp'
mols = gto.M(atom=xyz, basis=small, verbose=0)
mfs = dft.RKS(mols).density_fit(auxbasis='def2-universal-jkfit'); mfs.xc = xc
Sss = mols.intor('int1e_ovlp'); Ssb = gto.intor_cross('int1e_ovlp', mols, mol)
Pm = np.linalg.solve(Sss, Ssb)   # small x big
print('nao big/small', mol.nao, mols.nao)
def Ksmall(dm):
    dms = Pm @ dm @ Pm.T
    k = hyb*mfs.get_k(mols, dms) if hyb!=0 else 0
    if omega!=0: k = k + (alpha-hyb)*mfs.get_k(mols, dms, omega=omega)
    return Pm.T @ k @ Pm
cnt = dict(fineXC=0, coarseXC=0, Kfull=0, J=0)
def vxc(dm, fine, nlc=True):
    g, ng = (gf,nf) if fine else (gc,nc)
    cnt['fineXC' if fine else 'coarseXC'] += 1
    n, exc, v = ni.nr_rks(mol, g, xc, dm)
    if isvv and nlc:
        cnt['nlc_fine' if fine else 'nlc_coarse'] = cnt.get('nlc_fine' if fine else 'nlc_coarse',0)+1
        _, enlc, vnlc = ni.nr_nlc_vxc(mol, ng, xc, dm); v = v + vnlc
    return v
def vnlc_only(dm, fine):
    g, ng = (gf,nf) if fine else (gc,nc)
    _, enlc, vnlc = ni.nr_nlc_vxc(mol, ng, xc, dm); return vnlc
def Jm(dm): cnt['J']+=1; return mf.get_j(mol, dm)
def Kfull(dm):
    cnt['Kfull']+=1
    k = hyb*mf.get_k(mol, dm) if hyb!=0 else 0
    if omega!=0: k = k + (alpha-hyb)*mf.get_k(mol, dm, omega=omega)
    return k
def cnorm(f, dm): x = f@dm@s1e; return np.linalg.norm(x-x.T)
def solve(f):
    e, mo = mf.eig(f, s1e); occ = mf.get_occ(e, mo); return mf.make_rdm1(mo,occ), mo, occ
# stage 1: coarse everything (no ACE) to 1e-3
dm = mf.get_init_guess(key='minao'); d = scfdiis.CDIIS(mf)
for c in range(40):
    f = h1e + Jm(dm) + vxc(dm, False) - 0.5*Kfull(dm)
    g = cnorm(f, dm)
    print(f'stage1 {c} |g|={g:.2e}', flush=True)
    if g < 1e-3: break
    f = d.update(s1e, dm, f); dm, mo, occ = solve(f)
nK1, nc1 = cnt['Kfull'], cnt['coarseXC']
Dref = dm
mode = sys.argv[5]  # 'ace' or 'exactK'
for outer in range(10):
    Jr = Jm(Dref); Vf = vxc(Dref, True)
    Vc = vxc(Dref, False, nlc=(mode not in ('frozenNLC','smallK'))); Kr = Kfull(Dref)
    Ftrue = h1e + Jr + Vf - 0.5*Kr
    gt = cnorm(Ftrue, Dref)
    print(f'OUTER {outer}: |g_true|={gt:.2e}', flush=True)
    if gt < 1e-7: break
    Delta = Vf - Vc
    # occupied orbitals of Dref
    ev, mo = mf.eig(Ftrue, s1e)  # eigenvectors for ACE span: use occupied orbitals of Dref via D = C C^T
    # get C_o from Dref: eigen-decompose Dref S
    w, U = np.linalg.eig(Dref @ s1e); idx = np.argsort(-w.real)[:mol.nelectron//2]
    Co = U[:, idx].real
    Co = Co @ np.linalg.inv(np.linalg.cholesky(Co.T @ s1e @ Co)).T  # S-orthonormalise
    DeltaK = -0.5*(Kr - Ksmall(Dref))
    Vx = -0.5*Kr
    X = Vx @ Co; M = Co.T @ X
    Vace = X @ np.linalg.solve(M, X.T)
    d = scfdiis.CDIIS(mf); dmi = Dref; f = Ftrue
    inner_tol = max(gt*1e-3, 1e-9)
    for it in range(40):
        f = d.update(s1e, dmi, f); dmi, mo, occ = solve(f)
        if mode=='ace': Kterm = Vace
        elif mode=='smallK': Kterm = -0.5*Ksmall(dmi) + DeltaK
        else: Kterm = -0.5*Kfull(dmi)  # exactK / frozenNLC
        f = h1e + Jm(dmi) + vxc(dmi, False, nlc=(mode not in ('frozenNLC','smallK'))) + Delta + Kterm
        gs = cnorm(f, dmi)
        if gs < inner_tol: break
    print(f'    inner cycles {it+1}, last |g_surr|={gs:.2e}', flush=True)
    Dref = dmi
print('RESULT', mode, cnt, 'stage1 coarse evals', nc1)
