# CPU emulation of the E3 "defect-corrected dual-grid SCF" on a small molecule: count fine/coarse evaluations
import numpy as np, time, sys, json
from pyscf import gto, dft, scf, lib
from pyscf.scf import diis as scfdiis
lib.num_threads(4)
xyz = sys.argv[1]; basis = sys.argv[2]; xc = sys.argv[3]
mol = gto.M(atom=xyz, basis=basis, verbose=0)
mf = dft.RKS(mol).density_fit(auxbasis='def2-universal-jkfit')
mf.xc = xc
isvv10 = 'wb97m_v' in xc or 'wb97x_v' in xc or 'b97m_v' in xc
if isvv10: mf.nlc = 'VV10'
import copy
def mk(atom_grid, nlc_grid=None):
    g = dft.gen_grid.Grids(mol); g.atom_grid = atom_grid; g.build()
    ng = None
    if isvv10:
        ng = dft.gen_grid.Grids(mol); ng.atom_grid = nlc_grid; ng.build()
    return g, ng
gf, nf = mk((99,590), (50,194))
gc, nc = mk((50,194), (30,110))
print('nao', mol.nao, 'ng fine/coarse', gf.coords.shape[0], gc.coords.shape[0], flush=True)
s1e = mf.get_ovlp(); h1e = mf.get_hcore()
cnt = {'fine':0,'coarse':0}
def fock(dm, which):
    if which=='fine': mf.grids, mf.nlcgrids = gf, nf
    else: mf.grids, mf.nlcgrids = gc, nc
    cnt[which]+=1
    vhf = mf.get_veff(mol, dm)
    return h1e + vhf, vhf
def gradnorm(f, dm_mo):
    mo, occ = dm_mo
    return np.linalg.norm(mf.get_grad(mo, occ, f))
def cnorm(f, dm):
    x = f @ dm @ s1e
    return np.linalg.norm(x - x.T)
def newdm(f):
    e, mo = mf.eig(f, s1e); occ = mf.get_occ(e, mo)
    return mf.make_rdm1(mo, occ), (mo, occ)
def energy(dm, vhf): return mf.energy_tot(dm, h1e, vhf)
# initial guess
dm = mf.get_init_guess(key='minao')
t0=time.time()
# --- baseline: plain DIIS on the fine grid, convergence to 1e-6 (orbital gradient)
def scf_plain(dm, which, tol, maxc=60, label=''):
    d = scfdiis.CDIIS(mf); hist=[]
    for c in range(maxc):
        f, vhf = fock(dm, which)
        e = energy(dm, vhf)
        g = cnorm(f, dm)
        hist.append((c, e, g))
        print(f'{label} {which} cyc {c:2d} E={e:.10f} |g|={g:.2e}', flush=True)
        if g < tol: return dm, e, hist
        f = d.update(s1e, dm, f)
        dm, _ = newdm(f)
    return dm, e, hist
mode = sys.argv[4]
if mode=='baseline':
    dm, e, h = scf_plain(dm, 'fine', 1e-7, label='base')
    print('RESULT baseline fine evals', cnt['fine'], 'E', e)
elif mode=='dual':
    # stage 1: coarse to |g|<1e-3
    dm, e, h = scf_plain(dm, 'coarse', 1e-3, label='stage1')
    nstage1 = cnt['coarse']
    tol = 1e-7
    Dref = dm
    for outer in range(8):
        Ff, vf = fock(Dref, 'fine'); 
        gt = cnorm(Ff, Dref)
        Etrue = energy(Dref, vf)
        print(f'OUTER {outer} fine eval: E_true={Etrue:.10f} |g_true|={gt:.2e}  (fine so far {cnt["fine"]}, coarse so far {cnt["coarse"]})', flush=True)
        if gt < tol: break
        Fc, vc = fock(Dref, 'coarse'); Delta = Ff - Fc
        d = scfdiis.CDIIS(mf); dmi = Dref; Fm=Ff
        # inner loop: surrogate F(D) = Fc(D) + Delta; start from diagonalising Ff (true) 
        f = Ff
        inner_tol = max(gt*1e-3, 1e-9)
        for it in range(30):
            f = d.update(s1e, dmi, f)
            dmi, (mo, occ) = newdm(f)
            Fc_i, vc_i = fock(dmi, 'coarse'); f = Fc_i + Delta
            gs = cnorm(f, dmi)
            print(f'   inner {it:2d} |g_surr|={gs:.2e}', flush=True)
            if gs < inner_tol: break
        Dref = dmi
    print('RESULT dual: fine evals', cnt['fine'], 'coarse evals (incl stage1', nstage1, ')', cnt['coarse'], 'E_true', Etrue)
print('time', time.time()-t0)
