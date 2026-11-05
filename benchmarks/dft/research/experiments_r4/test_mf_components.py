'''GPU checks of the multi-fidelity SCF components (water dimer and Vitamin C, def2-TZVP, wB97M-V + DF).'''
import os, sys, time
import numpy as np
import cupy
from pyscf import gto
from gpu4pyscf import dft
from gpu4pyscf.dft import numint
from gpu4pyscf.scf import multifidelity as mfd, mixed_precision as mp
from gpu4pyscf.lib.cupy_helper import tag_array

here = os.path.dirname(os.path.abspath(__file__))
xyz = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, 'h2o_dimer.xyz')
mol = gto.M(atom=xyz, basis='def2-tzvp', verbose=0)
mf = dft.RKS(mol, xc='wb97m-v').density_fit()
mf.grids.atom_grid = (99, 590)
mf.nlcgrids.atom_grid = (50, 194)
mf.conv_tol = 1e-11
mf.conv_tol_grad = 1e-7
t0 = time.perf_counter()
e_ref = mf.kernel()
print('reference FP64 SCF E = %.12f  (%.1f s)' % (e_ref, time.perf_counter() - t0), flush=True)
ni = mf._numint
dm = mf.make_rdm1()
dm = tag_array(dm, mo_coeff=mf.mo_coeff, mo_occ=mf.mo_occ)
Co = mf.mo_coeff[:, mf.mo_occ > 0]

# 1. occupied-only XC vs full matrix
for prec in ('fp64', 'crt'):
    numint.XC_PRECISION = prec
    n, e, v = ni.nr_rks(mol, mf.grids, mf.xc, dm)
    n2, e2, vc = numint.nr_rks_vco(ni, mol, mf.grids, mf.xc, mf.mo_coeff, mf.mo_occ)
    ref = v @ Co
    print(f'[1] nr_rks_vco {prec}: max|VC - V C| / max|V C| = {float(abs(vc - ref).max() / abs(ref).max()):.2e}  '
          f'dExc = {e2 - e:.2e}  dN = {n2 - n:.2e}', flush=True)
numint.XC_PRECISION = 'fp64'

# 1b. fused XC gradient vs rks_grad.get_exc
from gpu4pyscf.grad import rks as rks_grad
for prec in ('fp64', 'crt'):
    numint.XC_PRECISION = prec
    g_ref = rks_grad.get_exc(ni, mol, mf.grids, mf.xc, dm)[1] * 2
    g_f = numint.nr_rks_vco(ni, mol, mf.grids, mf.xc, mf.mo_coeff, mf.mo_occ, with_grad=True)[3]
    print(f'[1b] fused XC gradient {prec}: max|dg| = {float(abs(np.asarray(g_f) - np.asarray(g_ref)).max()):.2e}  '
          f'(max|g_xc| {float(abs(np.asarray(g_ref)).max()):.2e})', flush=True)
numint.XC_PRECISION = 'fp64'

# 2. FP16 GEMM
a = cupy.random.standard_normal((300, 64)).astype(np.float16)
b = cupy.random.standard_normal((200, 64)).astype(np.float16)
r = a.astype(np.float32) @ b.astype(np.float32).T
print('[2] gemm16_nt err', float(abs(mfd.gemm16_nt(a, b) - r).max()), flush=True)

# 3. surrogate exchange: increment vs accurate evaluation
omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=0)
sk = mfd.SurrogateK(mf, 'def2-svp-ri', omega, alpha, hyb, mode='fp16')
s1e = cupy.asarray(mf.get_ovlp())
# perturbed reference orbitals (rotate occ-virt by a small angle)
rng = np.random.default_rng(0)
C = cupy.asarray(mf.mo_coeff)
nocc = Co.shape[1]
kappa = cupy.asarray(rng.standard_normal((C.shape[1] - nocc, nocc))) * 1e-3
Cp = C[:, :nocc] + C[:, nocc:] @ kappa
Lc = cupy.linalg.cholesky(Cp.T @ s1e @ Cp)
Cp = cupy.linalg.solve(Lc, Cp.T).T
Dref = tag_array(2 * Cp @ Cp.T, mo_coeff=cupy.hstack([Cp, C[:, nocc:]]),
                 mo_occ=cupy.asarray([2.] * nocc + [0.] * (C.shape[1] - nocc)))
kref = sk.k_ref(Dref)
sk.set_ref(Cp, s1e)
kexact = sk.k_ref(dm)
kinc = kref + sk.k_inc(cupy.asarray(Co))
dk = kexact - kref
print('[3] K_s increment: |K(D)-K(Dref)| = %.2e, error of FP16 increment = %.2e (rel %.1e)  naux %d' % (
    float(abs(dk).max()), float(abs(kinc - kexact).max()), float(abs(kinc - kexact).max() / abs(dk).max()), sk.naux),
    flush=True)

# 4. eigensolver
f = mf.get_fock(dm=dm)
eg = mfd.FastEig(s1e, 'fp32oa', nocc)
t0 = time.perf_counter(); e1, c1 = eg(f); cupy.cuda.Stream.null.synchronize(); t1 = time.perf_counter() - t0
t0 = time.perf_counter(); e2, c2 = mf.eig(f, s1e); cupy.cuda.Stream.null.synchronize(); t2 = time.perf_counter() - t0
d1 = c1[:, :nocc] @ c1[:, :nocc].T
d2 = c2[:, :nocc] @ c2[:, :nocc].T
print('[4] FastEig density err %.2e  eps_occ err %.2e  (%.3f s vs mf.eig %.3f s)' % (
    float(abs(d1 - d2).max()), float(abs(e1[:nocc] - e2[:nocc]).max()), t1, t2), flush=True)

# 5. whole multi-fidelity SCF (FP64 kernels, then crt)
for label, mprec in (('fp64', None), ('crt', {'scheme': 'crt'})):
    m2 = dft.RKS(mol, xc='wb97m-v').density_fit()
    m2.grids.atom_grid = (99, 590)
    m2.nlcgrids.atom_grid = (50, 194)
    m2.conv_tol = 1e-11
    if mprec:
        m2.mixed_precision = mprec
    m2.multifidelity = {'conv_tol_grad': 1e-6}
    t0 = time.perf_counter()
    e = m2.kernel()
    st = m2.multifidelity_stats
    print(f'[5] multi-fidelity ({label}): E = {e:.12f}  dE = {e - e_ref:.2e}  converged {m2.converged} '
          f'cycles {st["cycles"]}  {time.perf_counter() - t0:.1f} s', flush=True)
    print('    outer |g|:', ['%.1e' % g for g, _ in st['outer_hist']], flush=True)
    gm = m2.nuc_grad_method().kernel()
    gr = mf.nuc_grad_method().kernel()
    print('    gradient vs FP64 reference: max|dg| = %.2e  fused XC used: %s' % (
        float(abs(np.asarray(gm) - np.asarray(gr)).max()), m2._mf_xc_grad is not None), flush=True)
print('done')
