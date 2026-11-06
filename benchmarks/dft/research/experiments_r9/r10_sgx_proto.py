'''Round 10 prototype: seminumerical (COSX-type) exchange surrogate on the GPU, without any stored 3-index tensor.
    K_mn = sum_g phi_m(g) w_g sum_s A_ns(g) G_sg,   G = D Phi^T,   A_ns(g) = (n| v(r - g) |s)
with v = hyb / r + (alpha - hyb) erf(omega r) / r (erf(omega r)/r: a Gaussian charge of exponent omega^2, so the same
1e grid-integral kernel), and overlap fitting Phi -> Phi Q, Q = S_g^-1 S (S_g = Phi^T W Phi on the same grid).
Measures, for a range-separated hybrid operator (wB97M-V weights), the response error on an occupied-virtual
perturbation against the exact DF-K (def2-universal-jkfit), next to the SVP-RI DF surrogate, and the time per call.
usage: MOL=033_Bisphenol_A.xyz python r10_sgx_proto.py'''
import os, sys, time, ctypes
import numpy as np
import cupy as cp
from pyscf import gto, lib
from gpu4pyscf import dft
from gpu4pyscf.gto import int3c1e
from gpu4pyscf.lib.cupy_helper import cart2sph

here = os.path.dirname(os.path.abspath(__file__))
mol = gto.M(atom=os.path.join(here, '..', '..', '..', 'molecules', 'organic', os.environ.get('MOL', '033_Bisphenol_A.xyz')),
            basis='def2-tzvp', verbose=0, max_memory=14000)
nao = mol.nao
nocc = mol.nelectron // 2
print('nao', nao, 'nocc', nocc, flush=True)


def sync():
    cp.cuda.Stream.null.synchronize()


def int3c1e_dev(intopt, grids, expo):
    '''A[g, n, s] on the device for a grid block (point charges, or Gaussian charges of exponent expo)'''
    ng = grids.shape[0]
    out = cp.zeros([ng, nao, nao])
    grids = cp.asarray(grids, order='C')
    ex = None if expo is None else cp.zeros(ng) + expo
    for cp_ij_id, _ in enumerate(intopt.log_qs):
        log_q_ij = intopt.log_qs[cp_ij_id]
        if len(log_q_ij) == 0:
            continue
        cpi = intopt.cp_idx[cp_ij_id]
        cpj = intopt.cp_jdx[cp_ij_id]
        li = intopt.angular[cpi]
        lj = intopt.angular[cpj]
        i0, i1 = intopt.cart_ao_loc[cpi], intopt.cart_ao_loc[cpi + 1]
        j0, j1 = intopt.cart_ao_loc[cpj], intopt.cart_ao_loc[cpj + 1]
        ni, nj = i1 - i0, j1 - j0
        sl = cp.zeros([ng, nj, ni], order='C')
        err = int3c1e.libgint.GINTfill_int3c1e(
            ctypes.cast(cp.cuda.get_current_stream().ptr, ctypes.c_void_p), intopt.bpcache,
            ctypes.cast(grids.data.ptr, ctypes.c_void_p),
            ctypes.cast(ex.data.ptr if ex is not None else 0, ctypes.c_void_p), ctypes.c_int(ng),
            ctypes.cast(sl.data.ptr, ctypes.c_void_p),
            np.array([ni, ni * nj], dtype=np.int32).ctypes.data_as(ctypes.c_void_p),
            np.array([i0, j0], dtype=np.int32).ctypes.data_as(ctypes.c_void_p),
            np.array([0, len(log_q_ij)], dtype=np.int32).ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(1), ctypes.c_int(cp_ij_id), ctypes.c_double(0.))
        assert err == 0
        i0, i1 = intopt.ao_loc[cpi], intopt.ao_loc[cpi + 1]
        j0, j1 = intopt.ao_loc[cpj], intopt.ao_loc[cpj + 1]
        sl = cart2sph(sl, axis=1, ang=lj)
        sl = cart2sph(sl, axis=2, ang=li)
        out[:, j0:j1, i0:i1] = sl
    r, c = np.tril_indices(nao)
    out[:, r, c] = out[:, c, r]
    return out          # sorted AO order (intopt), unsorted by the caller


class SGXK:
    def __init__(self, mol, atom_grid, omega, alpha, hyb, blk=None, fit=True):
        mf = dft.RKS(mol)
        mf.grids.atom_grid = atom_grid
        mf.grids.build(with_non0tab=False, sort_grids=False)
        self.coords = cp.asarray(mf.grids.coords)
        self.w = cp.asarray(mf.grids.weights)
        from gpu4pyscf.dft.numint import eval_ao
        self.phi = cp.ascontiguousarray(eval_ao(mol, self.coords, deriv=0, transpose=False))   # (nao, ng)
        self.intopt = int3c1e.VHFOpt(mol)
        self.intopt.build(1e-13, aosym=True)
        self.omega, self.alpha, self.hyb = omega, alpha, hyb
        ng = self.coords.shape[0]
        self.blk = blk or max(64, int(1.5e9 / (8 * nao * nao)))
        S = cp.asarray(mol.intor('int1e_ovlp'))
        if fit:
            Sg = (self.phi * self.w) @ self.phi.T
            self.Q = cp.linalg.solve(Sg, S)       # Phi -> Q^T Phi (overlap fitting on the left index)
        else:
            self.Q = cp.eye(nao)
        print('SGX grid', atom_grid, 'points', ng, 'block', self.blk, flush=True)

    def k(self, D):
        D = cp.asarray(D)
        ng = self.coords.shape[0]
        G = D @ self.phi                            # (nao, ng)
        K = cp.zeros((nao, nao))
        uns = self.intopt.unsort_orbitals
        for g0 in range(0, ng, self.blk):
            g1 = min(ng, g0 + self.blk)
            A = None
            if abs(self.hyb) > 0:
                A = self.hyb * int3c1e_dev(self.intopt, self.coords[g0:g1], None)
            if self.omega and abs(self.alpha - self.hyb) > 0:
                A2 = (self.alpha - self.hyb) * int3c1e_dev(self.intopt, self.coords[g0:g1], self.omega ** 2)
                A = A2 if A is None else A + A2
                A2 = None
            A = uns(A, axis=[1, 2])
            Y = cp.matmul(A, cp.ascontiguousarray(G[:, g0:g1].T)[:, :, None])[:, :, 0]   # (ng, nao)
            K += (self.phi[:, g0:g1] * self.w[g0:g1]) @ Y
            A = None
        K = self.Q.T @ K
        return 0.5 * (K + K.T)


def main():
    xc = 'wb97m-v'
    mf = dft.RKS(mol, xc=xc).density_fit()
    omega, alpha, hyb = mf._numint.rsh_and_hybrid_coeff(xc)
    mf.grids.atom_grid = (50, 194)
    mf.nlcgrids.atom_grid = (20, 50)
    mf.conv_tol = 1e-7
    mf.kernel()
    C = mf.mo_coeff
    Co = C[:, :nocc]
    Cv = C[:, nocc:]
    rng = np.random.default_rng(7)
    U = cp.asarray(rng.standard_normal((Cv.shape[1], nocc)) * 1e-3)
    dD = Cv @ U @ Co.T
    dD = dD + dD.T                                       # ov response perturbation
    D0 = mf.make_rdm1()

    def kx(dm):
        v = hyb * mf.get_k(mol, dm, hermi=1)
        v += (alpha - hyb) * mf.get_k(mol, dm, hermi=1, omega=omega)
        return v

    def ov(K):
        return Cv.T @ K @ Co

    kx_d = ov(kx(dD))
    kx_0 = ov(kx(D0))
    ref = float(cp.linalg.norm(kx_d))
    print('exact |K(dD)_ov| %.3e' % ref, flush=True)
    # SVP-RI DF surrogate (current default)
    from gpu4pyscf.scf.multifidelity import SurrogateK
    sk = SurrogateK(mf, 'def2-svp-ri', omega, alpha, hyb, mode='fp32')
    e = float(cp.linalg.norm(ov(sk.k_ref(dD)) - kx_d)) / ref
    print('SVP-RI DF surrogate: rel. response error %.2e' % e, flush=True)
    for grid in [(20, 50), (30, 110), (40, 194)]:
        for fit in (True, False):
            sg = SGXK(mol, grid, omega, alpha, hyb, fit=fit)
            sync(); t0 = time.perf_counter()
            kd = sg.k(dD)
            sync(); t1 = time.perf_counter()
            e = float(cp.linalg.norm(ov(kd) - kx_d)) / ref
            e0 = float(cp.linalg.norm(ov(sg.k(D0)) - kx_0)) / float(cp.linalg.norm(kx_0))
            print('SGX %s fit=%s: rel. response error %.2e  (absolute D0 ov: %.2e)  time/call %.2f s' % (
                grid, fit, e, e0, t1 - t0), flush=True)
            sg = None
            cp.get_default_memory_pool().free_all_blocks()


main()
print('ALLDONE')
