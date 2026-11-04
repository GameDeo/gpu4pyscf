# Copyright 2021-2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
INT8 residue (CRT) GEMM emulation: gemm_nt against a FP64 reference, and the XC matrix of nr_rks with
XC_PRECISION='crt' against the FP64 Vxc (needs a GPU with INT8 GEMM support, i.e. sm_75 or newer).
'''
import unittest
import numpy as np
import cupy
from pyscf import gto
from gpu4pyscf import dft
from gpu4pyscf.dft import numint
from gpu4pyscf.lib import crt_gemm


def setUpModule():
    global mol, mf
    mol = gto.M(
        atom='''O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692;
                C 2.6 0.2 0.1; H 3.2 0.9 0.6; H 3.2 -0.5 -0.4; H 2.0 -0.3 0.9; H 2.0 0.9 -0.5''',
        basis='def2-svp', verbose=0)
    mf = dft.RKS(mol, xc='pbe').density_fit()
    mf.grids.atom_grid = (50, 194)
    mf.max_cycle = 3
    mf.kernel()


def tearDownModule():
    global mol, mf
    del mol, mf


class KnownValues(unittest.TestCase):
    def test_gemm_nt(self):
        rng = np.random.default_rng(1)
        a = cupy.asarray(rng.standard_normal((2, 37, 1000)) * np.exp(-3 * np.abs(rng.standard_normal((2, 37, 1000)))))
        b = cupy.asarray(rng.standard_normal((2, 21, 1000)) * np.exp(-3 * np.abs(rng.standard_normal((2, 21, 1000)))))
        ref = cupy.einsum('cig,cjg->ij', a, b)
        scale = float(cupy.abs(ref).max())
        for nmod, tol in ((8, 1e-5), (11, 1e-9), (14, 1e-12)):
            c = crt_gemm.gemm_nt(a, b, nmod)
            err = float(cupy.abs(c - ref).max()) / scale
            print(f'[gemm_nt nmod={nmod}] rel err {err:.2e}')
            self.assertLess(err, tol)

    def test_gemm_nt_chunked(self):
        '''reduction length above the int32 accumulation limit is chunked'''
        rng = np.random.default_rng(2)
        a = cupy.asarray(rng.standard_normal((1, 16, 300000)))
        b = cupy.asarray(rng.standard_normal((1, 16, 300000)))
        ref = a[0] @ b[0].T
        c = crt_gemm.gemm_nt(a, b, 12)
        self.assertLess(float(cupy.abs(c - ref).max() / cupy.abs(ref).max()), 1e-9)

    def test_gemm_planes(self):
        '''transposed-layout operand: out[(i, l), b] = sum_j A[i, j, l] bt[b, j] for a stride-1 axis-1 view'''
        rng = np.random.default_rng(3)
        w = cupy.asarray(rng.standard_normal((50, 50, 40)) * np.exp(-2 * np.abs(rng.standard_normal((50, 50, 40)))))
        src = cupy.ascontiguousarray(w[:, :, :37].transpose(0, 1, 2))  # [i, j, L]  (L contiguous)
        view = src.transpose(0, 2, 1)                                  # (i, L, j)
        bt = cupy.asarray(rng.standard_normal((9, 50)))
        ref = cupy.einsum('ilj,bj->ilb', view, bt)
        out, P1 = crt_gemm.gemm_planes(view, bt, 12)
        out = out.reshape(50, P1, 9)[:, :37]
        err = float(cupy.abs(out - ref).max() / cupy.abs(ref).max())
        print(f'[gemm_planes] rel err {err:.2e}')
        self.assertLess(err, 1e-9)

    def test_rho(self):
        from gpu4pyscf.dft.numint import _eval_rho2, _eval_rho2_crt
        ni = mf._numint
        opt = ni.gdftopt
        mo = cupy.asarray(mf.mo_coeff)[:, mf.mo_occ > 0] * cupy.asarray(mf.mo_occ[mf.mo_occ > 0]) ** .5
        n = 0
        for ao_mask, idx, weight, _ in ni.block_loop(opt._sorted_mol, mf.grids, mol.nao, 1, max_memory=None):
            cpos = cupy.asarray(mo[idx], order='C')
            for xctype in ('LDA', 'GGA', 'MGGA'):
                ao = ao_mask[0] if xctype == 'LDA' else ao_mask
                r0 = _eval_rho2(ao_mask[0] if xctype == 'LDA' else ao_mask, cpos, xctype)
                r1 = _eval_rho2_crt(ao, cpos, xctype)
                self.assertLess(float(cupy.abs(r1 - r0).max() / cupy.abs(r0).max()), 1e-9)
            n += 1
            if n >= 3:
                break

    def test_k(self):
        from gpu4pyscf.df import df_jk
        dm = mf.make_rdm1()
        vj0, vk0 = mf.get_jk(mol, dm)
        old = df_jk.K_PRECISION, df_jk.K_CRT_MODULI
        df_jk.K_PRECISION, df_jk.K_CRT_MODULI = 'crt', 12
        try:
            vj1, vk1 = mf.get_jk(mol, dm)
        finally:
            df_jk.K_PRECISION, df_jk.K_CRT_MODULI = old
        err = float(cupy.abs(vk1 - vk0).max())
        print(f'[K crt] max|dK| {err:.2e}')
        self.assertLess(err, 1e-9)

    def test_cderi(self):
        '''DF tensor from the INT8-emulated aux-metric transform vs FP64'''
        from gpu4pyscf.df import df
        mf2 = dft.RKS(mol, xc='pbe').density_fit()
        ref = mf2.with_df
        ref.build()
        c0 = cupy.asarray(ref._cderi[0])
        old = df.CDERI_PRECISION
        df.CDERI_PRECISION = 'crt'
        try:
            mf3 = dft.RKS(mol, xc='pbe').density_fit()
            mf3.with_df.build()
        finally:
            df.CDERI_PRECISION = old
        c1 = cupy.asarray(mf3.with_df._cderi[0])
        err = float(cupy.abs(c1 - c0).max() / cupy.abs(c0).max())
        print(f'[cderi crt] max rel err {err:.2e}')
        self.assertLess(err, 1e-11)

    def _check(self, xc, nmod, tol):
        dm = mf.make_rdm1()
        n0, e0, v0 = mf._numint.nr_rks(mol, mf.grids, xc, dm)
        old = numint.XC_PRECISION, numint.XC_CRT_MODULI
        numint.XC_PRECISION, numint.XC_CRT_MODULI = 'crt', nmod
        try:
            n1, e1, v1 = mf._numint.nr_rks(mol, mf.grids, xc, dm)
        finally:
            numint.XC_PRECISION, numint.XC_CRT_MODULI = old
        dv = float(cupy.abs(v1 - v0).max())
        print(f'[{xc} crt nmod={nmod}] dExc={e1 - e0:.2e} max|dV|={dv:.2e}')
        self.assertLess(abs(e1 - e0), tol)
        self.assertLess(dv, tol)

    def test_lda(self):
        self._check('lda,vwn', 12, 1e-10)

    def test_gga(self):
        self._check('pbe', 12, 1e-10)

    def test_mgga(self):
        self._check('wb97m-v', 12, 1e-10)


if __name__ == "__main__":
    print("Full Tests for INT8 CRT GEMM / XC")
    unittest.main()
