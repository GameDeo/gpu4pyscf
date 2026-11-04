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
The FP32 K-build path (df_jk.K_PRECISION = 'fp32') must agree with the FP64 path to
~1e-6 relative accuracy on a fixed density.  A deviation near 1e-3 would mean the
contraction silently ran in TF32.
'''
import unittest
import numpy as np
import cupy
from pyscf import gto
from gpu4pyscf import scf
from gpu4pyscf.df import df_jk


def setUpModule():
    global mol, mf
    mol = gto.M(
        atom='''O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692;
                C 2.6 0.2 0.1; H 3.2 0.9 0.6; H 3.2 -0.5 -0.4; H 2.0 -0.3 0.9; H 2.0 0.9 -0.5''',
        basis='def2-tzvp', verbose=0)
    mf = scf.RHF(mol).density_fit()
    mf.max_cycle = 3
    mf.kernel()


def tearDownModule():
    global mol, mf
    del mol, mf


class KnownValues(unittest.TestCase):
    def _jk(self, precision, dm, **kw):
        old = df_jk.K_PRECISION
        df_jk.K_PRECISION = precision
        try:
            return mf.with_df.get_jk(dm, hermi=1, **kw)
        finally:
            df_jk.K_PRECISION = old

    def test_k_fp32_with_mo_coeff(self):
        dm = mf.make_rdm1()   # carries mo_coeff -> occupied-orbital factorized path
        vj0, vk0 = self._jk('fp64', dm)
        vj1, vk1 = self._jk('fp32', dm)
        scale = float(cupy.abs(vk0).max())
        err = float(cupy.abs(vk1 - vk0).max()) / scale
        print(f'[{self._testMethodName}] max|dK|/max|K| = {err:.2e}')
        self.assertLess(err, 5e-6)
        # J is unchanged by the K precision
        self.assertAlmostEqual(float(cupy.abs(vj1 - vj0).max()), 0, 12)

    def test_k_tf32x3(self):
        # 3xTF32 must reach FP32-level accuracy, not the ~1e-3 of a plain TF32 GEMM
        dm = mf.make_rdm1()
        vj0, vk0 = self._jk('fp64', dm)
        vj1, vk1 = self._jk('tf32x3', dm)
        err = float(cupy.abs(vk1 - vk0).max()) / float(cupy.abs(vk0).max())
        print(f'[test_k_tf32x3] max|dK|/max|K| = {err:.2e}')
        self.assertLess(err, 5e-6)

    def test_k_fp32_plain_dm(self):
        dm = cupy.asarray(np.asarray(mf.make_rdm1().get()))   # plain array -> SVD factorization
        vj0, vk0 = self._jk('fp64', dm)
        vj1, vk1 = self._jk('fp32', dm)
        scale = float(cupy.abs(vk0).max())
        err = float(cupy.abs(vk1 - vk0).max()) / scale
        print(f'[{self._testMethodName}] max|dK|/max|K| = {err:.2e}')
        self.assertLess(err, 5e-6)

    def test_k_fp32_lr(self):
        dm = mf.make_rdm1()
        vj0, vk0 = self._jk('fp64', dm, omega=0.3, with_j=False)
        vj1, vk1 = self._jk('fp32', dm, omega=0.3, with_j=False)
        scale = float(cupy.abs(vk0).max())
        err = float(cupy.abs(vk1 - vk0).max()) / scale
        print(f'[{self._testMethodName}] max|dK|/max|K| = {err:.2e}')
        self.assertLess(err, 5e-6)


if __name__ == "__main__":
    print("Full Tests for DF K fp32")
    unittest.main()
