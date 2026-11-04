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
The reduced-precision XC grid contractions (numint.XC_PRECISION) must reproduce the FP64
Exc and Vxc matrix on a fixed density to ~1e-6 relative accuracy.
'''
import unittest
import cupy
from pyscf import gto
from gpu4pyscf import dft
from gpu4pyscf.dft import numint


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


def _vxc(xc, precision):
    old = numint.XC_PRECISION
    numint.XC_PRECISION = precision
    try:
        dm = mf.make_rdm1()
        return mf._numint.nr_rks(mol, mf.grids, xc, dm)
    finally:
        numint.XC_PRECISION = old


class KnownValues(unittest.TestCase):
    def _check(self, xc, precision):
        n0, e0, v0 = _vxc(xc, 'fp64')
        n1, e1, v1 = _vxc(xc, precision)
        dv = float(cupy.abs(v1 - v0).max())
        print(f'[{xc} {precision}] dN={float(n1 - n0):.2e} dExc={e1 - e0:.2e} max|dV|={dv:.2e}')
        self.assertLess(abs(float(n1 - n0)), 1e-4)
        self.assertLess(abs(e1 - e0), 1e-4)
        self.assertLess(dv, 1e-5)

    def test_lda_fp32(self):
        self._check('lda,vwn', 'fp32')

    def test_gga_fp32(self):
        self._check('pbe', 'fp32')

    def test_mgga_fp32(self):
        self._check('tpss', 'fp32')

    def test_mgga_tf32x3(self):
        self._check('wb97m-v', 'tf32x3')

    def test_nlc_grid_fp32(self):
        '''nr_nlc_vxc on the NLC grid: FP32 rho and Vxc contractions vs FP64 (fixed density).'''
        mf2 = dft.RKS(mol, xc='wb97m-v').density_fit()
        mf2.grids.atom_grid = (30, 110)
        mf2.nlcgrids.atom_grid = (20, 50)
        mf2.max_cycle = 3
        mf2.kernel()
        dm = mf2.make_rdm1()
        mf2.nlcgrids.build()
        res = {}
        for prec in ('fp64', 'fp32'):
            old = numint.XC_PRECISION
            numint.XC_PRECISION = prec
            try:
                res[prec] = mf2._numint.nr_nlc_vxc(mol, mf2.nlcgrids, mf2.xc, dm)
            finally:
                numint.XC_PRECISION = old
        (n0, e0, v0), (n1, e1, v1) = res['fp64'], res['fp32']
        dv = float(cupy.abs(v1 - v0).max())
        print(f'[nlc fp32] dN={float(n1 - n0):.2e} dE={e1 - e0:.2e} max|dV|={dv:.2e}')
        self.assertLess(abs(e1 - e0), 1e-6)
        self.assertLess(dv, 1e-6)


if __name__ == "__main__":
    print("Full Tests for XC fp32")
    unittest.main()
