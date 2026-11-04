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
The contract-first XC gradient (grad.rks._exc1_block) is an exact re-association of the legacy
[3, nao, nao]-matrix path: gradients must agree to rounding error.
'''
import unittest
import numpy as np
from pyscf import gto
from gpu4pyscf import dft
from gpu4pyscf.grad import rks as rks_grad


def setUpModule():
    global mol
    mol = gto.M(
        atom='''O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692;
                C 2.6 0.2 0.1; H 3.2 0.9 0.6; H 3.2 -0.5 -0.4; H 2.0 -0.3 0.9; H 2.0 0.9 -0.5''',
        basis='def2-svp', verbose=0)


def tearDownModule():
    global mol
    del mol


def _grads(xc, density_fit):
    mf = dft.RKS(mol, xc=xc)
    if density_fit:
        mf = mf.density_fit()
    mf.grids.atom_grid = (50, 194)
    if mf._numint.libxc.is_nlc(mf.xc):
        mf.nlcgrids.atom_grid = (20, 50)
    mf.conv_tol = 1e-10
    mf.kernel()
    out = {}
    for legacy in (True, False):
        rks_grad.LEGACY_EXC_GRAD = legacy
        try:
            out[legacy] = mf.nuc_grad_method().kernel()
        finally:
            rks_grad.LEGACY_EXC_GRAD = False
    return out[True], out[False]


class KnownValues(unittest.TestCase):
    def _check(self, xc, density_fit=True):
        g0, g1 = _grads(xc, density_fit)
        d = float(np.abs(g1 - g0).max())
        print(f'[{xc}] max|dgrad| contract-first vs legacy = {d:.2e}')
        self.assertLess(d, 1e-9)

    def test_lda(self):
        self._check('lda,vwn')

    def test_gga(self):
        self._check('pbe')

    def test_mgga(self):
        self._check('tpss')

    def test_wb97mv(self):
        self._check('wb97m-v')

    def test_crt_gemms(self):
        '''XC-gradient GEMMs as INT8 residue GEMMs (needs sm_75+) vs FP64'''
        mf = dft.RKS(mol, xc='wb97m-v').density_fit()
        mf.grids.atom_grid = (50, 194)
        mf.nlcgrids.atom_grid = (20, 50)
        mf.conv_tol = 1e-10
        mf.kernel()
        g0 = mf.nuc_grad_method().kernel()
        rks_grad.XC_GRAD_CRT_MODULI = 12
        try:
            g1 = mf.nuc_grad_method().kernel()
        finally:
            rks_grad.XC_GRAD_CRT_MODULI = 0
        d = float(np.abs(g1 - g0).max())
        print(f'[wb97m-v] max|dgrad| CRT-12 vs FP64 = {d:.2e}')
        self.assertLess(d, 1e-8)


if __name__ == "__main__":
    print("Full Tests for contract-first XC gradient")
    unittest.main()
