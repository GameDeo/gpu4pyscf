# Copyright 2025 The PySCF Developers. All Rights Reserved.
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

import unittest
import numpy as np
import cupy
import pyscf
from gpu4pyscf import dft
from gpu4pyscf.dft import numint
from gpu4pyscf.grad import rks as rks_grad
from gpu4pyscf.lib.cupy_helper import tag_array

atom = '''
O  -1.551007  -0.114520   0.000000
H  -1.934259   0.762503   0.000000
H  -0.599677   0.040712   0.000000
O   1.350625   0.111469   0.000000
H   1.680398  -0.373741  -0.758561
H   1.680398  -0.373741   0.758561
'''


def _rks(mol):
    mf = dft.RKS(mol, xc='wb97m-v').density_fit()
    mf.grids.atom_grid = (75, 302)
    mf.nlcgrids.atom_grid = (50, 194)
    mf.conv_tol = 1e-11
    mf.conv_tol_grad = 1e-7
    return mf


def setUpModule():
    global mol, mf_ref, e_ref, g_ref
    mol = pyscf.M(atom=atom, basis='def2-svp', verbose=0)
    mf_ref = _rks(mol)
    e_ref = mf_ref.kernel()
    g_ref = mf_ref.nuc_grad_method().kernel()


def tearDownModule():
    global mol, mf_ref
    mol = mf_ref = None


class KnownValues(unittest.TestCase):
    def test_nr_rks_vco(self):
        ni = mf_ref._numint
        dm = tag_array(mf_ref.make_rdm1(), mo_coeff=mf_ref.mo_coeff, mo_occ=mf_ref.mo_occ)
        n, e, v = ni.nr_rks(mol, mf_ref.grids, mf_ref.xc, dm)
        n2, e2, vc, g = numint.nr_rks_vco(ni, mol, mf_ref.grids, mf_ref.xc, mf_ref.mo_coeff, mf_ref.mo_occ,
                                          with_grad=True)
        Co = mf_ref.mo_coeff[:, mf_ref.mo_occ > 0]
        self.assertAlmostEqual(float(abs(vc - v @ Co).max()), 0, 10)
        self.assertAlmostEqual(e2, e, 10)
        g_xc = rks_grad.get_exc(ni, mol, mf_ref.grids, mf_ref.xc, dm)[1] * 2
        self.assertAlmostEqual(float(abs(np.asarray(g) - np.asarray(g_xc)).max()), 0, 10)

    def test_multifidelity_energy_gradient(self):
        mf = _rks(mol)
        mf.multifidelity = {'conv_tol_grad': 1e-6}
        e = mf.kernel()
        self.assertTrue(mf.converged)
        self.assertAlmostEqual(e, e_ref, 8)
        self.assertLessEqual(mf.multifidelity_stats['cycles']['outer'], 5)
        g = mf.nuc_grad_method().kernel()
        self.assertAlmostEqual(float(abs(np.asarray(g) - np.asarray(g_ref)).max()), 0, 6)

    def test_multifidelity_pure_functional(self):
        # no exact exchange: zero K terms, staged XC grids and J only
        ref = dft.RKS(mol, xc='pbe').density_fit()
        ref.grids.atom_grid = (75, 302)
        ref.conv_tol = 1e-11
        ref.conv_tol_grad = 1e-7
        e0 = ref.kernel()
        mf = dft.RKS(mol, xc='pbe').density_fit()
        mf.grids.atom_grid = (75, 302)
        mf.conv_tol = 1e-10
        mf.multifidelity = True
        e = mf.kernel()
        self.assertTrue(mf.converged)
        self.assertAlmostEqual(e, e0, 8)
        g = mf.nuc_grad_method().kernel()
        g0 = ref.nuc_grad_method().kernel()
        self.assertAlmostEqual(abs(np.asarray(g) - np.asarray(g0)).max(), 0, 5)


if __name__ == '__main__':
    print('Full tests for the multi-fidelity SCF')
    unittest.main()
