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
Tests of the solvent-response cache (self._intermediates['dm']), the single-solve C-PCM shortcut and the
DA-free left_multiply_R.  All checks are exact up to rounding: the optimizations must not change any number.
(GPU tests, not run on the machine where they were written.)
'''
import unittest
from unittest import mock
import numpy
import cupy
from pyscf import gto
from gpu4pyscf import scf, dft
from gpu4pyscf.solvent import pcm, smd

def setUpModule():
    global mol
    mol = gto.M(atom='''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000''', basis='6-31g', verbose=0)

def tearDownModule():
    global mol
    del mol

def _scf(method='IEF-PCM', xc='b3lyp', unrestricted=False, lowmem=False, lebedev_order=17):
    cm = pcm.PCM(mol)
    cm.eps = 35.9
    cm.verbose = 0
    cm.lebedev_order = lebedev_order
    cm.method = method
    cm.lowmem_intermediate_storage = lowmem
    cls = dft.UKS if unrestricted else dft.RKS
    mf = cls(mol, xc=xc).PCM(cm)
    mf.verbose = 0
    mf.conv_tol = 1e-11
    mf.kernel()
    return mf

@unittest.skipIf(pcm.libsolvent is None, "solvent extension not compiled")
class KnownValues(unittest.TestCase):
    def test_cache_written_by_scf(self):
        mf = _scf()
        sol = mf.with_solvent
        cached = sol._intermediates['dm']
        assert cached.shape == (mol.nao, mol.nao)
        assert cupy.linalg.norm(cached - mf.make_rdm1()) < 1e-9

    def test_gradient_skips_resolve_and_matches(self):
        for method in ['C-PCM', 'IEF-PCM', 'SS(V)PE']:
            mf = _scf(method)
            sol = mf.with_solvent
            with mock.patch.object(pcm.PCM, '_get_vind', autospec=True, side_effect=pcm.PCM._get_vind) as m:
                g_cached = mf.Gradients().kernel()
                # the converged density is the one the last Fock build used: no solvent solve in the gradient
                assert m.call_count == 0, f'{method}: {m.call_count} redundant _get_vind calls'
            # Same gradient when the cache is deliberately invalidated (old behaviour: three re-solves)
            sol._intermediates.pop('dm')
            with mock.patch.object(pcm.PCM, '_get_vind', autospec=True, side_effect=pcm.PCM._get_vind) as m:
                g_fresh = mf.Gradients().kernel()
                assert m.call_count == 1   # the first consumer re-solves once, the others reuse it
            assert abs(g_cached - g_fresh).max() < 1e-10, method

    def test_cache_miss_for_another_density(self):
        mf = _scf()
        sol = mf.with_solvent
        dm = mf.make_rdm1()
        dm2 = dm + 1e-3 * cupy.eye(mol.nao)
        with mock.patch.object(pcm.PCM, '_get_vind', autospec=True, side_effect=pcm.PCM._get_vind) as m:
            sol._ensure_vind(dm)
            assert m.call_count == 0
            sol._ensure_vind(dm2)
            assert m.call_count == 1
            assert cupy.linalg.norm(sol._intermediates['dm'] - dm2) < 1e-14
            sol._ensure_vind(dm2)
            assert m.call_count == 1
        # charges belong to dm2 now
        q_ref, _ = sol._get_qsym(dm2, with_nuc=True)
        assert abs(sol._intermediates['q_sym'] - q_ref).max() < 1e-10

    def test_cache_uks_total_density(self):
        mf = _scf(unrestricted=True)
        sol = mf.with_solvent
        dm = mf.make_rdm1()
        assert dm.ndim == 3
        assert cupy.linalg.norm(sol._intermediates['dm'] - (dm[0] + dm[1])) < 1e-9
        with mock.patch.object(pcm.PCM, '_get_vind', autospec=True, side_effect=pcm.PCM._get_vind) as m:
            sol._ensure_vind(dm)
            sol._ensure_vind(dm[0] + dm[1])
            assert m.call_count == 0

    def test_frozen_right_density_is_never_cached(self):
        mf = _scf()
        sol = mf.with_solvent
        dm = mf.make_rdm1()
        sol.frozen_dm0_for_finite_difference_without_response = dm * 0.99
        try:
            sol._get_vind(dm)
            assert 'dm' not in sol._intermediates
            with mock.patch.object(pcm.PCM, '_get_vind', autospec=True, side_effect=pcm.PCM._get_vind) as m:
                sol._ensure_vind(dm)
                assert m.call_count == 1
        finally:
            sol.frozen_dm0_for_finite_difference_without_response = None

    def test_vind_without_vmat(self):
        mf = _scf()
        sol = mf.with_solvent
        dm = mf.make_rdm1()
        e1, v1 = sol._get_vind(dm)
        e2, v2 = sol._get_vind(dm, with_vmat=False)
        assert v2 is None
        assert abs(float(e1) - float(e2)) < 1e-14

    def test_cpcm_single_solve_equals_symmetrized_pair(self):
        mf = _scf('C-PCM')
        sol = mf.with_solvent
        dm = mf.make_rdm1()
        v = sol._get_vgrids(dm, with_nuc=True)
        q_sym, q = sol._solve_q_sym(v)
        b = sol.left_multiply_R(v.T)
        qa = sol.left_solve_K(b).T
        vK_1 = sol.left_solve_K(v.T, K_transpose=True)
        qt = sol.left_multiply_R(vK_1, R_transpose=True).T
        assert abs(q_sym - (qa + qt) / 2).max() < 1e-10
        assert abs(q - qa).max() < 1e-12
        assert q_sym is not q   # callers may modify one in place

    def test_left_multiply_R_without_DA_matrix(self):
        for method in ['IEF-PCM', 'SS(V)PE']:
            mf = _scf(method)
            sol = mf.with_solvent
            D = sol._intermediates['D']
            A = sol._intermediates['A']
            f = sol._intermediates['f_epsilon']
            DA = D * A
            x2 = cupy.random.rand(A.shape[0], 1)
            x1 = x2[:, 0].copy()
            for x in (x1, x2):
                ref = -f * (x - 1.0 / (2.0 * numpy.pi) * cupy.dot(DA, x))
                refT = -f * (x - 1.0 / (2.0 * numpy.pi) * cupy.dot(DA.T, x))
                assert abs(sol.left_multiply_R(x) - ref).max() < 1e-12
                assert abs(sol.left_multiply_R(x, R_transpose=True) - refT).max() < 1e-12

    def test_lowmem_cpcm_builds_once(self):
        cm = pcm.PCM(mol)
        cm.eps = 35.9
        cm.verbose = 0
        cm.lebedev_order = 17
        cm.method = 'C-PCM'
        cm.lowmem_intermediate_storage = True
        mf = scf.RHF(mol).PCM(cm)
        mf.verbose = 0
        dm = mf.get_init_guess()
        with mock.patch.object(pcm.PCM, 'build', autospec=True, side_effect=pcm.PCM.build) as m:
            cm.reset()
            cm.kernel(dm)
            assert m.call_count == 1, m.call_count

    def test_smd_borrowed_methods_present(self):
        for name in ['_get_vind', '_ensure_vind', '_is_built', '_solve_q_sym']:
            assert getattr(smd.SMD, name) is getattr(pcm.PCM, name)

    def test_smd_cds_uses_atomic_numbers_for_ecp_atoms(self):
        m = gto.M(atom='I 0 0 0; C 0 0 2.1; H 1 0 2.5', basis='def2-svp', ecp='def2-svp', verbose=0)
        assert list(m.atom_charges()) == [25, 6, 1]          # effective charges Z - ncore
        sol = smd.SMD(m, solvent='water')
        z = smd._atomic_numbers(m)
        assert list(z) == [53, 6, 1]
        assert sol.mol is m

if __name__ == "__main__":
    print("Full Tests for the solvent-response cache")
    unittest.main()
