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
Compatibility of the staged mixed-precision SCF (gpu4pyscf.scf.mixed_precision, schemes 'crt' and 'classic') with the
implicit-solvent wrappers: energies and gradients must agree with the plain FP64 run to the staged-SCF accuracy, for
RKS, UKS, a charged radical, an ECP and SMD, and the opt-in coarse solvent surface must not change the final answer.
(GPU tests, not run on the machine where they were written.  The 'crt' scheme needs INT8 tensor cores: sm_75 or newer.)
'''
import unittest
import numpy
import cupy
from pyscf import gto
from gpu4pyscf import dft
from gpu4pyscf.solvent import pcm, smd

def _cc():
    try:
        return int(cupy.cuda.Device().compute_capability)
    except Exception:
        return 0

METHANOL = '''
C  -0.0470  0.6590  0.0000
O  -0.0470 -0.7630  0.0000
H  -1.0870  0.9840  0.0000
H   0.4580  1.0640  0.8890
H   0.4580  1.0640 -0.8890
H   0.8610 -1.0990  0.0000'''

def _build(atom=METHANOL, basis='def2-svp', xc='pbe0', solvent='pcm', unrestricted=False, **molkw):
    mol = gto.M(atom=atom, basis=basis, verbose=0, **molkw)
    cls = dft.UKS if (unrestricted or mol.spin != 0) else dft.RKS
    mf = cls(mol, xc=xc).density_fit()
    mf.grids.atom_grid = (60, 302)
    mf.conv_tol = 1e-10
    if solvent == 'pcm':
        mf = mf.PCM()
        mf.with_solvent.method = 'IEF-PCM'
        mf.with_solvent.eps = 78.3553
        mf.with_solvent.lebedev_order = 29
    elif solvent == 'smd':
        mf = mf.SMD()
        mf.with_solvent.solvent = 'water'
    return mf

def _run(mf, mixed=None):
    if mixed is not None:
        mf.mixed_precision = mixed
    e = mf.kernel()
    assert mf.converged
    g = mf.Gradients().kernel()
    return e, g

@unittest.skipIf(pcm.libsolvent is None, "solvent extension not compiled")
@unittest.skipIf(_cc() < 75, "INT8 tensor cores (sm_75+) required")
class KnownValues(unittest.TestCase):
    ETOL = 5e-8     # staged SCF vs FP64 (the lead measured ~1e-10 on 50-atom systems)
    GTOL = 2e-5

    def _compare(self, factory, mixed):
        e0, g0 = _run(factory())
        e1, g1 = _run(factory(), mixed)
        assert abs(e1 - e0) < self.ETOL, (e1 - e0)
        assert abs(g1 - g0).max() < self.GTOL, abs(g1 - g0).max()
        return e0, e1

    def test_rks_pcm_crt(self):
        self._compare(lambda: _build(), {'scheme': 'crt'})

    def test_rks_pcm_classic(self):
        self._compare(lambda: _build(), {'scheme': 'classic'})

    def test_uks_charged_radical_pcm_crt(self):
        self._compare(lambda: _build(charge=1, spin=1), {'scheme': 'crt'})

    def test_rks_cpcm_ecp_crt(self):
        def factory():
            mf = _build(atom='Al 0 0 0; F 1.65 0 0; F -0.825 1.428941 0; F -0.825 -1.428941 0', basis='CRENBL',
                        xc='pbe', ecp='CRENBL')
            mf.with_solvent.method = 'C-PCM'
            mf.with_solvent.lebedev_order = 17
            return mf
        self._compare(factory, {'scheme': 'crt'})

    @unittest.skipIf(smd.libsolvent is None, "solvent extension not compiled")
    def test_rks_smd_crt(self):
        self._compare(lambda: _build(solvent='smd'), {'scheme': 'crt'})

    def test_coarse_solvent_surface_does_not_change_the_result(self):
        e0, g0 = _run(_build())
        mf = _build()
        e1, g1 = _run(mf, {'scheme': 'crt', 'coarse_solvent_lebedev': 17})
        # the user's solvent object is restored and was used by the final stage
        assert mf.with_solvent.lebedev_order == 29
        assert mf.with_solvent.surface['grid_coords'].shape[0] > 0
        assert abs(e1 - e0) < self.ETOL
        assert abs(g1 - g0).max() < self.GTOL

    def test_solvent_object_state_after_staged_scf(self):
        # the energy and Fock contribution stored on the solvent object belong to the final (full-grid, full-surface) density
        mf = _build()
        mf.mixed_precision = {'scheme': 'crt', 'coarse_solvent_lebedev': 17}
        mf.kernel()
        sol = mf.with_solvent
        dm = mf.make_rdm1()
        e, v = sol._get_vind(dm)
        assert abs(float(e) - float(mf.scf_summary['e_solvent'])) < 1e-6   # make_rdm1() differs from the last Fock density by O(|g|)
        assert sol._intermediates['q_sym'].shape[0] == sol.surface['grid_coords'].shape[0]

    def test_accept_converged_init_shortcut_with_solvent(self):
        # Restarting a converged run: the coarse stage and the hf._kernel `_accept_converged_init` shortcut (one exact Fock
        # build certifies the handed-over density) must return the solvated energy, solvent term included.
        mf = _build()
        mf.mixed_precision = {'scheme': 'crt'}
        e0 = mf.kernel()
        mf.max_cycle = 50
        e1 = mf.kernel(dm0=mf.make_rdm1())
        assert abs(e1 - e0) < 1e-8
        assert 'e_solvent' in mf.scf_summary and abs(mf.scf_summary['e_solvent']) > 1e-4

if __name__ == "__main__":
    print("Full Tests for the mixed-precision SCF with solvent")
    unittest.main()
