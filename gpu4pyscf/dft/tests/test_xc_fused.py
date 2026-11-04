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
The fused block kernels and merged GEMMs of the XC grid loops (numint.XC_FUSED) must reproduce the original
chains of small operations: same Vxc matrix, N_elec and Exc up to the summation order (~1e-14 relative).
Covers LDA/GGA/MGGA, the MO-based and the density-matrix rho path, blocks with screened AOs (a molecule
with two distant fragments) and the unrestricted and NLC-grid drivers, plus the FP32 contractions.
'''
import unittest
import numpy as np
import cupy
from pyscf import gto
from gpu4pyscf import dft
from gpu4pyscf.dft import numint
from gpu4pyscf.lib import cutensor as cutensor_lib

# compact molecule: every grid block touches all AOs
ATOM_COMPACT = '''O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692;
                  C 2.6 0.2 0.1; H 3.2 0.9 0.6; H 3.2 -0.5 -0.4; H 2.0 -0.3 0.9; H 2.0 0.9 -0.5'''
# two fragments 40 A apart: blocks around each fragment screen out the other fragment's AOs
ATOM_SPARSE = '''O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692;
                 O 0 0 40.1173; H 0 0.7572 39.5308; H 0 -0.7572 39.5308'''

mols = {}
mfs = {}


def _make(key, atom, xc, spin=0):
    mol = gto.M(atom=atom, basis='def2-svp', spin=spin, verbose=0)
    if spin:
        mf = dft.UKS(mol, xc=xc)
    else:
        mf = dft.RKS(mol, xc=xc)
    mf = mf.density_fit()
    mf.grids.atom_grid = (30, 110)
    mf.max_cycle = 2
    mf.conv_tol = 1e-4
    mf.kernel()
    mols[key] = mol
    mfs[key] = mf


def tearDownModule():
    mols.clear()
    mfs.clear()


def _with_flags(fn, **flags):
    '''run fn() with numint module flags temporarily set'''
    old = {k: getattr(numint, k) for k in flags}
    try:
        for k, v in flags.items():
            setattr(numint, k, v)
        return fn()
    finally:
        for k, v in old.items():
            setattr(numint, k, v)


def _rks(key, xc, fused, dm_only=False, **flags):
    mol, mf = mols[key], mfs[key]
    dm = mf.make_rdm1()
    if dm_only:
        dm = cupy.array(dm)   # plain array: no mo_coeff tag, density-matrix rho path
    return _with_flags(lambda: mf._numint.nr_rks(mol, mf.grids, xc, dm), XC_FUSED=fused, **flags)


def _rel(a, b):
    return float(cupy.abs(a - b).max()) / max(float(cupy.abs(b).max()), 1e-300)


class KnownValues(unittest.TestCase):
    def _check_rks(self, key, xc, dm_only=False, tol=1e-12, **flags):
        if key not in mfs:
            atom = ATOM_SPARSE if key == 'sparse' else ATOM_COMPACT
            _make(key, atom, xc)
        n0, e0, v0 = _rks(key, xc, False, dm_only, **flags)
        n1, e1, v1 = _rks(key, xc, True, dm_only, **flags)
        self.assertLess(_rel(v1, v0), tol)
        self.assertLess(abs(float(n1 - n0)), tol * 10)
        self.assertLess(abs(e1 - e0), tol * max(abs(e0), 1.) * 10)

    def test_lda(self):
        self._check_rks('compact', 'lda,vwn')

    def test_gga(self):
        self._check_rks('compact', 'pbe')

    def test_mgga(self):
        self._check_rks('compact', 'tpss')

    def test_mgga_not_merged(self):
        self._check_rks('compact', 'tpss', XC_MERGE_VMAT_GEMM=False)

    def test_mgga_dm_path(self):
        self._check_rks('compact', 'tpss', dm_only=True)

    def test_sparse_blocks_gga(self):
        self._check_rks('sparse', 'pbe')

    def test_sparse_blocks_mgga(self):
        self._check_rks('sparse', 'tpss')

    def test_sparse_blocks_actually_screened(self):
        # the sparse molecule must contain blocks that do not touch all AOs (otherwise the test is void)
        self._check_rks('sparse', 'pbe')
        mol, mf = mols['sparse'], mfs['sparse']
        opt = mf._numint.gdftopt
        sizes = {len(x[1]) for x in mf.grids.get_non0ao_idx(opt)}
        self.assertTrue(any(n < opt._sorted_mol.nao for n in sizes))

    def test_uks_mgga(self):
        if 'uks' not in mfs:
            _make('uks', 'O 0 0 0; H 0 0.9 0.4; H 0 -0.9 0.4; H 3 0 0', 'tpss', spin=1)
        mol, mf = mols['uks'], mfs['uks']
        dm = mf.make_rdm1()
        ni = mf._numint
        n0, e0, v0 = _with_flags(lambda: ni.nr_uks(mol, mf.grids, 'tpss', dm), XC_FUSED=False)
        n1, e1, v1 = _with_flags(lambda: ni.nr_uks(mol, mf.grids, 'tpss', dm), XC_FUSED=True)
        self.assertLess(_rel(v1, v0), 1e-12)
        self.assertLess(float(abs(e1 - e0)), 1e-11)

    def test_nlc_grid(self):
        mol = gto.M(atom=ATOM_SPARSE, basis='def2-svp', verbose=0)
        mf = dft.RKS(mol, xc='wb97m-v').density_fit()
        mf.grids.atom_grid = (30, 110)
        mf.nlcgrids.atom_grid = (20, 50)
        mf.max_cycle = 2
        mf.conv_tol = 1e-4
        mf.kernel()
        dm = mf.make_rdm1()
        mf.nlcgrids.build()
        ni = mf._numint
        n0, e0, v0 = _with_flags(lambda: ni.nr_nlc_vxc(mol, mf.nlcgrids, mf.xc, dm), XC_FUSED=False)
        n1, e1, v1 = _with_flags(lambda: ni.nr_nlc_vxc(mol, mf.nlcgrids, mf.xc, dm), XC_FUSED=True)
        self.assertLess(_rel(v1, v0), 1e-11)
        self.assertLess(abs(e1 - e0), 1e-11)

    def test_fp32_contractions(self):
        for key, xc in (('compact', 'tpss'), ('sparse', 'pbe'), ('compact', 'lda,vwn')):
            if key not in mfs:
                _make(key, ATOM_SPARSE if key == 'sparse' else ATOM_COMPACT, xc)
            n0, e0, v0 = _rks(key, xc, False, XC_PRECISION='fp32')
            n1, e1, v1 = _rks(key, xc, True, XC_PRECISION='fp32')
            # both run FP32 contractions with different (block-wise vs per-op) accumulation: FP32 level
            self.assertLess(_rel(v1, v0), 2e-5)
            self.assertLess(abs(e1 - e0), 1e-5)
            _, _, v64 = _rks(key, xc, True)
            self.assertLess(_rel(v1, v64), 2e-5)

    def _check_grad(self, key, xc, dense_dm=False):
        from gpu4pyscf.grad import rks as rks_grad
        from gpu4pyscf.lib.cupy_helper import tag_array
        if key not in mfs:
            _make(key, ATOM_SPARSE if key == 'sparse' else ATOM_COMPACT, 'pbe')
        mol, mf = mols[key], mfs[key]
        dm = mf.make_rdm1()
        if dense_dm:
            # dm != C C^T: forces the dense density-matrix branch of the gradient
            dm = tag_array(dm + 1e-3 * cupy.ones_like(dm), mo_coeff=dm.mo_coeff, mo_occ=dm.mo_occ)
        ni = mf._numint
        g0 = _with_flags(lambda: rks_grad.get_exc(ni, mol, mf.grids, xc, dm)[1], XC_FUSED=False)
        g1 = _with_flags(lambda: rks_grad.get_exc(ni, mol, mf.grids, xc, dm)[1], XC_FUSED=True)
        self.assertLess(float(cupy.abs(cupy.asarray(g1) - cupy.asarray(g0)).max()), 1e-11)

    def test_grad_lda(self):
        self._check_grad('compact', 'lda,vwn')

    def test_grad_gga(self):
        self._check_grad('compact', 'pbe')

    def test_grad_mgga(self):
        self._check_grad('compact', 'tpss')

    def test_grad_sparse_blocks_mgga(self):
        self._check_grad('sparse', 'tpss')

    def test_grad_dense_dm_mgga(self):
        self._check_grad('compact', 'tpss', dense_dm=True)

    @unittest.skipIf(cutensor_lib.cutensor is None, 'cuTENSOR not available')
    def test_deterministic(self):
        '''two runs of the fused loop are bit-identical (fixed reduction order)'''
        if 'compact' not in mfs:
            _make('compact', ATOM_COMPACT, 'tpss')
        _, _, va = _rks('compact', 'tpss', True)
        _, _, vb = _rks('compact', 'tpss', True)
        self.assertTrue(bool(cupy.all(va == vb)))

    def test_fused_kernels_vs_numpy(self):
        from gpu4pyscf.dft import xc_fused
        rng = np.random.default_rng(7)
        ng, nao, nocc = 301, 17, 5
        for dtype, tol in ((np.float64, 1e-13), (np.float32, 2e-5)):
            c = rng.standard_normal((4, nocc, ng)).astype(dtype)
            rho = cupy.zeros((5, 2 * ng))
            xc_fused.rho_from_c(rho, ng // 2, cupy.asarray(c), ng, nocc, 'MGGA')
            ref = np.empty((5, ng))
            ref[0] = (c[0] * c[0]).sum(0)
            for i in range(1, 4):
                ref[i] = 2 * (c[0] * c[i]).sum(0)
            ref[4] = .5 * (c[1:] ** 2).sum((0, 1))
            self.assertLess(np.abs(rho.get()[:, ng//2:ng//2+ng] - ref).max(), tol * 10)
            self.assertEqual(float(abs(rho[:, :ng//2]).max()), 0.)

            ao = rng.standard_normal((4, nao, ng)).astype(dtype)
            wv = rng.standard_normal((5, 2 * ng)).astype(dtype)
            B = cupy.zeros((4, nao, ng), dtype=dtype)
            xc_fused.build_B(B, cupy.asarray(ao), cupy.asarray(wv), 11, ng, nao, 'MGGA')
            w = wv[:, 11:11 + ng]
            ref0 = np.einsum('xig,xg->ig', ao, w[:4])
            self.assertLess(np.abs(B[0].get() - ref0).max(), tol * 10)
            for x in range(1, 4):
                self.assertLess(np.abs(B[x].get() - w[4] * ao[x]).max(), tol * 10)


if __name__ == "__main__":
    print("Full Tests for the fused XC grid loops")
    unittest.main()
