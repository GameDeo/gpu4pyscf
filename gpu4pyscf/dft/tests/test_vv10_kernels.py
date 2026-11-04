# Copyright 2021-2025 The PySCF Developers. All Rights Reserved.
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
from gpu4pyscf.dft import numint, vv10_kernels

# wB97M-V non-local parameters (b, C)
NLC_PARS = (6.0, 0.01)


def _reference_UWE(xyz, rw, omega, kappa):
    '''Dense FP64 evaluation of the formulas in lib/gdft/vv10.cu (chunked over i).'''
    n = xyz.shape[0]
    U = cupy.empty(n); W = cupy.empty(n); E = cupy.empty(n)
    for i0 in range(0, n, 256):
        sl = slice(i0, min(n, i0 + 256))
        d = xyz[sl, None, :] - xyz[None, :, :]
        r2 = (d * d).sum(axis=2)
        g_ij = omega[sl, None] * r2 + kappa[sl, None]
        g_ji = omega[None, :] * r2 + kappa[None, :]
        g_sum = g_ij + g_ji
        inv = 1.0 / (g_ij * g_ji * g_sum)
        E_ij = -rw[None, :] * inv
        U_ij = E_ij * (g_sum + g_ij) * g_ji * inv
        U[sl] = -1.5 * U_ij.sum(axis=1)
        W[sl] = -1.5 * (U_ij * r2).sum(axis=1)
        E[sl] = 1.5 * E_ij.sum(axis=1)
    return U, W, E


def _random_inputs(n, seed=0, spread=12.0):
    rng = np.random.default_rng(seed)
    xyz = cupy.asarray(rng.normal(size=(n, 3)) * spread / 2)
    rho = cupy.asarray(np.exp(rng.uniform(-9, 0, n)))
    omega = cupy.asarray(rng.uniform(0.02, 40, n))
    kappa = cupy.asarray(rng.uniform(0.3, 40, n))
    w = cupy.asarray(rng.uniform(1e-4, 1e-1, n))
    return xyz, rho * w, omega, kappa


def _weighted_rel(a, b, rw):
    return float((rw * cupy.abs(a - b)).sum() / (rw * cupy.abs(b)).sum())


class KnownValues(unittest.TestCase):
    def test_sym_fp64_vs_reference(self):
        # n is not a multiple of the tile size; several tile counts (odd and even)
        for n, nt, mi in ((1000, 128, 4), (2500, 64, 4), (4096 + 77, 128, 2)):
            xyz, rw, omega, kappa = _random_inputs(n, seed=n)
            ref = _reference_UWE(xyz, rw, omega, kappa)
            for sym in (True, False):
                out = vv10_kernels.eval_UWE(xyz, rw, omega, kappa, precision='fp64',
                                            symmetric=sym, nthreads=nt, mi=mi)
                for a, b in zip(out, ref):
                    assert float(cupy.abs(a / b - 1).max()) < 1e-10

    def test_sym_fp32_vs_reference(self):
        xyz, rw, omega, kappa = _random_inputs(3000, seed=3)
        ref = _reference_UWE(xyz, rw, omega, kappa)
        for sym in (True, False):
            out = vv10_kernels.eval_UWE(xyz, rw, omega, kappa, precision='fp32', symmetric=sym)
            for a, b in zip(out, ref):
                assert _weighted_rel(a, b, rw) < 2e-6

    def test_rcut_skips_far_tiles(self):
        # points on two far apart clusters; rcut drops the inter-cluster tile pairs
        xyz, rw, omega, kappa = _random_inputs(2048, seed=5, spread=4.0)
        xyz[1024:, 0] += 5000.0
        ref = _reference_UWE(xyz, rw, omega, kappa)
        out = vv10_kernels.eval_UWE(xyz, rw, omega, kappa, precision='fp64', rcut=100.0)
        for a, b in zip(out, ref):
            assert float(cupy.abs(a / b - 1).max()) < 1e-10

    def test_nlc_vxc_kernel_switch(self):
        # numint._vv10nlc with the symmetric kernel against the libgdft kernel
        mol = pyscf.M(atom='O 0 0 0.1; H 0 0.75 -0.47; H 0 -0.75 -0.47; He 3.0 0.5 0.2',
                      basis='6-31g', verbose=0)
        from gpu4pyscf.dft import Grids
        from gpu4pyscf.scf import hf
        grids = Grids(mol)
        grids.atom_grid = (30, 110)
        grids.build()
        dm = hf.RHF(mol).get_init_guess(mol, 'minao')
        ni = numint.NumInt()
        rho = ni.get_rho(mol, dm, grids)  # LDA density only: add a gradient from finite differences
        # use a smooth artificial gradient so that the GGA-dependent omega is non-trivial
        rng = np.random.default_rng(1)
        drho = cupy.asarray(rng.normal(size=(3, rho.size))) * rho * 0.5
        rho_drho = cupy.vstack([rho[None], drho])
        old = (numint.VV10_KERNEL, numint.VV10_PRECISION)
        try:
            numint.VV10_KERNEL = 'libgdft'
            e0, v0 = numint._vv10nlc(rho_drho, grids.coords, grids.weights, NLC_PARS)
            numint.VV10_KERNEL, numint.VV10_PRECISION = 'sym', 'fp64'
            e1, v1 = numint._vv10nlc(rho_drho, grids.coords, grids.weights, NLC_PARS)
            numint.VV10_PRECISION = 'fp32'
            e2, v2 = numint._vv10nlc(rho_drho, grids.coords, grids.weights, NLC_PARS)
        finally:
            numint.VV10_KERNEL, numint.VV10_PRECISION = old
        w = grids.weights * rho
        assert abs(float((w * (e1 - e0)).sum())) < 1e-10
        assert float(cupy.abs(v1 - v0).max()) < 1e-10 * float(cupy.abs(v0).max()) + 1e-12
        assert abs(float((w * (e2 - e0)).sum()) / float((w * e0).sum())) < 1e-6
        assert _weighted_rel(v2[0], v0[0], w) < 1e-5


if __name__ == "__main__":
    print("Full Tests for the VV10 pair kernels")
    unittest.main()
