#!/usr/bin/env python
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
Regression test for the mixed-precision (FP32 pair arithmetic) VV10 kernel.

The FP32 kernel must reproduce the FP64 non-local correlation energy and potential
on a *fixed* density.  This is the quantity that enters the Fock matrix, so this test is
stricter than comparing converged SCF energies (which are variational in the density).
'''
import unittest
import numpy as np
import cupy
import pyscf
from pyscf import gto
from gpu4pyscf import dft
from gpu4pyscf.dft import numint


def setUpModule():
    global mol, mf, dm
    mol = gto.M(
        atom='''
        O  0.0000  0.0000  0.1173
        H  0.0000  0.7572 -0.4692
        H  0.0000 -0.7572 -0.4692
        C  2.6000  0.2000  0.1000
        H  3.2000  0.9000  0.6000
        H  3.2000 -0.5000 -0.4000
        H  2.0000 -0.3000  0.9000
        H  2.0000  0.9000 -0.5000
        ''',
        basis='def2-svp', verbose=0)
    mf = dft.RKS(mol, xc='wb97m-v').density_fit()
    mf.grids.atom_grid = (50, 194)
    mf.nlcgrids.atom_grid = (30, 110)
    mf.max_cycle = 3        # the density needs to be reasonable but not converged
    mf.kernel()
    dm = mf.make_rdm1()


def tearDownModule():
    global mol, mf, dm
    del mol, mf, dm


def _nlc(precision):
    ni = mf._numint
    old = numint.VV10_PRECISION
    numint.VV10_PRECISION = precision
    try:
        return ni.nr_nlc_vxc(mol, mf.nlcgrids, mf.xc, dm)
    finally:
        numint.VV10_PRECISION = old


class KnownValues(unittest.TestCase):
    def test_vv10_fp32_fixed_density(self):
        n64, e64, v64 = _nlc('fp64')
        n32, e32, v32 = _nlc('fp32')
        print(f'[vv10 fixed density] E64={e64:.12f} dE={e64 - e32:.2e} '
              f'max|dV|={float(cupy.abs(v64 - v32).max()):.2e} (max|V|={float(cupy.abs(v64).max()):.2e})')
        self.assertAlmostEqual(float(n64), float(n32), 12)
        # energy: relative error of the pair sum must stay at the FP32-noise/sqrt(N) level
        self.assertLess(abs(e64 - e32), 1e-8)
        # potential matrix
        self.assertLess(float(cupy.abs(v64 - v32).max()), 1e-7)

    def test_vv10_fp32_scalar_kernels(self):
        '''Compare U, W, E from the two raw kernels on a random point cloud.'''
        import ctypes
        libgdft = numint.libgdft
        rng = np.random.default_rng(7)
        n = 3000
        # a molecule-like cloud far from the origin to stress the (hi, lo) coordinate split
        coords = cupy.asarray(rng.normal(size=(n, 3)) * 4.0 + np.array([20., -35., 12.]))
        rho = cupy.asarray(10 ** rng.uniform(-8, 1, size=n))
        w = cupy.asarray(rng.uniform(1e-4, 1., size=n))
        omega = cupy.asarray(10 ** rng.uniform(-1, 2, size=n))
        kappa = cupy.asarray(0.5 * rho ** (1. / 6) + 0.01)
        rw = rho * w
        stream = cupy.cuda.get_current_stream()

        def run(fn):
            U = cupy.empty(n); W = cupy.empty(n); E = cupy.empty(n)
            err = fn(ctypes.cast(stream.ptr, ctypes.c_void_p),
                     *[ctypes.cast(x.data.ptr, ctypes.c_void_p) for x in (U, W, E, coords, rw, omega, kappa)],
                     ctypes.c_int(n))
            self.assertEqual(err, 0)
            return U, W, E

        r64 = run(libgdft.VXC_vv10nlc_fock_eval_UWE)
        r32 = run(libgdft.VXC_vv10nlc_fock_eval_UWE_fp32)
        for a, b, name in zip(r64, r32, 'UWE'):
            rel = float(cupy.abs(a - b).max() / cupy.abs(a).max())
            print(f'[vv10 kernel {name}] max rel dev = {rel:.2e}')
            self.assertLess(rel, 5e-6, msg=f'{name}: max rel dev {rel:.2e}')
            # the sums are dominated by many same-signed terms, so the *typical* error is far smaller
            rel_med = float(cupy.median(cupy.abs(a - b) / cupy.abs(a)))
            self.assertLess(rel_med, 1e-6, msg=f'{name}: median rel dev {rel_med:.2e}')


if __name__ == "__main__":
    print("Full Tests for VV10 FP32 kernel")
    unittest.main()
