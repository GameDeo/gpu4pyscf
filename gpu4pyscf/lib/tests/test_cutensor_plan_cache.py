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

'''The cuTENSOR plan cache must be bit-exact, also for views with different pointer alignment.'''
import unittest
import numpy as np
import cupy
from gpu4pyscf.lib import cutensor as ct


@unittest.skipIf(ct.cutensor is None, 'cuTENSOR not available')
class KnownValues(unittest.TestCase):
    def _run(self, pattern, a, b, out_shape, dtype):
        res = []
        for cache in (False, True, True):   # uncached, cache miss, cache hit
            old = ct.PLAN_CACHE
            ct.PLAN_CACHE = cache
            try:
                out = cupy.zeros(out_shape, dtype=dtype)
                res.append(ct.contraction(pattern, a, b, 1.0, 0.0, out=out).get())
            finally:
                ct.PLAN_CACHE = old
        return res

    def test_bit_exact(self):
        rng = np.random.default_rng(1)
        for dtype in (np.float64, np.float32):
            a = cupy.asarray(rng.random((37, 211)), dtype=dtype)
            b = cupy.asarray(rng.random((211, 29)), dtype=dtype)
            r0, r1, r2 = self._run('ik,kj->ij', a, b, (37, 29), dtype)
            self.assertTrue(np.array_equal(r0, r1) and np.array_equal(r1, r2))
            self.assertLess(np.abs(r2 - a.get() @ b.get()).max(), 1e-3 if dtype == np.float32 else 1e-10)

    def test_misaligned_views(self):
        rng = np.random.default_rng(2)
        big = cupy.asarray(rng.random(4 * 37 * 211 + 8))
        a = big[1:1 + 37 * 211].reshape(37, 211)          # 8-byte aligned only
        b = cupy.asarray(rng.random((211, 29)))
        r0, r1, r2 = self._run('ik,kj->ij', a, b, (37, 29), np.float64)
        self.assertTrue(np.array_equal(r0, r1) and np.array_equal(r1, r2))
        a2 = cupy.ascontiguousarray(big[:37 * 211].reshape(37, 211))   # aligned
        r0, r1, r2 = self._run('ik,kj->ij', a2, b, (37, 29), np.float64)
        self.assertTrue(np.array_equal(r0, r1) and np.array_equal(r1, r2))


if __name__ == "__main__":
    print("Full Tests for the cuTENSOR plan cache")
    unittest.main()
