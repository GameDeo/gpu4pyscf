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
import cupy as cp
import pyscf
from gpu4pyscf.df.int8_df import Int8DF
from gpu4pyscf import scf


def setUpModule():
    global mol, dm
    mol = pyscf.M(atom='''O 0 0 0; H 0 .76 .59; H 0 -.76 .59; O 3 0 0; H 3 .76 .59; H 3 -.76 .59''',
                  basis='def2-tzvp', verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis='def2-universal-jfit')
    mf.kernel()
    dm = cp.asarray(mf.make_rdm1())


def tearDownModule():
    global mol, dm
    del mol, dm


class KnownValues(unittest.TestCase):
    def test_j_host_matches_device_and_fp64(self):
        ref = scf.RHF(mol).density_fit(auxbasis='def2-universal-jfit').get_j(mol, dm)
        dev = Int8DF(mol, 'def2-universal-jfit', bits=32, host=False)
        dev.build()
        host = Int8DF(mol, 'def2-universal-jfit', bits=32, host=True)
        host.host_block_bytes = 1e5        # several streamed blocks: exercises the double buffering
        host.build()
        jd = dev.j(dm)
        jh = host.j(dm)
        self.assertLess(float(abs(jh - jd).max()), 1e-9 * float(abs(jd).max()))
        # FP32 storage: ~1e-7 relative to the FP64 DF J
        self.assertLess(float(abs(jd - ref).max()), 1e-5 * float(abs(ref).max()))


if __name__ == "__main__":
    print("Full Tests for the INT8/FP32-stored surrogate DF tensor")
    unittest.main()
