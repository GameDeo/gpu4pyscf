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
Staged mixed-precision SCF ("precision-adaptive" SCF with FP64 polishing).

Stage 1 runs the SCF with the expensive pieces of the Fock build in reduced precision
(FP32 / 3xTF32 tensor-core DF-K, FP32 VV10 pair kernel, ...) down to a loose gradient
threshold.  The orbital gradient of an FP32 Fock matrix cannot go below its noise floor
(~1e-5 for DF-K in FP32), so stage 1 is stopped well above it.

Stage 2 is the ordinary FP64 SCF, restarted from the stage-1 density with a fresh DIIS
space.  All reported quantities (energy, orbitals, and anything derived from them such as
nuclear gradients) therefore come from FP64 Fock builds and are converged to the same
thresholds as a pure FP64 run; stage 1 only supplies a very good starting density so that
few (typically 3-5) FP64 iterations are left.

Enable with::

    mf.mixed_precision = True            # or a dict of options, see DEFAULTS
'''
import contextlib
import cupy
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import tag_array

def _default_gemm_precision():
    '''3xTF32 tensor-core emulation of FP32 needs Ampere (sm_80) or newer; otherwise plain FP32.'''
    try:
        major = cupy.cuda.Device().compute_capability
        return 'tf32x3' if int(major) >= 80 else 'fp32'
    except Exception:
        return 'fp32'


DEFAULTS = {
    'k_precision': None,          # DF-K GEMMs: 'fp32' | 'tf32x3'; None -> auto by GPU generation
    'xc_precision': None,         # XC grid GEMMs (rho, vmat): same choices
    'vv10_precision': 'fp32',
    'conv_tol_grad': 3e-4,        # stage-1 orbital gradient threshold: the only real criterion
    'conv_tol': 1e-3,             # stage-1 energy threshold (FP32 noise in E is ~1e-5..1e-4)
    'max_cycle': 30,
}


def get_options(mf):
    opt = dict(DEFAULTS)
    cfg = getattr(mf, 'mixed_precision', None)
    if isinstance(cfg, dict):
        opt.update(cfg)
    for key in ('k_precision', 'xc_precision'):
        if opt[key] is None:
            opt[key] = _default_gemm_precision()
    return opt


@contextlib.contextmanager
def fast_precision(opt):
    '''Temporarily switch the reduced-precision kernels on.'''
    from gpu4pyscf.dft import numint
    from gpu4pyscf.df import df_jk
    old = (numint.VV10_PRECISION, df_jk.K_PRECISION, numint.XC_PRECISION)
    numint.VV10_PRECISION = opt['vv10_precision']
    df_jk.K_PRECISION = opt['k_precision']
    numint.XC_PRECISION = opt['xc_precision']
    try:
        yield
    finally:
        numint.VV10_PRECISION, df_jk.K_PRECISION, numint.XC_PRECISION = old


def run_fast_stage(mf, dm0, **kwargs):
    '''Run stage 1 and return the density matrix (tagged with mo_coeff/mo_occ) to
    start the FP64 stage from.'''
    from gpu4pyscf.scf import hf
    log = logger.new_logger(mf)
    opt = get_options(mf)
    saved = mf.max_cycle
    try:
        mf.max_cycle = min(saved, opt['max_cycle'])
        with fast_precision(opt):
            log.info('Mixed precision SCF, stage 1: K=%s XC=%s VV10=%s conv_tol_grad=%g',
                     opt['k_precision'], opt['xc_precision'], opt['vv10_precision'],
                     opt['conv_tol_grad'])
            conv, e_tot, mo_energy, mo_coeff, mo_occ = hf._kernel(
                mf, opt['conv_tol'], opt['conv_tol_grad'], dm0=dm0,
                callback=None, conv_check=False, **kwargs)
    finally:
        mf.max_cycle = saved
    mf.cycles_stage1 = getattr(mf, 'cycles', -1)
    log.info('Mixed precision SCF, stage 1 done: E=%.12g converged=%s cycles=%s',
             e_tot, conv, mf.cycles_stage1)
    dm = mf.make_rdm1(mo_coeff, mo_occ)
    return tag_array(dm, mo_coeff=mo_coeff, mo_occ=mo_occ)
