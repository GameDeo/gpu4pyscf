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
    '''GPUs with FP64 tensor cores (sm_80+: A100/H100) run FP64 GEMMs as fast as FP32 SIMT, and the only
    reduced-precision GEMM path that exists here (cuTENSOR's "3xTF32") is plain FP32 SIMT, which is slower
    than FP64 there (measured).  Keep K / XC GEMMs in FP64 on those GPUs and use FP32 only where FP64 is
    slow (Turing, consumer and L-series parts).  VV10 stays FP32 everywhere: it is SIMT-bound.'''
    try:
        major = cupy.cuda.Device().compute_capability
        return 'fp64' if int(major) >= 80 else 'fp32'
    except Exception:
        return 'fp32'


DEFAULTS = {
    'k_precision': None,          # DF-K GEMMs: 'fp32' | 'tf32x3'; None -> auto by GPU generation
    'xc_precision': None,         # XC grid GEMMs (rho, vmat): same choices
    'vv10_precision': 'fp32',
    'conv_tol_grad': 3e-4,        # stage-1 orbital gradient threshold: the only real criterion
    'conv_tol': 1e-3,             # stage-1 energy threshold (FP32 noise in E is ~1e-5..1e-4)
    'max_cycle': 30,
    # Stage-2 (FP64) gradient threshold = final_grad_factor * (usual threshold).  A staged run
    # reaches the usual threshold with a larger residual than a pure FP64 DIIS path, which showed
    # up as 3.4x larger gradient errors; tightening by 0.1 gives gradients ~5x closer to a |g|<1e-7 reference than plain FP64 at default tolerance.
    'final_grad_factor': 0.1,
    'stage1b': True,              # FP64-K / reduced-XC polish between stage 1 and the FP64 stage
    'max_cycle_1b': 15,
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


class _Plateau(Exception):
    '''Raised from the SCF callback when |g| stops improving (reduced-precision noise floor).'''


def _plateau_callback(target, window=3, rel=0.7):
    '''Callback that remembers the latest orbitals and aborts the cycle loop once |g| has not improved
    by at least `rel` over the last `window` cycles while still above `target`.'''
    hist = []
    state = {}

    def cb(env):
        g = float(env['norm_gorb'])
        hist.append(g)
        state['mo_coeff'] = env['mo_coeff']
        state['mo_occ'] = env['mo_occ']
        state['e_tot'] = env['e_tot']
        state['cycle'] = env['cycle'] + 1
        if g > target and len(hist) > window and g > rel * min(hist[-window - 1:-1]):
            raise _Plateau
    return cb, state


def _run_stage(mf, dm0, conv_tol, conv_tol_grad, max_cycle, callback=None, **kwargs):
    from gpu4pyscf.scf import hf
    saved = mf.max_cycle
    saved_tol = mf.conv_tol
    try:
        mf.max_cycle = max_cycle
        mf.conv_tol = conv_tol  # _kernel reads mf.conv_tol and ignores its conv_tol argument
        return hf._kernel(mf, conv_tol, conv_tol_grad, dm0=dm0, callback=callback,
                          conv_check=False, **kwargs)
    finally:
        mf.max_cycle = saved
        mf.conv_tol = saved_tol


def run_fast_stage(mf, dm0, **kwargs):
    '''Run the reduced-precision stages and return the density matrix (tagged with mo_coeff/mo_occ)
    to start the final FP64 stage from.

    stage 1  : K, XC, VV10 all reduced precision, down to |g| < conv_tol_grad (K's FP32 noise floor
               is |g| ~ 1e-4).
    stage 1b : K in FP64, XC (and VV10) still reduced, down to the final FP64-stage threshold or until
               |g| plateaus.  FP32 XC alone has a noise floor |g| ~ 1e-5 (measured, Bisphenol A /
               def2-TZVP), well below the final threshold, so the expensive FP64 XC is only needed
               for the last few cycles that fix the energy.  Skipped when XC is FP64 anyway.
    '''
    log = logger.new_logger(mf)
    opt = get_options(mf)
    with fast_precision(opt):
        log.info('Mixed precision SCF, stage 1: K=%s XC=%s VV10=%s conv_tol_grad=%g',
                 opt['k_precision'], opt['xc_precision'], opt['vv10_precision'], opt['conv_tol_grad'])
        conv, e_tot, mo_energy, mo_coeff, mo_occ = _run_stage(
            mf, dm0, opt['conv_tol'], opt['conv_tol_grad'], min(mf.max_cycle, opt['max_cycle']), **kwargs)
    mf.cycles_stage1 = getattr(mf, 'cycles', -1)
    log.info('Mixed precision SCF, stage 1 done: E=%.12g converged=%s cycles=%s',
             e_tot, conv, mf.cycles_stage1)
    dm = tag_array(mf.make_rdm1(mo_coeff, mo_occ), mo_coeff=mo_coeff, mo_occ=mo_occ)

    mf.cycles_stage1b = 0
    if opt['xc_precision'] != 'fp64' and opt['stage1b'] and mf.max_cycle > 0:
        target = final_conv_tol_grad(mf)
        opt1b = dict(opt, k_precision='fp64')
        cb, state = _plateau_callback(target)
        with fast_precision(opt1b):
            log.info('Mixed precision SCF, stage 1b: K=fp64 XC=%s VV10=%s conv_tol_grad=%g',
                     opt['xc_precision'], opt['vv10_precision'], target)
            try:
                conv, e_tot, mo_energy, mo_coeff, mo_occ = _run_stage(
                    mf, dm, 1e-6, target, min(mf.max_cycle, opt['max_cycle_1b']), callback=cb, **kwargs)
            except _Plateau:
                log.info('Mixed precision SCF, stage 1b: |g| plateau reached')
                mo_coeff, mo_occ, e_tot = state['mo_coeff'], state['mo_occ'], state['e_tot']
            mf.cycles_stage1b = state.get('cycle', getattr(mf, 'cycles', -1))
        log.info('Mixed precision SCF, stage 1b done: E=%.12g cycles=%s', e_tot, mf.cycles_stage1b)
        dm = tag_array(mf.make_rdm1(mo_coeff, mo_occ), mo_coeff=mo_coeff, mo_occ=mo_occ)
    return dm


def final_conv_tol_grad(mf):
    '''Gradient threshold for the FP64 stage of a staged run.'''
    opt = get_options(mf)
    tol = mf.conv_tol_grad
    if tol is None:
        tol = mf.conv_tol**.5
    return tol * opt['final_grad_factor']
