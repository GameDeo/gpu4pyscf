# Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
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

import os
import threading
import numpy as np
import cupy
from gpu4pyscf.lib import logger

try:
    import cupy_backends.cuda.libs.cutensor  # NOQA
    from cupyx import cutensor
    from cupy_backends.cuda.libs import cutensor as cutensor_backend
    ALGO_DEFAULT = cutensor_backend.ALGO_DEFAULT
    OP_IDENTITY = cutensor_backend.OP_IDENTITY
    JIT_MODE_NONE = cutensor_backend.JIT_MODE_NONE
    WORKSPACE_RECOMMENDED = cutensor_backend.WORKSPACE_MIN
    #WORKSPACE_RECOMMENDED = cutensor_backend.WORKSPACE_RECOMMENDED
    _tensor_descriptors = {}
except (ImportError, AttributeError):
    cutensor = None
    ALGO_DEFAULT = None
    OP_IDENTITY = None
    JIT_MODE_NONE = None
    WORKSPACE_RECOMMENDED = None

def _auto_create_mode(array, mode):
    if not isinstance(mode, cutensor.Mode):
        mode = cutensor.create_mode(*mode)
    if array.ndim != mode.ndim:
        raise ValueError(
            'ndim mismatch: {} != {}'.format(array.ndim, mode.ndim))
    return mode

def _create_tensor_descriptor(a):
    if any(x == 0 for x in a.strides):
        strides = list(a.strides)
        if strides[0] == 0:
            strides[0] = a.nbytes
        for i, x in enumerate(strides[1:]):
            if x == 0:
                strides[i+1] = strides[i]
        a = cupy.ndarray(a.shape, a.dtype, a.data, strides)
    return cutensor.create_tensor_descriptor(a)

def _contract_einsum(pattern, a, b, alpha, beta, out=None, einsum=cupy.einsum):
    if out is None:
        out = cupy.asarray(einsum(pattern, a, b), order='C')
        out *= alpha
    elif beta == 0.:
        out[...] = einsum(pattern, a, b)
        out *= alpha
    else:
        tmp = einsum(pattern, a, b)
        tmp *= alpha
        out *= beta
        out += tmp
    return out

# cuTENSOR plan cache.  Building the descriptors, the contraction operator and the plan (heuristic search)
# costs ~130 us of host time per call, which is comparable to the GPU time of a small grid-block GEMM and
# makes the XC loops launch-bound on fast GPUs.  The shapes/strides repeat in every SCF cycle, so plans are
# cached; a plan is only valid for the pointer alignment it was created with, which is part of the key.
PLAN_CACHE = os.environ.get('GPU4PYSCF_CUTENSOR_PLAN_CACHE', '1') != '0'
_plan_cache = {}
_PLAN_CACHE_MAX = 8192

def _alignment(arr):
    ptr = arr.data.ptr
    return min(256, ptr & -ptr) if ptr else 256

_pattern_cache = {}

def _parse_pattern(pattern):
    res = _pattern_cache.get(pattern)
    if res is None:
        p = pattern.replace(" ", "")
        str_a, rest = p.split(',')
        str_b, str_c = rest.split('->')
        if len(str_c) != len(set(str_c)):
            raise ValueError('Output subscripts string includes the same subscript multiple times.')
        res = (p, str_a, str_b, str_c)
        _pattern_cache[pattern] = res
    return res

_scalar_cache = {}

def _scalar(x, dtype):
    # cuTENSOR reads alpha/beta through a host pointer: keep one numpy scalar per value
    try:
        key = (x, dtype.str)
        v = _scalar_cache.get(key)
    except TypeError:  # unhashable (e.g. 0-d array)
        return np.asarray(x, dtype=dtype)
    if v is None:
        if len(_scalar_cache) > 256:
            _scalar_cache.clear()
        v = _scalar_cache[key] = np.asarray(x, dtype=dtype)
    return v

def contraction(
    pattern, a, b, alpha, beta,
    out=None,
    op_a=OP_IDENTITY,
    op_b=OP_IDENTITY,
    op_c=OP_IDENTITY,
    algo=ALGO_DEFAULT,
    jit_mode=JIT_MODE_NONE,
    compute_desc=0,
    ws_pref=WORKSPACE_RECOMMENDED
):
    if a.size == 0 or b.size == 0:
        # cutensor does not support the 0-sized operands
        return _contract_einsum(pattern, a, b, alpha, beta, out)

    pattern, str_a, str_b, str_c = _parse_pattern(pattern)

    dtype = np.result_type(a.dtype, b.dtype)
    if a.dtype != dtype:
        a = cupy.asarray(a, dtype=dtype)
    if b.dtype != dtype:
        b = cupy.asarray(b, dtype=dtype)
    if out is None:
        shape = {k: v for k, v in zip(str_a + str_b, list(a.shape) + list(b.shape))}
        out = cupy.empty([shape[k] for k in str_c], order='C', dtype=dtype)
    c = out

    if c.size == 0:
        raise ValueError(f"cutensor contraction doesn't support zero-sized array (a.shape = {a.shape}, b.shape = {b.shape}, expected c.shape = {c.shape})")

    cache_key = None
    if PLAN_CACHE:
        cache_key = (pattern, a.shape, a.strides, b.shape, b.strides, c.shape, c.strides,
                     dtype.char, _alignment(a), _alignment(b), _alignment(c),
                     op_a, op_b, op_c, algo, jit_mode, compute_desc, ws_pref,
                     cupy.cuda.Device().id, threading.get_ident())
        hit = _plan_cache.get(cache_key)
    else:
        hit = None
    if hit is not None:
        plan, ws_size = hit
    else:
        mode_a = list(str_a)
        mode_b = list(str_b)
        mode_c = list(str_c)
        desc_a = _create_tensor_descriptor(a)
        desc_b = _create_tensor_descriptor(b)
        desc_c = _create_tensor_descriptor(c)

        mode_a = _auto_create_mode(a, mode_a)
        mode_b = _auto_create_mode(b, mode_b)
        mode_c = _auto_create_mode(c, mode_c)
        operator = cutensor.create_contraction(
            desc_a, mode_a, op_a, desc_b, mode_b, op_b, desc_c, mode_c, op_c,
            compute_desc)
        plan_pref = cutensor.create_plan_preference(algo=algo, jit_mode=jit_mode)
        ws_size = cutensor_backend.estimateWorkspaceSize(
            cutensor._get_handle().ptr, operator.ptr, plan_pref.ptr, ws_pref)
        plan = cutensor.create_plan(operator, plan_pref, ws_limit=ws_size)
        if cache_key is not None:
            if len(_plan_cache) >= _PLAN_CACHE_MAX:
                _plan_cache.clear()
            _plan_cache[cache_key] = (plan, ws_size)
    if ws_size:
        ws = cupy.empty(ws_size, dtype=np.int8)
        ws_ptr = ws.data.ptr
    else:
        ws_ptr = 0

    alpha = _scalar(alpha, dtype)
    beta = _scalar(beta, dtype)

    handler = cutensor._get_handle()
    cutensor_backend.contract(handler.ptr, plan.ptr,
                             alpha.ctypes.data, a.data.ptr, b.data.ptr,
                             beta.ctypes.data, c.data.ptr, c.data.ptr,
                             ws_ptr, ws_size)
    return c

import os
contract_engine = None
if cutensor is None:
    contract_engine = 'cupy'  # default contraction engine
contract_engine = os.environ.get('CONTRACT_ENGINE', contract_engine)

# override the 'contract' function if einsum is customized or cutensor is not found
if contract_engine is not None:
    einsum = None
    if contract_engine == 'opt_einsum':
        import opt_einsum
        einsum = opt_einsum.contract
    elif contract_engine == 'cuquantum':
        from cuquantum import contract as einsum # type: ignore
    elif contract_engine == 'cupy':
        einsum = cupy.einsum
    else:
        raise RuntimeError('unknown tensor contraction engine.')

    import warnings
    warnings.warn(f'using {contract_engine} as the tensor contraction engine.')
    def contract(pattern, a, b, alpha=1.0, beta=0.0, out=None):
        try:
            return _contract_einsum(pattern, a, b, alpha, beta, out, einsum)
        except cupy.cuda.memory.OutOfMemoryError:
            print('Out of memory error caused by cupy.einsum. '
                  'It is recommended to install cutensor to resolve this.')
            raise
else:
    def contract(pattern, a, b, alpha=1.0, beta=0.0, out=None):
        '''
        a wrapper for general tensor contraction
        pattern has to be a standard einsum notation
        '''
        return contraction(pattern, a, b, alpha, beta, out=out)
