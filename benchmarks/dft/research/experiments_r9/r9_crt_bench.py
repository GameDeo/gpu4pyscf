'''Round 9: fused CRT operand preparation vs the round-8 version (accuracy and time, XC-like and K-like shapes).'''
import os, time, importlib.util
import numpy as np, cupy
from gpu4pyscf.lib import crt_gemm as new
here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('crt_old', os.path.join(here, 'crt_gemm_old.py'))
old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)


def bench(f, n=20):
    f(); cupy.cuda.Stream.null.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        f()
    cupy.cuda.Stream.null.synchronize()
    return (time.perf_counter() - t) / n


rng = cupy.random.default_rng(0)
for (ncomp, m, n, ng, nmod) in [(1, 1000, 1000, 8000, 7), (4, 1000, 1000, 6000, 7), (1, 1000, 80, 20000, 7),
                                 (1, 1000, 1000, 8000, 12)]:
    a = rng.standard_normal((ncomp, m, ng)) * cupy.exp(rng.standard_normal((1, m, 1)))
    b = rng.standard_normal((ncomp, n, ng))
    ref = cupy.einsum('cmg,cng->mn', a, b)
    for name, mod in (('old', old), ('new', new)):
        c = mod.gemm_nt(a, b, nmod)
        err = float(abs(c - ref).max() / abs(ref).max())
        t = bench(lambda: mod.gemm_nt(a, b, nmod))
        print('gemm_nt  ncomp %d m %d n %d ng %d nmod %2d  %s  %.3f ms  rel err %.1e' % (ncomp, m, n, ng, nmod, name, 1e3 * t, err), flush=True)
    # PlanesT path (gemm_planes) on a [i, j, L] tensor viewed as (i, L, j)-ordered rows
    d0, d1, d2 = 60, 300, 1200
    T = rng.standard_normal((d0, d2, d1))                      # contiguous axis is d1
    view = T.transpose(0, 2, 1)                                # (d0, d1, d2) with stride-1 axis 1
    bt = rng.standard_normal((n if n < 200 else 120, d2))
    refp = cupy.einsum('ild,bd->ilb', view, bt)
    for name, mod in (('old', old), ('new', new)):
        o, P1 = mod.gemm_planes(view, bt, nmod)
        err = float(abs(o.reshape(d0, P1, -1)[:, :d1] - refp).max() / abs(refp).max())
        t = bench(lambda: mod.gemm_planes(view, bt, nmod))
        print('gemm_planes d %s nmod %2d  %s  %.3f ms  rel err %.1e' % ((d0, d1, d2), nmod, name, 1e3 * t, err), flush=True)
