#!/usr/bin/env python
'''
E1/E4 micro-benchmark: GEMM accuracy and throughput on *real* DFT operands.

Operands are captured from a stage-1-quality SCF (default 6 cycles) of the chosen molecule:
  rho  : C_occ^T (nocc x nao_sub) @ ao_c (nao_sub x ng)         K = nao_sub
  vmat : ao0 (nao x ng*n) @ aow^T (ng*n x nao)                   K = ng*n   (embedded in full nao)
  K1   : cderi (L*nao x nao) @ C_occ (nao x nocc)                K = nao
  K2   : R (nao x L*nocc) @ R^T                                  K = L*nocc

Variants (inputs rounded to FP32 first, so the reported error is arithmetic error only):
  fp32_simt   cuBLAS SGEMM, TF32 off
  tf32_1x     cuBLAS with TF32 tensor cores (10-bit mantissa inputs)
  tf32_3x     hand split A=Ah+Al, B=Bh+Bl; Ah@Bh + (Ah@Bl + Al@Bh); main product accumulated in
              K-chunks of `kc` whose partial results are added in FP32 *outside* the tensor core
              (kc=0: whole K inside the tensor core)
  cutensor3x  cuTENSOR COMPUTE_DESC_3xTF32 (what df_jk 'tf32x3' uses)
Errors are relative to an FP64 product of the FP32-rounded inputs.  Timings are GEMM-only
(operands pre-split / pre-cast), median of several runs, CUDA events.
'''
import argparse
import time
import numpy as np
import cupy
from pyscf import gto


def sync():
    cupy.cuda.get_current_stream().synchronize()


def bench(fn, reps=7):
    fn(); sync()
    ts = []
    for _ in range(reps):
        e0 = cupy.cuda.Event(); e1 = cupy.cuda.Event()
        e0.record(); fn(); e1.record(); e1.synchronize()
        ts.append(cupy.cuda.get_elapsed_time(e0, e1) * 1e-3)
    return float(np.median(ts))


def set_tf32(on):
    h = cupy.cuda.device.get_cublas_handle()
    cupy.cuda.cublas.setMathMode(h, 3 if on else 0)   # 3 = CUBLAS_TF32_TENSOR_OP_MATH


def split_tf32(a):
    '''a (float32) = hi + lo with hi holding 10 mantissa bits (round-to-nearest-away), lo exact.'''
    u = a.view(cupy.uint32)
    hi = ((u + np.uint32(0x1000)) & np.uint32(0xFFFFE000)).view(cupy.float32)
    return hi, a - hi


class Gemm3x:
    def __init__(self, a, b, kc):
        self.kc = kc
        self.ah, self.al = split_tf32(a)
        self.bh, self.bl = split_tf32(b)
        # correction terms in a single product over a doubled contraction: [Ah|Al] @ [Bl;Bh]
        self.acat = cupy.ascontiguousarray(cupy.concatenate([self.ah, self.al], axis=1))
        self.bcat = cupy.ascontiguousarray(cupy.concatenate([self.bl, self.bh], axis=0))
        self.k = a.shape[1]

    def __call__(self):
        set_tf32(True)
        if self.kc and self.kc < self.k:
            out = None
            for k0 in range(0, self.k, self.kc):
                p = cupy.matmul(self.ah[:, k0:k0+self.kc], self.bh[k0:k0+self.kc])
                out = p if out is None else out + p      # FP32 adds outside the tensor core
        else:
            out = cupy.matmul(self.ah, self.bh)
        out += cupy.matmul(self.acat, self.bcat)
        set_tf32(False)
        return out


def cutensor3x(a, b):
    from gpu4pyscf.lib import cutensor as ct
    out = cupy.zeros((a.shape[0], b.shape[1]), dtype=cupy.float32)
    desc = ct.cutensor_backend.COMPUTE_DESC_3xTF32

    def run():
        ct.contraction('ik,kj->ij', a, b, 1.0, 0.0, out=out, compute_desc=desc)
        return out
    return run


def err_stats(c, ref):
    c = c.astype(cupy.float64)
    d = cupy.abs(c - ref)
    scale = float(cupy.abs(ref).max())
    return float(d.max()) / scale, float(cupy.linalg.norm(c - ref) / cupy.linalg.norm(ref))


def run_case(name, a64, b64, kcs=(0, 1024, 256, 64), reps=7):
    a = a64.astype(cupy.float32); b = b64.astype(cupy.float32)
    ref = cupy.matmul(a.astype(cupy.float64), b.astype(cupy.float64))
    m, k = a.shape; n = b.shape[1]
    flops = 2.0 * m * n * k
    rows = []

    set_tf32(False)
    t = bench(lambda: cupy.matmul(a, b), reps)
    rows.append(('fp32_simt', t, err_stats(cupy.matmul(a, b), ref)))
    set_tf32(True)
    t = bench(lambda: cupy.matmul(a, b), reps)
    rows.append(('tf32_1x', t, err_stats(cupy.matmul(a, b), ref)))
    set_tf32(False)
    for kc in kcs:
        g = Gemm3x(a, b, kc)
        t = bench(g, reps)
        rows.append((f'tf32_3x kc={kc}', t, err_stats(g(), ref)))
    try:
        f = cutensor3x(a, b)
        t = bench(f, reps)
        rows.append(('cutensor3x', t, err_stats(f(), ref)))
    except Exception as ex:
        rows.append((f'cutensor3x FAILED {type(ex).__name__}', float('nan'), (float('nan'),) * 2))
    t64 = bench(lambda: cupy.matmul(a64, b64), reps)
    rows.append(('fp64 native', t64, (0.0, 0.0)))
    print(f'--- {name}: M={m} N={n} K={k}')
    print(f'{"variant":22s} {"ms":>9s} {"TFLOP/s":>9s} {"max|dC|/max|C|":>15s} {"rel Frob":>10s}')
    for v, t, (emax, efro) in rows:
        print(f'{v:22s} {t * 1e3:9.3f} {flops / t / 1e12:9.2f} {emax:15.2e} {efro:10.2e}')
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mol', default='057_Tamoxifen')
    p.add_argument('--basis', default='def2-tzvp')
    p.add_argument('--cycles', type=int, default=6)
    p.add_argument('--nblocks', type=int, default=16)
    p.add_argument('--save', default=None, help='np.savez the captured XC operands here')
    args = p.parse_args()

    import os
    from gpu4pyscf import dft
    from gpu4pyscf.dft import numint
    here = os.path.dirname(os.path.abspath(__file__))
    mol = gto.M(atom=os.path.join(here, '..', 'molecules', 'organic', args.mol + '.xyz'),
                basis=args.basis, verbose=0, max_memory=40000)
    mf = dft.RKS(mol, xc='wb97m-v').density_fit()
    mf.grids.atom_grid = (99, 590)
    mf.nlcgrids.atom_grid = (50, 194)
    mf.max_cycle = args.cycles
    mf.kernel()
    dm = mf.make_rdm1()
    nao = mol.nao
    print(f'# {args.mol} nao={nao}; captured at stage-1-quality density after {args.cycles} cycles')

    # ---- capture XC operands from one nr_rks call -------------------------------------------
    rho_calls, tau_calls, scale_calls = [], [], []
    o_rho, o_tau, o_scale = numint._eval_rho2, numint._tau_dot, numint._scale_ao

    def w_rho(ao, cpos, *a, **k):
        rho_calls.append(None)
        if len(rho_calls) in (30, 200, 350):
            rho_calls[-1] = (ao.copy(), cpos.copy())
        return o_rho(ao, cpos, *a, **k)

    def w_scale(ao, wv, out=None):
        if wv.ndim == 2:
            scale_calls.append(None)
            if len(scale_calls) in (30, 200, 350) or 100 <= len(scale_calls) < 100 + args.nblocks:
                scale_calls[-1] = (ao.copy(), wv.copy())
        return o_scale(ao, wv, out=out)

    numint._eval_rho2, numint._scale_ao = w_rho, w_scale
    try:
        mf._numint.nr_rks(mol, mf.grids, mf.xc, dm)
    finally:
        numint._eval_rho2, numint._scale_ao = o_rho, o_scale
    rho_ops = [x for x in rho_calls if x is not None]
    scale_ops = [x for x in scale_calls if x is not None]
    print(f'# captured {len(rho_ops)} rho blocks, {len(scale_ops)} vmat blocks')

    # ---- XC-shaped cases --------------------------------------------------------------------
    for i, (ao, cpos) in enumerate(rho_ops[:2]):
        nao_sub, nocc = cpos.shape
        run_case(f'XC rho block (nao_sub={nao_sub}, nocc={nocc})', cpos.T.copy(), cupy.ascontiguousarray(ao[1]))
    for i, (ao, wv) in enumerate(scale_ops[:2]):
        aow = o_scale(ao, wv)
        run_case(f'XC vmat single block (nao_sub={ao.shape[1]})', cupy.ascontiguousarray(ao[0]),
                 cupy.ascontiguousarray(aow.T))

    # K-length sweep of the vmat contraction: stack consecutive blocks, embedded in full nao
    # (AO indices of a block are not captured, so blocks are padded to a common width instead)
    blocks = scale_ops[3:] if len(scale_ops) > 6 else scale_ops
    width = max(b[0].shape[1] for b in blocks)
    A, B = [], []
    for ao, wv in blocks:
        aow = o_scale(ao, wv)
        pad = width - ao.shape[1]
        a0 = cupy.pad(ao[0], ((0, pad), (0, 0)))
        bw = cupy.pad(aow, ((0, pad), (0, 0)))
        A.append(a0); B.append(bw)
    for nb in (1, 2, 4, 8, len(A)):
        if nb > len(A):
            continue
        a = cupy.ascontiguousarray(cupy.concatenate(A[:nb], axis=1))
        b = cupy.ascontiguousarray(cupy.concatenate(B[:nb], axis=1).T)
        run_case(f'XC vmat K-sweep: {nb} stacked blocks', a, b, reps=5)

    # ---- K-shaped cases from the real DF tensor ----------------------------------------------
    from gpu4pyscf.df.df_jk import factorize_dm
    from gpu4pyscf.lib.cupy_helper import fill_symmetric
    dfobj = mf.with_df
    if dfobj._cderi is None:
        dfobj.build()
    c_sorted = dfobj._cderi[0]
    pair_idx = cupy.asarray(dfobj._cderi_idx[0], dtype=np.int32)
    cl, _ = factorize_dm(dm, 1)
    nocc = cl.shape[1]
    Lmax = 128
    blk = cupy.asarray(c_sorted[:Lmax])
    work = cupy.zeros((nao, nao, Lmax))
    full = fill_symmetric(blk.T, pair_idx, nao, 0, Lmax, out=work).transpose(2, 0, 1)   # [L,i,j]
    full = cupy.ascontiguousarray(full)
    run_case(f'K1: cderi[{Lmax}*{nao},{nao}] @ C[{nao},{nocc}]', full.reshape(Lmax * nao, nao),
             cupy.ascontiguousarray(cl), reps=5)
    R = (full.reshape(Lmax * nao, nao) @ cl).reshape(Lmax, nao, nocc)        # [L,i,k]
    for L in (8, 16, 32, 64, 128):
        A2 = cupy.ascontiguousarray(R[:L].transpose(1, 0, 2).reshape(nao, L * nocc))
        run_case(f'K2: R[{nao},{L}*{nocc}] @ R^T  (contraction length sweep)', A2,
                 cupy.ascontiguousarray(A2.T), reps=5)

    if args.save:
        np.savez(args.save, **{f'rho{i}_ao': ao.get() for i, (ao, c) in enumerate(rho_ops)},
                 **{f'rho{i}_c': c.get() for i, (ao, c) in enumerate(rho_ops)},
                 **{f'v{i}_ao': ao.get() for i, (ao, wv) in enumerate(scale_ops[:3])},
                 **{f'v{i}_wv': wv.get() for i, (ao, wv) in enumerate(scale_ops[:3])})
        print('# saved', args.save)


if __name__ == '__main__':
    main()
