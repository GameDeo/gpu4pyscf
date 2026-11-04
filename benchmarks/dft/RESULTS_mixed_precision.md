# wB97M-V + DF: stage profile and mixed-precision results

All runs: RKS, wB97M-V, `def2-universal-jkfit`, grid (99,590), NLC grid (50,194), conv_tol 1e-9,
build = this branch, CUDA 13.0, single GPU. Driver: `benchmarks/dft/bench_wb97mv.py`.
Timers synchronise the stream, so per-stage numbers are wall time.

## 1. FP64 baseline stage profile (A100-40GB, sm_80)

Tamoxifen, def2-TZVP (57 atoms, 1042 AO, 1.84 M grid pts, 0.38 M NLC pts), 12 SCF cycles, E = -1137.8785397043

| stage | total / s | per call / s | share of SCF (24.6 s) |
|---|---|---|---|
| XC main (`nr_rks`, 13 calls) | 11.0 | 0.85 | 45 % |
| NLC total (`nr_nlc_vxc`) | 9.6 | 0.74 | 39 % |
|   of which VV10 pair kernel | 7.7 | 0.59 | 31 % |
| DF `get_jk` (J + 2 K builds / cycle) | 3.3 | 0.13 | 13 % |
| DF tensor build (2: full + LR) | 1.2 | 0.6 | 5 % |

Inside XC main (fine timers, 5 cycles): rho contraction 1.9 s, vmat contractions ~3.5 s, AO eval 0.9 s,
libxc 0.015 s -> the XC stage is GEMM-bound; AO evaluation and libxc are minor.

Run-to-run noise (Bisphenol A, 2 identical runs): dE = 4.5e-13 Eh, max |dgrad| = 2.9e-8.

## 2. Reduced-precision building blocks (fixed-density regression tests)

| block | switch | deviation from FP64 (fixed density) |
|---|---|---|
| VV10 pair kernel, FP32 (hi/lo coords, FP64 chunked accumulation, rcp.approx) | `numint.VV10_PRECISION='fp32'` | dE_nlc ~ 2e-11 Eh, max dVnlc ~ 1e-11; SCF E within 1e-9 |
| DF-K, FP32 GEMMs | `df_jk.K_PRECISION='fp32'` | 6.5e-7 relative |
| DF-K, 3xTF32 (cuTENSOR) | `'tf32x3'` | < 5e-6 relative (but see caveats) |
| XC rho + Vxc GEMMs, FP32 | `numint.XC_PRECISION='fp32'` | PBE: dExc 2e-8, max dV 1e-6 |

Tests: `gpu4pyscf/dft/tests/test_vv10_fp32.py`, `test_xc_fp32.py`, `gpu4pyscf/df/tests/test_df_k_fp32.py`.

Using FP32 K for *all* SCF cycles does **not** converge (50 cycles, E off by 1.1e-4 Eh): the FP32
Fock noise floor is |g| ~ 1e-4. Hence the staged scheme (`mf.mixed_precision = True`,
`gpu4pyscf/scf/mixed_precision.py`): FP32 cycles to |g| < 3e-4, then an ordinary FP64 SCF from that density.

## 3. A100 (FP64 tensor cores ~ FP32 SIMT rate): no gain

| Tamoxifen def2-TZVP | SCF wall | note |
|---|---|---|
| FP64 | 24.6 s | 12 cycles |
| FP32 VV10 only | 20.7 s | VV10 0.59 -> 0.30 s/call; E within 8e-10 |
| staged mixed (fp32) | 39.0 s | 11 FP32 + 3 FP64 cycles; E within 1.5e-10; **slower** |
| staged mixed (tf32x3) | 108.8 s | stage 1 never reached its threshold (noise); **not working** |

FP32 K (0.28 s/call) and FP32 XC (1.3 s/call) are slower than the FP64 versions (0.13 / 0.85 s) on A100.

## 4. T4 (FP64 = 1/32 FP32): measured, Bisphenol A def2-TZVP (33 atoms, 623 AO)

| config | SCF wall | speedup | E - E_ref |
|---|---|---|---|
| FP64 | 249.0 s | 1.0x | 1.9e-10 |
| staged mixed, VV10 FP32 in stage 1 only | 138.3 s | 1.8x | 2.2e-10 |
| staged mixed, VV10 FP32 in both stages | 100.7 s | 2.5x | -2.7e-10 |

Per call (3-cycle runs): XC 8.8 -> 1.64 s (5.4x), VV10 7.9 -> 0.38 s (21x), DF get_jk 1.56 -> 1.40 s (1.1x).
Gradients (vs FP64 conv_tol 1e-11): FP64 default tol 3.4e-7, staged mixed 1.2e-6 (max abs, Eh/bohr).

## 5. Stage-2 experiments (see PLAN_stage2_algorithms.md)

Operands are captured from a real stage-1-quality SCF (`gemm_precision_bench.py`), not random matrices.

### E1 - tensor-core FP32 emulation (A100, Tamoxifen operands)
* cuTENSOR `COMPUTE_DESC_3xTF32` is **not** a tensor-core path: errors and times are identical to FP32 SIMT
  (e.g. K=65536: 7.83e-7, 3.75 ms vs 3.74 ms). The earlier `tf32x3` SCF mode was therefore plain FP32 (no gain) and
  its non-convergence was not a tensor-core effect; treat `tf32x3` as unsupported.
* Hand-split 3xTF32 (cuBLAS TF32): accumulating the whole K inside the tensor core gives 1.6e-5..2.8e-5 relative
  error (FP32 SIMT: 5e-7..8e-7) - the round-toward-zero in-TC accumulation of Ootomo & Yokota. Moving the main
  product's accumulation outside the TC every kc K-steps restores FP32 accuracy (kc=256: ~2e-7 on vmat, kc=64: ~1e-6
  on K2), but a library-call loop only reaches 0.5-2 TFLOP/s: a fused CUTLASS-style kernel would be needed.
* Plain TF32 (1x): 4e-5 (vmat) to 5e-4 (R R^T) relative error: first SCF cycles only.

### E2 - defect correction F(D) ~ F_low(D) + [F_hi(D0) - F_low(D0)]  (BPA, |dD|_F ~ 3-5e-2)
* K full range: error reduced 4-5x only (ratio |e(D)-e(D0)|/|e(D)| = 0.18-0.23); LR K: 0.35-0.39.
* Vxc: ratio 1.1 - the FP32 error is rounding noise that decorrelates with D; correction does nothing.
* Verdict: cannot replace the FP64 polish stage. NO-GO as a stand-alone scheme.

### E3 - one K build (`range_separated_mode='mix_inside_kernel'`)
| | BPA | Tamoxifen |
|---|---|---|
| dE vs two-build reference | 6.8e-7 Eh | 1.1e-6 Eh |
| DF tensors in memory | 2.98 -> 2.02 GB | (7.4 GB, one tensor) |
| `get_jk` time | 1.02 -> 0.51 s | 3.31 -> 2.01 s |
Opt-in approximation only (fails a 1e-9 gate); saves ~5 % of an A100 SCF and 1/3 of the DF memory.

### Staged-SCF gradient quality (BPA, reference: FP64 conv_tol 1e-13, |g| < 1e-7)
| run | FP64 cycles | max abs gradient error |
|---|---|---|
| FP64, default tol | 12 | 4.9e-7 |
| FP64, conv_tol 1e-11 | 14 | 8.8e-8 |
| staged, final_grad_factor 0.3 | 4 | 9.4e-7 |
| staged, final_grad_factor 0.1 (default now) | 6 | 1.0e-7 |
| staged, 0.03 / 0.01 | 7 / 8 | 3.0e-8 / 2.4e-8 |
FP32 VV10 alone does not degrade the gradient (5.2e-7 vs 5.1e-7 for FP64).

### E4 - emulated FP64 from INT8 tensor cores
* cuBLAS 13.1 fixed-point FP64 emulation exists (math mode 8, mantissa-bit control). **A100:** works, error 1e-9 (31 b),
  1e-11 (39 b), 1e-13 (47 b), 1e-15 (55 b), but is 2-4x *slower* than native FP64 (strong FP64 tensor cores).
  **T4 (Turing):** silently falls back to native FP64 (bit-identical results, same time).
* Own Ozaki-I implementation (`ozaki_int8.py`: power-of-two row/column scaling, 7-bit int8 slices, triangular
  s+t<S products via `cublasGemmEx` TN; Turing needs m,n,k % 16 == 0), T4, real BPA operands, GEMM time only:
| shape | native FP64 | S=6 (42 b) | S=8 (56 b) |
|---|---|---|---|
| XC vmat K=61440 | 168 ms | 23.6 ms (7.1x), err 2.4e-11 | 40.8 ms (4.1x), err 5.9e-15 |
| K2 = R R^T, K=7808 | 30.9 ms | 4.3 ms (7.2x), err 2.1e-9 | 7.8 ms (4.0x), err 2.3e-13 |
| K1 = cderi @ C (N=61) | 28.6 ms | 17.8 ms (1.6x) | 30 ms (0.95x) |
  Slicing in cupy elementwise ops is far too slow (334 ms for the K=61440 operands; FP64 ops on a 1/32-rate GPU):
  a fused integer-arithmetic slicing kernel is required before this can be used inside the XC loop.
* Static operands (the DF tensor) can be sliced once; the precision ladder is then the number of slices used.


## Round 2 (T4, Bisphenol A, def2-TZVP, wB97M-V, DF, default grids)

### Where one FP64 XC evaluation goes (kernel-level profile, `profile_xc_kernels.py`, T4)
| | FP64 `nr_rks` (8.64 s) | FP32 `nr_rks` (1.50 s) |
|---|---|---|
| GEMM | 8.30 s (95 %) | 0.69 s (cuBLAS 0.47 + cuTENSOR 0.22) |
| AO evaluation (gdft) | 0.36 s (4 %) | 0.33 s |
| elementwise / casts | 0.08 s | 0.20 s |
| idle / launch gaps | - | 0.28 s |
So an FP64 XC evaluation is GEMM-bound (an INT8/TF32 emulation of those GEMMs is not Amdahl-limited), and the
FP32 path is already within ~1.5x of its floor (5.8x faster than FP64).

### Noise floor of each reduced-precision component (20 SCF cycles each, `noise_floor.py`)
| configuration | self-consistent \|g\| plateau | E drift |
|---|---|---|
| XC FP32, K FP64, VV10 FP32 | 7e-6 .. 1e-5 | 5e-7 |
| K FP32, XC FP64, VV10 FP32 | 8e-5 .. 1e-4 | 9e-7 |
But the **FP64** gradient at the FP32-XC solution is ~4e-5, not 1e-5: the FP32 Fock is offset from the FP64 Fock
(`xc_error_split.py`: rho stage contributes 1.8e-7, the Vxc-matrix stage 2.0e-5 in \|dg_ov\|_F). The offset is FP32
accumulation over the 4096-point block (error ~ linear in chunk length: 1.0e-5 at 2048 points, 3.2e-6 at 128, 1.7e-6 at 64)
and chunking costs 10x in time, so it is not a cheap fix. A reduced-precision stage can therefore replace most but not all
of the FP64 polish: 2-3 exact Fock builds remain.

### Staged scheme now: stage 1 (K,XC,VV10 reduced) -> stage 1b (K FP64, XC+VV10 reduced, plateau-detected) -> FP64
| run (SCF only) | wall | FP64 cycles | dE vs tight ref | max gradient error |
|---|---|---|---|---|
| FP64 | 249.7 s | 12 | 4e-11 | 4.3e-7 |
| staged, no stage 1b (round 1 scheme) | 123.0 s | 6 | 3.9e-10 | 1.3e-7 |
| staged + 1b + FP32 NLC-grid XC (default) | 87.0 s | 1-2 | 3.7e-10 | 1.2e-7 |
`final_grad_factor` 1.0 (threshold 3e-5, equal to plain FP64 default): 76 s but the gradient error is 1.7e-6 (fails the gate); 0.1 stays the default.

### Gradient, exact restructuring (`grad/rks.py`)
The XC gradient built three [nao, nao] matrices per block (15 GEMMs for MGGA) only to contract them with D.
Contracting D into the weighted AO factors first needs 5 GEMMs, and applying D as C(C^T x) over the occupied orbitals
costs nocc/nao = 0.14 of a dense product. Result identical to 1e-14 (4 tests: LDA/GGA/MGGA/wB97M-V).
| Bisphenol A gradient, T4 | time |
|---|---|
| legacy | 39.8 s (47.1 s in the earlier full run) |
| contract-first | 22.1 s |
| + occupied-orbital factor | 17.2 s |
Applies to every GPU (an FP64 tensor-core GPU also runs fewer FP64 FLOPs).

### End to end (SCF + gradient, T4)
| | SCF | gradient | total | dE | max grad error |
|---|---|---|---|---|---|
| original code, FP64 | 249.7 s | 47.1 s | 296.8 s | 4e-11 | 4.3e-7 |
| mixed precision (this branch) | 87.0 s | 17.3 s | 104.3 s | 3.7e-10 | 1.2e-7 |
(2.85x; SCF alone 2.9x.)  Remaining mixed-SCF budget: FP64-stage XC ~22 s, FP32-stage XC ~22 s, K 28 s, NLC 12.6 s (VV10 7.7 s).

### A100 (SXM4-40GB), same Bisphenol A job - what the precision work does and does not buy there
| | FP64 reference | mixed, auto precision (K/XC FP64, VV10 FP32) |
|---|---|---|
| SCF | 12.35 s (12 cycles) | 12.8 s (9 + 5 cycles, final_grad_factor 0.1); 12.0 s with factor 1.0 but gradient error 1.0e-6 |
| NLC | 4.11 s | 3.1-3.4 s (FP32 VV10: the only real gain) |
| gradient (new code) | 2.83 s | 2.37 s |
| dE / max grad error | 1.4e-10 / 3.5e-7 | 6.6e-10 / 1.9e-7 |
* Mixed precision gives **no end-to-end speedup on A100** at this size (FP64 tensor cores = FP32 SIMT throughput; the
  reduced-precision path is only the 2x VV10 gain, offset by the extra FP64 cycles of the tighter staged threshold).
  `mixed_precision` is opt-in and should be left off on FP64-tensor-core GPUs; `_default_gemm_precision()` now picks
  FP64 K/XC there so the mode is at worst neutral.
* The contract-first XC gradient is **not** faster on A100 (2.40 s legacy vs 2.43 s): the A100 gradient is not GEMM-bound.
  Results identical to 1e-13.
* One XC evaluation on A100 (FP64, 0.49 s wall) has only 0.245 s of GPU kernel time: GEMM 0.158 s, AO evaluation 0.061 s,
  elementwise 0.019 s. **Half the wall time is idle gaps** (python/cupy launch overhead in the 2 x 260-block loops).
  On this hardware the XC cost is orchestration, not arithmetic.
