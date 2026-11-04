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
