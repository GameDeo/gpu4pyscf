# Stage 2 plan: algorithmic / precision research for wB97M-V + DF

Stage 1 (done, see RESULTS_mixed_precision.md): hardware-utilisation work on the existing algorithm.
Stage 2 (this plan): replace FP64 work by cheaper *equivalent* work, and cut the number of expensive builds.

Ground rules: every item has a go/no-go metric, a fixed-density regression test, and is reported even if it fails.
Compute is spent sparingly: micro-benchmarks first (minutes on one GPU), SCF runs only for survivors.

## What the literature says (read this session)

| source | finding that matters here |
|---|---|
| Ootomo & Yokota, arXiv:2203.03341 | FP32-accurate GEMM from TF32/FP16 tensor cores needs the **large product accumulated outside the tensor core** (TC accumulator rounds toward zero); drop the dA*dB term (3 products). Naive in-TC accumulation reproduces Markidis-type error. |
| NVIDIA cuEST docs (emulation / gemm_emulation) | DF-K uses Ozaki-I (INT8 slices) / Ozaki-II (moduli) emulated FP64 GEMM; slices>=4 / moduli 7-10 give SCF-grade accuracy; defaults ~5 slices, 8 moduli reach |g|~1e-7; **recommend low precision early, higher later**. VV10 uses a float-float (47-bit) type when FP32:FP64 >= 32. Needs CUDA >= 13.0.2. |
| Huang, Shao, Hammond, arXiv:2601.08077 (J. Phys. Chem. A 2026) | Adaptive-precision INT8 DF-K in PySCF: precision level chosen from relative SCF energy change (23/31/39/47 mantissa bits -> FP64), never reverts; up to 3.0x (RTX 4090), 4.6x (RTX 6000 Ada), 1.37x (H100); same iteration count (+<=2); abs. energy error < 1e-7 Eh. Only K is accelerated; J and XC stay FP64. |
| Dawson, Domke, Nakajima, Ozaki, arXiv:2407.13299 | Accumulation precision dominates error (~37 bits suffice); FP32 storage fine; one FP64 refinement step after low-precision convergence repairs the energy. |
| Luehr, Ufimtsev, Martinez, JCTC 7, 949 (2011) | Dynamic-precision ERIs: largest integrals in double, rest single; threshold tightened through the SCF. |
| Lin, ACE operator, arXiv:1601.07159 | Low-rank (ACE) exchange operator built once per outer iteration; inner SCF iterations need **no K build**. |
| Laikov, arXiv:2511.04169 | Separable, density-fitted VV10 approximation: no O(N_grid^2) sum. Changes the functional (up to ~18 % in noble-gas dimer bond energies), so research branch only. |
| Nepal & Wang, arXiv:2603.29975 | Ozaki-II (GEMMul8) emulation in a DFT code via BLAS interposition, no algorithm change. |

## Hypotheses to test (ordered by expected value per compute unit)

| id | experiment | go / no-go |
|---|---|---|
| E1 | **Why did tf32x3 stall?** Hypothesis: in-TC RZ accumulation. Micro-benchmark GEMM shapes of K and XC: cuTENSOR 3xTF32 vs hand-split TF32 (pre-split operands, 3 GEMMs, partial results summed in FP32 outside the TC, K-chunked). Error vs FP64 and throughput on A100. | error within 2x of FP32 SIMT and >= 2x faster than FP32 on A100 |
| E2 | **Defect correction (cut FP64 builds).** F(D) ~ F_low(D) + [F_hi(D0) - F_low(D0)]. Check on fixed densities that (F_low - F_hi) is mostly *linear in D* (systematic rounding of the DF tensor / AO values), i.e. the correction cancels it to O(eps*|D-D0|). If yes, replace the 4-5 FP64 polish evaluations by 1-2. | residual after correction < 10 % of uncorrected error for |D-D0|/|D| ~ 1e-2 |
| E3 | **One K build instead of two** (`range_separated_mode='mix_inside_kernel'`): wB97M-V currently builds full-range and LR K every iteration from two DF tensors. | dE vs mix_outside_kernel reported; accept if < 1e-6 Eh and K time halves |
| E4 | **Ozaki-II INT8 FP64 emulation (GEMMul8) for the FP64 stage** on T4 (INT8 130 TOPS vs FP64 0.25 TFLOPS) and A100: micro-benchmark XC-shaped (batched [nao_sub x 4096] x [4096 x nao_sub]) and K-shaped GEMMs, accuracy vs moduli count. cuBLAS 13.0.2 emulated DGEMM probed too. | >= 5x native FP64 on T4 at SCF-grade accuracy (moduli ~8-10) |
| E5 | **Energy-keyed precision ladder** (Huang et al.): one `precision level` knob for K and XC GEMMs, driven by relative energy change, never reverting. Replaces the hard stage-1/stage-2 split. Needs E1 or E4. | same iteration count (+<=2), E within 1e-7 of FP64, wall time <= staged mixed |
| E6 | **K-build structure:** (a) SYRK/half-triangle for the final contraction (exact 25 % FLOP cut), (b) FP32/BF16 storage of the DF tensor (2x capacity: Penicillin V/Tamoxifen fit on a T4), (c) ACE two-level SCF (K only in outer iterations). | (a) exact; (b) memory 2x; (c) K builds 12 -> ~4 at equal E |
| E7 | **XC GEMM restructuring:** stack the 3 tau GEMMs + GGA GEMM into one K=4*4096 GEMM; fuse rho/tau/grad elementwise ops; replace per-block `astype` by an FP32-output AO kernel; AO cache in FP16/FP32 when it fits. | XC FP32 path >= 3x faster on T4 (now 5.4x vs ideal ~30x) |
| E8 | **VV10:** symmetric-pair kernel (~1.5x fewer ops); coarse-NLC-grid study (approximate, report dE); Laikov-type DF-VV10 as an *approximate-functional* research branch only. | exact items must stay within 1e-9 Eh |
| E9 | **Orchestration:** overlap VV10 (compute-bound) with XC/K (bandwidth/GEMM) on separate streams; multi-GPU split of NLC pairs. | measured overlap > 15 % |

## Known caveats that shape the plan
* On A100/H100 FP64 tensor cores make plain FP32 *not* faster; only TF32/INT8 paths can win there (E1, E4).
* On T4/L4/RTX the FP64 stage is the bottleneck of the current staged scheme (about 60 % of its time), so E2 and E4 matter most there.
* Gradients: staged mixed run gave 1.2e-6 vs 3.4e-7 for FP64 at default tolerance (reference conv_tol 1e-11); final-stage tolerance must be tightened or the polish extended.


## Status after the first A100/T4 experiment round

| id | result |
|---|---|
| E1 | cuTENSOR 3xTF32 is not a tensor-core path (== FP32 SIMT). Hand-split 3xTF32 confirms in-TC RZ accumulation is the accuracy limiter; needs a fused kernel. Open. |
| E2 | NO-GO as FP64-polish replacement (K error cut only 4-5x; Vxc error is uncorrelated rounding noise). |
| E3 | Approximation (dE 7e-7..1e-6 Eh), halves K time and 1/3 of DF memory. Opt-in only. |
| E4 | cuBLAS emulation: slower than native on A100, silent no-op on T4. Own Ozaki-I INT8: 4-7x over native FP64 on T4 at 42-56 bits for big-K GEMMs. Needs fused slicing kernel + integration. **Highest-value next step.** |
| gradient | final_grad_factor 0.1 gives gradients ~5x closer to a tight reference than plain FP64 at default tolerance. |
| denied | Cloning/building third-party GEMMul8 on the runtime was refused by the permission classifier; own implementation used instead. |
