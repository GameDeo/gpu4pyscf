# Research report 9: round 8: memory-light surrogates, the FP32 gradient tier, and where the time goes

Date 2026-11-05.
- **Hardware:** Colab high-RAM T4 (`t4hm`, 50 GB host memory); local CPU (4 cores) for the surrogate experiments.
- **Systems:** BPA and Taxol, wB97M-V/def2-TZVP; phenol HF/def2-TZVP for the CPU emulations.
- **Logs:** `experiments_r8/logs/`.
- **Tags:** [M] measured, [E] estimate.

## 0. Bottom line

1. **Taxol on one T4, SCF + gradient (section 1).**
   - The SCF converges in 3 exact checkpoints (22 min) with 5 GB peak GPU pool.
   - The round-7 gradient attempt was killed by the host OOM killer: cupy's pinned pool kept the 39 GB of surrogate
     tensors after the SCF. Fixed. TAXOL8_SUMMARY
2. **Memory scaling can change asymptotically without losing contraction (section 2) [M, CPU emulation].**
   - Two surrogates need no stored 3-index tensor and converge at least as fast as the stored ones:
     - a grid-fitted J with the exact auxiliary basis;
     - a seminumerical (COSX-type) exchange on a small grid.
   - Contraction per exact checkpoint is 884/996/377x, against 228/111/103x for the round-7 surrogate. With the
     inner loop converged to 1e-3 of the outer gradient, ~1000x is the ceiling: the seminumerical surrogate sits on it,
     so its true rate is at least that (a run with inner_factor 1e-4 is below).
   - **Limits of this evidence:** HF on phenol on a CPU (lambda_min(S) 1.1e-5, vs 3.4e-6 for Taxol); the
     long-range erf exchange that dominates wB97M-V is not in the test.
   - Surrogate memory drops from O(npairs * naux), about N^3 for compact molecules (39 GB at Taxol), to
     O(N_grid batch), about linear.
   - Negative results: our LS-THC (K-means points) still has a 6 % response error at 16x nao, and NAF compression of
     the SVP-RI auxiliary basis keeps 374 of 420 functions at 1e-4 with no accuracy change.
3. **Gradient: FP32 Rys tier made accurate (section 3) [M].**
   - The FP32 derivative kernel's error was mostly its FP32 accumulation. With FP64 sums, the all-FP32 error falls
     from 2.3e-4 to 6.2e-6.
   - The 10 % tier adds 1.6e-7 and is now the default. The BPA gradient drops from 5.55 s to 4.55 s, unchanged
     accuracy vs the tight reference (4.8e-7).
4. **BPA now (T4, profiled run):** SCF 14.6 s + gradient 4.7 s = 19.3 s, dE 1.6e-10, max|dg| 4.9e-7.
   - The 16 s target needs ~3.3 s more. Section 4 ranks the remaining inefficiencies from a fresh trace.

## 1. Taxol (nao 2228, naux 6598 exact / 3599 jfit / 3690 SVP-RI, nocc 226) on one T4 [M]

SCF (`r7_taxol.log`, `inc_store='f32'`, `exact_k='direct'`): E = -2929.6724601330, 1319.6 s.
- **Checkpoints:** outer |g| 3.8e-2, 4.2e-4, 1.6e-5 (3 checkpoints), with 10 stage-1 cycles and 13 inner cycles.
- **Peak GPU pool:** 5.05 GB.

| stage | s | per call | share |
|---|---|---|---|
| exact K + J (direct, FR + LR) | 726 | 242 | 55 % |
| K_s increments (host-streamed FP32, 19.6 GB) | 135 | 10.4 | 10 % |
| surrogate tensor builds (2 x ~50 s) | 99 | | 8 % |
| J_s (host-streamed FP32, 19.1 GB) | 97 | 4.0 | 7 % |
| K_s stage 1 | 86 | 7.8 | 7 % |
| fine XC (+ XC gradient) | 55 | | 4 % |
| fit for the gradient (`grad fit_oo`) | 24 | | 2 % |
| rest (NLC, coarse XC, DIIS, eig, setup) | 98 | | 7 % |

- **Surrogate cost:** a third of the SCF (433 s) is spent building and streaming the surrogate tensors. That is the
  part section 2 removes.
- **Exact checkpoints:** the other half. Each is two streamed integral passes per operator (FR, LR), with the
  INT8-CRT half transforms dominant (report 7 section 3).

TAXOL8_GRAD

## 2. Memory scaling: surrogates without stored tensors [M, CPU emulation]

**Method.** `r8_thc_cpu.py` and `r8_gridj_cpu.py` emulate the multi-fidelity outer loop of `scf/multifidelity.py` exactly:
- the exact J at the checkpoint is the reference, with surrogate J increments;
- surrogate K;
- the occupied-projected defect Delta~;
- an inner loop to 1e-3 of the outer gradient.

The exact operator is DF-HF with def2-universal-jkfit; the molecule is phenol, def2-TZVP (nao 253, lambda_min(S)
1.1e-5). All runs start from the same density, and the exact orbital gradient is reported at each checkpoint. Only
ratios between surrogates are meaningful: HF weights exchange differently from wB97M-V. Calibration: the round-7
surrogates give 228x (SVP-RI K + exact J) and 100x (SVP-RI K + jfit J), the same ordering as the wB97M-V measurements
on BPA (400x and 100x).

Operator errors are measured on the occupied-virtual block, for a random ov perturbation dD (the inner-loop response)
and for the density D0:

| surrogate | storage | rel. err. K or J (dD) | outer \|g\| (3 steps) | contraction per step |
|---|---|---|---|---|
| SVP-RI DF K + exact J (round-7 default) | npairs x naux (K) | 3.1e-3 | 1.46e-2, 6.4e-5, 5.8e-7, 5.6e-9 | 228, 111, 103 |
| SVP-RI DF K + jfit DF J (round-7 Taxol path) | 2 x npairs x naux | J 4.4e-2 | 1.46e-2, 1.4e-4, 1.4e-6, 1.4e-8 | 103, 102, 99 |
| our LS-THC (K-means points, 4x nao), K and J | ng^2 | K 3.0e-1 | 1.46e-2, 5.3e-4, 1.6e-4, 7.2e-5 | 27, 3, 2 |
| our LS-THC, 16x nao | ng^2 | K 6.4e-2 | 1.46e-2, 1.4e-4, 2.5e-5, 6.2e-6 | 106, 6, 4 |
| SVP-RI NAF-compressed (1e-4: 374 of 420 aux) | | 3.1e-3 | | (no gain) |
| SVP-RI K + **grid-fitted J** (jkfit aux, grid (50,194)) | naux + grid batch | J 3.2e-4 | 1.46e-2, 6.4e-5, 5.8e-7, 5.6e-9 | **227, 111, 103 (= exact J)** |
| grid-fitted J with jfit aux | | J 4.4e-2 | | (= jfit DF) |
| **seminumerical K (level-1 grid) + grid-fitted J** | grid batch | K 7.0e-4 | 1.46e-2, 1.65e-5, 1.65e-8, 4.4e-11 | **884, 996, 377** |
| seminumerical K, level-0 grid + grid-fitted J | grid batch | K 2.6e-3 | SGX0_ROW |
| seminumerical K (level 1) + grid-fitted J, inner_factor 1e-4 | grid batch | K 7.0e-4 | SGX1B_ROW |

The 884/996x rates are at the ~1000x ceiling set by the inner tolerance (1e-3 of the outer gradient); the
inner_factor 1e-4 row measures the surrogate itself.

**Grid-fitted J.** On a grid:
- r_P = sum_g w_g drho(g) V_P(g);
- c = M^-1 r;
- v(g) = sum_P c_P V_P(g);
- J = Phi^T diag(w v) Phi.

V_P(g), the potential of auxiliary function P, is analytic and cheap to recompute (naux x N_grid x ~50 flops: ~1e11 at
Taxol on the (50,194) grid, ~0.1 s FP32). The J increment then rides on the coarse-XC increment GEMMs: one more term in
the grid potential.
- Its accuracy is set by the auxiliary basis, not the grid: the jfit auxiliary basis on the same grid has the same
  4.4e-2 response error as the stored jfit tensor.
- Using the exact (jkfit) auxiliary basis costs nothing in storage and reproduces the exact-J contraction.
- **Memory:** O(naux) + one grid batch, instead of 19.1 GB of host memory at Taxol.

**Seminumerical K (COSX-type).**
- K_mn = sum_g w_g phi_m(g) sum_s A_ns(g) G_sg, with G = D Phi and A(g) the one-electron Coulomb integrals at grid
  point g, evaluated on the fly.
- The surrogate needs no exactness against the DF reference, only a faithful response, and it is better than the
  smaller-auxiliary DF surrogate: 7e-4 vs 3.1e-3, so 4–9x more contraction per exact step.
- **Memory:** one grid batch.
- **Cost [E]:** npairs x N_grid one-electron integrals per evaluation, fused with the G contraction. At Taxol on a
  level-1 grid (~3500 points per heavy atom; ~3e5 points) that is ~5e11 pair-point integrals, ~1e13 FP32 flops. Without
  P-junction screening that is ~2–5 s on a T4 [E], against 10.4 s for the host-streamed FP32 tensor today.
- **Scaling:** with P-junction screening (G_sg is local for a local density) it is asymptotically linear (Neese, Izsak
  and others; sn-LinK on GPUs, Laqua, Kussmann and Ochsenfeld).
- It does not change the converged answer: the exact checkpoints stay DF-K. It only replaces the surrogate.

**Why the operator structure matters (round 7 section 3).** Both new surrogates have the form Phi^T (...) Phi, or an
exact analytic linear functional of the AO functions. Along near-linearly-dependent AO combinations (lambda_min(S)
~1e-5), where sum v_m phi_m ~ 0 everywhere, they are as small as the exact operator. This is the property the
quantised tensors lacked. Pair-space compression (LS-THC) has it too, but our K-means LS-THC needs a rank far above
naux for all-electron TZVP: 16x nao still has a 6 % response error (QR-pivoted points might do better; not tried).
The near-null safety is argued from the operator structure and measured only at phenol's milder conditioning.

**Consequences [E]:**
- **Memory.** At Taxol, the 39 GB of host tensors and their 100 s build go away. GPU memory is set by the exact
  checkpoints: Z = J^-1 (P|ij), naux x nocc^2 = 2.7 GB per operator, plus integral batches. The high-RAM VM is no longer
  needed.
- **Time.** Taxol SCF: -433 s of surrogate build/streaming, +~13 x 3 s of seminumerical K, i.e. 1320 s to ~930 s.
- **Checkpoints.** Two checkpoints at Taxol need 3.8e-2 to 3.16e-5 in one step, i.e. ~1200x. That is impossible
  at inner_factor 1e-3 whatever the surrogate. It needs the better surrogate and inner_factor ~1e-4, which a cheap
  surrogate makes affordable. A stage 1 run with the better surrogate also starts closer: its |g| at checkpoint 0 is
  the surrogate's systematic error. If both hold, another -240 s; projected Taxol SCF ~10–12 min on a T4 [E].

**Asymptotic scaling after this change** (N ~ system size, compact molecules):

| part | today | with grid J + seminumerical K |
|---|---|---|
| surrogate memory | O(npairs naux) ~ N^3 (39 GB at Taxol) | O(N_grid batch) ~ N (screened) |
| surrogate time per inner cycle | O(npairs naux nocc) ~ N^4 | O(npairs N_grid) ~ N^2, ~N with P-junction screening |
| exact checkpoint time | O(naux nao^2 nocc) ~ N^4 (DF hybrid DFT itself) | unchanged |
| exact checkpoint memory | O(naux nocc^2) ~ N^3 (Z) | unchanged (Z can be blocked on the host) |
| number of exact checkpoints | 3 at Taxol | 2–3 |

The remaining N^4 is the exact DF exchange, the cost of standard DF hybrid DFT itself (goal 6: "at least as good as
standard DFT"). Lowering it needs local exchange: localized occupied orbitals and RI-LinK-type sparsity in
X_P,mi = (P|m i), exact up to screening thresholds.
- On compact molecules (Taxol ~15 Å across) the gain is ~2x [E]; it grows with size.
- A seminumerical exact K would change the reference by more than the standard thresholds, so it is used only as the
  surrogate.

## 3. Gradient: why the derivative integrals are slow, and what changed [M]

**Why.** The DF J/K gradient contracts Gamma_P,mn with d(P|mn)/dR over every significant (shell pair, aux shell)
triple: BPA 194k pairs x 1567 aux functions, Taxol 1.33M x 6598.
- **Kernel structure:** a Rys-quadrature kernel per triple class: recurrences over 2D integrals and a contraction of
  nf_i nf_j nf_k Cartesian components x 6 derivative directions x nroots. Translational invariance already removes the
  aux-centre derivative.
- **Precision:** it runs in scalar FP64. A T4 has ~0.25 TFLOP/s of FP64 (1/32 of FP32), and the kernel has no GEMM
  structure. On an A100 (9.7 TFLOP/s FP64) the same pass is ~40x faster, so this is mainly a T4 problem.
- **Two passes:** the FR and LR operators of wB97M-V each need a full pass.

**What changed.** The round-6 FP32 kernel summed its per-(pair, aux-shell) contributions and per-atom atomics in FP32:
millions of additions into a few accumulators with large cancelling terms. `gen_f32.py` now keeps:
- the integrals and per-quartet sums in FP32;
- the sums over aux shells and primitive pairs (`grad_*`, `out_*`) and the per-atom atomics in FP64.

That is a few FP64 operations per (pair, aux shell), so the kernel stays FP32-bound.

| FP32 share of Gamma weight (BPA, `r8_rys_acc64.log`) | DF J/K grad | extra max\|dg\| (round 6, FP32 sums) | vs ref |
|---|---|---|---|
| 0 (FP64) | 4.67 s | 0 | 4.8e-7 |
| 1 % | 4.23 s | 7.5e-9 (~0) | 4.7e-7 |
| **10 % (new default)** | **3.22 s** | **1.6e-7** (2.5e-6) | **4.8e-7** |
| 30 % | 2.69 s | 7.1e-7 (2e-5) | 7.7e-7 |
| 100 % | 2.23 s | 6.2e-6 (2.3e-4) | 5.8e-6 |

**Next options, ranked:**
1. **Find the remaining FP32 error (6.2e-6 all-FP32).** Candidates are the Rys roots/weights (polynomial fits in
   FP32) and the 2a I(l+1) - l I(l-1) cancellation for tight core exponents. Both are cheap to keep in FP64 per
   primitive triple. If the all-FP32 error reaches ~5e-7, the whole pass runs FP32: BPA -1 s more [E].
2. **Fuse the FR and LR passes:** one kernel, two Rys root sets, shared pair data, Gamma loads and atomics. This saves
   the memory traffic and setup of one pass, ~15–25 % of the Rys time [E].
3. **Reformulation onto tensor cores: McMurchie-Davidson/Hermite form of the DF gradient [E].**
   - The pair Hermite transform Gamma~^ab_t,P = sum_mn E^mn_t Gamma_mn,P is a GEMM over all aux functions for each
     shell pair. The aux-side Hermite transform is a GEMM over pairs.
   - What remains per triple is a bilinear form Gamma~_t,tau dR_{t+tau} with the Hermite Coulomb tensor R
     (Boys-function recursion).
   - For a (d d | g) triple with derivative this is ~6k flops + ~5k for R, against ~50k for the Rys contraction. So
     ~4–8x fewer flops in the d/f/g classes that dominate TZVP + jkfit.
   - The two transforms run on tensor cores: FP64 DMMA on an A100, CRT-INT8 on a T4.
   - This is the J-engine idea applied to the K-type Gamma; it is a new kernel family, not a tweak.
4. **Lagged-Gamma fused value + derivative pass:** Taxol only, where the tensor is not stored (report 7).

Gamma-weighted skipping is not an option (round 7: 1e-4 of the weight costs 1.7e-4).

## 4. Trace of BPA after this round and the inefficiency review (`r8_nsys.txt`) [M]

Profiled run: SCF 14.61 s, gradient 4.69 s, dE 1.6e-10, max|dg| 4.9e-7. The GPU was busy 16.4 s of 19.3 s (85 %).

| kernel class | s | launches |
|---|---|---|
| CRT quantize + reconstruct | 3.15 | 12,798 |
| DF/Rys integrals (gradient 1.55, DF builds 0.92) | 2.48 | 14 |
| other (grid weights 0.44, oz1_combine 0.37, AO-deriv kernels 0.79, decompress 0.20) | 2.43 | 11,226 |
| INT8 GEMM | 2.28 | 3,624 |
| FP64 GEMM | 1.81 | 1,983 |
| cupy elementwise / reductions | 1.17 | 43,477 |
| FP32 GEMM | 0.98 | 2,431 |
| cuTENSOR | 0.71 | 5,243 |
| VV10 | 0.68 | 24 |
| eigensolver | 0.52 | 4,258 |

Ranked candidates (BPA unless noted):

| # | item | evidence | est. gain |
|---|---|---|---|
| 1 | **CRT operand preparation costs more than the INT8 GEMMs it feeds** (quant 0.99, quant3t 0.86, recon 0.61, rowexp 0.42, colmax 0.26 s). Fuse: (a) the scale passes (colmax/rowexp) into the quantize kernels (one read of the operand, not two); (b) produce INT8 residues directly in the AO-evaluation kernels for the XC GEMMs; (c) the reconstruction into its consumer (XC elementwise, K accumulate) | trace | 1–1.5 s |
| 2 | **GPU idle 15 % (2.9 s)**: 43k cupy elementwise launches, 12.8k CRT launches. CUDA-graph capture of the inner-loop cycle (fixed shapes) and of the per-batch XC loop; fuse chains of small elementwise ops (`cupy_absolute` 0.27 s is abs().max() scaling: one fused amax reduction) | trace | 1–2 s |
| 3 | gradient FP32 tier beyond 10 % after fixing the remaining FP32 error (section 3) | measured 2.23 s at 100 % | ~1 s |
| 4 | FR + LR gradient passes fused | two `g:DF_JK` ranges, 3.25 s | 0.3–0.5 s |
| 5 | **Gamma built densely (nao^2 x dk) then gathered** to the pair layout (`cp.take`): build it on the pairs directly (sampled GEMM, or the gather fused into `oz1_combine`) | code | 0.2–0.4 s |
| 6 | Becke grid weights (`GDFTgrid_weight_kernel` 0.44 s): all atom pairs per point; skip far atom pairs (cell function exactly 0/1) | trace | 0.2–0.3 s |
| 7 | three DF tensor builds (FR exact 1.1 s, SVP-RI surrogate in stage 1, LR in the first exact K): with grid J + seminumerical K the surrogate build disappears; one build pass could serve FR and LR | NVTX | 0.5–1 s |
| 8 | Taxol direct K: pass 2 recomputes the integrals and the X half transform of pass 1 | code | ~2x on the checkpoint needs storing X (15 GB FP32 per operator: not on a T4; on an A100-80GB it fits) |
| 9 | Taxol host-streamed surrogates: synchronous copies, J read twice, K recomputes B C_ref every call | code | J fixed this round (one pass, double-buffered); K superseded by section 2 |

Items 1, 2 and 3 together cover the 3.3 s still missing for 16 s on BPA [E].

## 5. Changes in this round
- `lib/gvhf-rys/gen_f32.py`, `ejk_int3c2e_ip1_f32.cu`: FP64 accumulators in the FP32 derivative kernel
  (`RYS_F32_ACC64=0` regenerates the old one). `df/grad/rhf.py`: FP64 outputs for the FP32 tier.
- `scf/mixed_precision.py`: `grad_fp32_frac` 0.01 to 0.1 (every `crt`-scheme gradient).
- `scf/multifidelity.py`: the pinned memory pool is released at the end of the SCF.
- `df/int8_df.py`: host-resident J in one streamed pass with double-buffered asynchronous copies
  (`host_block_bytes`); `df/tests/test_int8_df.py` (host vs device vs FP64 DF J).
- `experiments_r8/`: CPU emulations (`r8_thc_cpu.py`, `r8_gridj_cpu.py`), logs, trace summary.

## 6. Not done / next round
- **GPU implementation of the grid-fitted J** (aux potentials on the coarse grid, fused into the coarse-XC
  increment) and of the **seminumerical K surrogate**. The latter needs a one-electron-integral kernel contracted
  with a per-point vector; gpu4pyscf's `int3c1e` kernels have the charge- and density-contracted variants. Then
  Taxol without the high-RAM VM, and BPA without the surrogate tensor build.
- The remaining FP32 Rys error source; fused FR + LR gradient pass; CRT fusion and CUDA graphs (section 4).
- **A100 (goal 4):** still not run; the sm_80 rebuild is needed for the new library code.
TESTS8
