# Research report: GPU4PySCF wB97M-V + DF on INT8/FP32-class hardware (T4 / A100 / H100)

Date of research: 2026-11-04. Scope: new techniques only; items already in `benchmarks/dft/NOTES_next_steps.md`, `PLAN_stage2_algorithms.md`, `RESULTS_mixed_precision.md` are not re-headlined (staged FP32->FP64, Ozaki-I, own CRT prototype, cuBLAS emulation, cuTENSOR plan cache, E2 defect correction, Huang 2601.08077 / Laikov 2511.04169 / Nepal-Wang as *known* references; only new details from them are given).

Legend
* [V] verified = I fetched the primary source (arXiv HTML/PDF text, GitHub README, vendor docs) and read the stated fact.
* [S] secondary = search-result snippet or press text only (not the paper).
* [G] guess / own derivation (arithmetic shown where possible).
* Exact = reproduces FP64 answer to the 1e-9 Eh gate; Approx = changes the answer/functional.
* Helps: T4 / A100 / H100.

State of the repo I found (so the sketches name real code): `gpu4pyscf/lib/crt_gemm.py` (own CRT GEMM; NVRTC-generated quantize/reconstruct kernels with compile-time moduli, per-row power-of-two scaling per call, `plan()` chooses q from `prod(p) > 2 K 2^(2q)`, `MAX_K = 120000`), `gpu4pyscf/dft/numint.py` (`XC_PRECISION='crt'`, `_vmat_block_crt`, `XC_CRT_MODULI=11`), `gpu4pyscf/df/df_jk.py::get_jk` (`K_PRECISION='crt'`; K is already **occupied-RI-K**: K1 `Lij,njk->nikL` is a thin GEMM with N=nocc, K2 `nikL,njkL->nij` is a fat GEMM; both have equal FLOPs 2*naux*nao^2*nocc), `gpu4pyscf/dft/numint.py::_vv10nlc` + `lib/gdft/vv10.cu`, `scf/mixed_precision.py`, `scf/gdm.py`. There is no sgx/COSX/THC/ISDF code in the repo (grep: none), so those would be from scratch.

---------------------------------------------------------------------------------------------------

## 0. Reality check on the goals (from measured numbers + peak specs) [G, arithmetic]

* One modulus = one INT8 GEMM. Rate in "full-precision-GEMM equivalents" = (achieved INT8 TOPS) / n_moduli. **Moduli are not Ozaki-I splits (Huang's table counts splits, ~S(S+1)/2 products).** Operand bits per modulus: `crt_gemm.plan()` gives q = floor((sum log2 p - log2 K - 1)/2), i.e. only ~4 operand bits per modulus (product has 2q bits). Computed with plain numpy from your modulus list:

  | n moduli | q at K=16384 (MGGA block) | q at K=61440 | q at K~244k (K2 nocc*nL) |
  |---|---|---|---|
  | 3 | 4 | 3 | 2 |
  | 4 | 8 | 7 | 6 |
  | 5 | 12 | 11 | 10 |
  | 6 | 16 | 15 | 14 |
  | 7 | 20 | 19 | 18 |
  | 8 | 24 | 23 | 22 |
  | 9 | 28 | 27 | 26 |
  | 10 | 32 | 31 | 30 |
  | 11 | 36 | 35 | 34 |
  | 12 | 39 | 38 | 37 |
  | 13 | 43 | 42 | 41 |
  | 14 | 44 (QMAX) | 44 | 44 |

  So "FP32-class" (24-bit operands, fixed-point relative to the row maximum, not per element) is **n ~ 8**, and "FP64-class" (~44 bits) n ~ 13-14; the module comment (10 moduli ~ 1e-9, i.e. q ~ 32) agrees.
  T4: 130 TOPS peak / 8 = 16 TF-eq (2x the T4 FP32 rate of 8.1, not 4x) and 130/11 = 11.8 TF-eq at n=11 (q=36), 130/13 = 10 TF-eq at q=43; at a realistic 60-70 % of peak (T4 is 70 W power-limited; no measurement) ~ 6-10 TF-eq. A100 native FP64 tensor-core DGEMM is ~15-19 TFLOPS. So **a T4 cannot reach A100-FP64 speed on the GEMM-bound parts by dtype changes alone** (not even at FP32-class precision: 16 TF-eq peak vs ~15-19). Closing the gap needs FLOP cuts (lagged VV10 #5a, occupied-projected Fock #6, K2 SYRK #3, fewer iterations) in addition to better dtypes, and non-GEMM parts (VV10, AO evaluation, quantize/reconstruct) must be as fast as the GEMMs.
* The GEMMul8 numbers show the same picture on big GEMMs: A100 fast-14 = 24.5 TFLOPS vs native ~19.5 (1.25x) [V, 2508.03984 fetched]; GH200 1.4x at 14 moduli [V]; RTX 5080 18.5x [V]. On A100 INT8 emulation only pays at low moduli counts (n <= 8: ~1.8-2x, [G]) and big square shapes; XC shapes (M=N~600, K~16k) are worse. Huang et al. got 1.37x on H100 and 3.0-4.6x on RTX 4090/6000 Ada [V]. **For A100/H100 the realistic wins are launch-overhead removal, VV10 and algorithmic cuts, not INT8.**
* Amdahl base (measured in RESULTS): A100-FP64 Tamoxifen SCF 24.6 s = XC 45 %, NLC 39 % (VV10 pair 31 %), DF get_jk 13 %, DF build 5 %. T4 mixed BPA 87 s = FP64-stage XC ~22 s, FP32-stage XC ~22 s, K 28 s, NLC 12.6 s (VV10 7.7 s).

---------------------------------------------------------------------------------------------------

## Ranking table (expected gain x 1/effort)

| # | Item | Exact? | Helps | Expected gain (workload) | Effort | Evidence |
|---|---|---|---|---|---|---|
| 1 | Energy-keyed CRT precision ladder (= PLAN E5) replacing stage1/1b/FP64 (XC + K on CRT, n = 5..13 moduli; see section 0 table) | exact | T4 (H100 low-n only) | SCF 87 s -> ~50-60 s on T4 [G]; stage 1b removal is a hypothesis to test | M | V (Huang schedule, cuEST), CRT code untested on GPU |
| 2 | CRT kernel engineering: Cauchy-Schwarz/scale-invariant scaling, fused quantize (form B inside kernel, share A planes with rho), batch the n modulus GEMMs, K limit 2^17 | exact | T4 | 1.3-2x on the CRT GEMM path [G] | S-M | V (2602.02549, 2508.03984, 2606.29129) |
| 3 | K-build: pre-quantized static DF residues + SYRK-type K2 + memory plan | exact | T4 | K 28 s -> ~10-14 s on T4 [G] | M | V/G |
| 4 | XC launch-overhead removal (CUDA graphs / grouped GEMM / fused rho kernel) | exact | A100, H100 (T4 a bit) | A100 XC 0.43 s -> ~0.25-0.3 s; SCF -15-25 % [G from idle share measured] | M | measured idle 50 % |
| 5 | Lagged / inner-outer VV10 potential + symmetric pairs + tile far-field skip | exact (lag: converged answer unchanged) | all, A100 most | NLC share 9-39 % -> 3-15 %; SCF -5..-20 % [G] | S-M | G |
| 6 | Occupied-projected Fock for XC/NLC (V C_occ, 0.14x FLOPs) with direct-minimization (GDM/L-BFGS) SCF | exact | T4 (GEMM-bound) | XC-GEMM FLOPs per iteration x0.14, maybe 1.5-2x more iterations => 2-4x on XC GEMM share [G] | M-L | G |
| 7 | Test cuBLAS 13.4 auto Ozaki-II / `CUBLAS_EMULATION_STRATEGY` on H100/A100 for K | exact | H100 (A100 marginal) | 0-1.4x of K only | XS | S (press) |
| 8 | CUTLASS-style fused 3xTF32 / split-FP16 GEMM with outside-TC accumulation (A100/H100 early cycles; T4 FP16) | exact-ish (FP32-class) | A100/H100 early cycles | <= 2x on FP32-class GEMMs of early cycles; dominated by #1 on T4 | L | V (Ootomo-Yokota, in repo notes) |
| 9 | Laikov DF-VV10 (v2 GTO-fitting version) | **Approx (changes functional)** | all | NLC -> ~0 | L | V (read), no timings published |
| 10 | THC/ISDF/LS-THC K, COSX/sn-LinK | Approx | - | none at 50-100 atoms vs occ-RI-K+INT8 [G arithmetic] | L | V (1911.00470, 2508.19212) |
| 11 | TF32/INT8 for DF 3-center / ERI evaluation | - | - | DF build is 5 % of A100 SCF | L | S |
| - | Dead ends: INT4 CRT, FP8 Ozaki-II (no T4 FP8), NTT, K-only INT8 as final answer | | | | | see section 11 |

---------------------------------------------------------------------------------------------------

## 1. Energy-keyed CRT ladder replacing the stage-1/1b/FP64 hard split  (rank 1)

**What.** Use one engine (CRT INT8, `lib/crt_gemm.py`) for *both* the "FP32-like" and the "FP64" phases of the SCF, selecting the modulus count n from the relative energy change, never reverting. This replaces `scf/mixed_precision.py`'s stages (FP32 K / FP32 XC with a hard plateau detection, stage 1b, FP64 polish).

**Why it should beat the current staged scheme on T4.**
* The measured FP32-XC problem (RESULTS "noise floor"): FP32 Vxc-matrix accumulation over the 4096-point block gives error linear in chunk length and makes the FP64 gradient at the FP32 solution ~4e-5, so 2-3 FP64 builds always remain. INT8 residue GEMMs accumulate **exactly in INT32 over the whole K** (no accumulation error at all); the only error is operand quantization (q bits per operand) which is relative to the largest element of each row/column, i.e. the natural DFT error measure. So the "chunk-length" floor disappears and stage 1b becomes unnecessary [G, but follows directly from the design].
* CORRECTION (advisor review): Huang's "3 splits = 23 bits" counts Ozaki-I splits. In CRT terms the table in section 0 applies: ~24-bit operands need n ~ 8 moduli (16 TF-eq peak on T4 = 2x its FP32 rate, not 4x), ~12 bits (q=12, enough for the first iterations, error ~2^-12 relative to row max) need n=5, and FP64-class (q~36-44) n=11-14. A ladder n = 5 -> 8 -> 11 -> 13 therefore saves 2.2x / 1.4x / 1.2x of GEMM time vs n=11, and vs the existing FP32 stage the GEMM-time gain is ~2x (T4: 16 vs 8.1 TF-eq peak) only if the INT8 kernels run near peak. **Most of the value of this item is (i) the FP64-stage speedup (the current FP64 XC 22 s + K at FP64 stage become n=11-13 INT8 at ~10 TF-eq vs 0.25 TF native), and (ii) the possible removal of stage 1b/the FP64 polish.** I withdraw the earlier "SCF 87 -> 45-60 s" figure; a defensible estimate is: FP64-stage XC 22 s -> ~3-4 s, FP32-stage XC 22 s -> ~10-15 s (2x), K 28 s -> ~14 s => SCF ~87 -> ~50-60 s [G, assumes ~60 % of peak INT8].
* Stage 1b elimination is a **hypothesis to test, not a result**: CRT removes accumulation error but quantizes in fixed point relative to each row's maximum (FP32 keeps 24 bits relative to each element). For peaked AO rows or tails that trade is not obviously a win. Decisive measurement: run `benchmarks/dft/xc_error_split.py` (fixed density) with the CRT vmat at n = 8-10 and compare |dg_ov|_F against the FP32 Vxc-matrix-stage offset of 2.0e-5 (RESULTS). Only if CRT n<=10 is clearly below that does stage 1b go away.
* Both the cuEST docs and Huang et al. recommend exactly this schedule: low precision early, high late [V]. NOTE: this item is essentially PLAN E5; the new content is only (a) running the *low-precision* phase on CRT as well, and (b) dropping stage 1b. Status: `XC_PRECISION='crt'` and `K_PRECISION='crt'` already exist in the repo (git log: "untested on GPU yet"), so the next action is the measurement in section 12 step 1.

**Verified schedule to copy** (Huang, Shao, Hammond, arXiv:2601.08077v3, https://arxiv.org/html/2601.08077v3, read in full [V]): trigger = relative |dE/E| between iterations; thresholds >= 2e-6 -> 23 bits (3 INT8 splits); [5e-9, 2e-6) -> 31 bits (4); [2e-11, 5e-9) -> 39 bits (5); [7e-14, 2e-11) -> 47 bits (6); below -> FP64; never revert. Fixed level-4 stalls at ~1e-5 Eh while the adaptive schedule reaches ~1e-9 Eh; iterations same or +1 on RTX cards [V]. Their map to moduli (Ozaki-II needs ~2x splits worth of bits: 8 moduli ~ 5 splits ~ |g| 1e-7 per cuEST docs https://docs.nvidia.com/cuda/cuest/gemm_emulation.html [V]; GEMMul8 README says 14-15 moduli for cuBLAS-55-bit accuracy [V]). Your own `XC_CRT_MODULI` comment says 10 -> 1e-9, 12 -> 3e-11, 14 -> 5e-13: use that table, not Huang's split table, for n.
Important difference to Huang: they only emulate K; J and XC stay FP64 and they say XC in FP32 "could become the bottleneck once K is accelerated" [V, future work]. Your work already covers XC, which is the larger share.

**Sketch.**
* `scf/mixed_precision.py`: replace the stage state machine by a `PrecisionLadder` object with `.level(dE_rel)` -> `(n_xc, n_k, n_vv10_precision)`; set `numint.XC_CRT_MODULI`, `df_jk.K_CRT_MODULI`, `numint.VV10_PRECISION` before each `get_veff` (hook point: the `old = (numint.VV10_PRECISION, ...)` block at lines ~82-89). Keep the energy-keyed thresholds as a table and calibrate on BPA/tamoxifen; never revert.
* Keep `final_grad_factor`-style tightening: the last 1-2 iterations at n >= 12 (cheap because few).
* VV10: switch to FP64 float-float only if the final-energy error requires (FP32 VV10 was already 1e-9-exact).
**Gain.** see the corrected estimate above (SCF ~87 -> ~50-60 s on T4 [G]). H100: low-n early iterations only (<= ~1.4x on K, Huang). A100: none.
**Risks.** (a) Energy-driven switching has no formal guarantee [V, Huang limitations]. (b) the 1e-9 gate for XC: per-row scaling bounds error relative to row-max; AO rows that are tiny in a block are poorly represented, but their contribution is correspondingly small (absolute error of Vxc_ij ~ 2^-q * rowmax_i * colmax_j * K^{1/2}); verify on fixed-density regression. (c) mixed gradient quality: keep final stage at n>=12. (d) wide-dynamic-range B operand: huge wv (vrho, vtau) values at a few low-density tail points set the row maximum of B and crush the bits of the other points; this is the case where fast-mode (Cauchy-Schwarz) scaling is weak and accurate mode (extra INT8 GEMM) or clipping/pre-normalizing wv per block may be needed [G; 2602.02549 abstract: fast mode degrades with wide exponent spread]. Test with tail-heavy systems (anions, diffuse functions).

---------------------------------------------------------------------------------------------------

## 2. CRT kernel engineering from the Ozaki-II literature  (rank 2)

All of these refine `lib/crt_gemm.py`; each is small.

**2a. Scaling by Cauchy-Schwarz (fast mode) instead of `K*2^(2q)`; use the scale-invariant formula.**
* `plan()` currently needs prod(p) > 2 K 2^(2q) (worst case: every product term at max magnitude). Ozaki-II "fast mode" picks per-row/column scales so that the Cauchy-Schwarz bound ||a_i||_2 ||b_j||_2 (computed with round-up arithmetic) satisfies the CRT uniqueness condition, with **no extra INT8 GEMM**; "accurate mode" estimates the bound with an extra INT8 GEMM and is more robust for wide exponent spread [V: arXiv 2508.03984 https://arxiv.org/html/2508.03984v2 ; error analysis 2602.02549 https://arxiv.org/html/2602.02549v1].
* Kawakami & Takahashi (arXiv 2606.29129, https://arxiv.org/abs/2606.29129, abstract read [V]) show the original fast-mode formula is **not scale-invariant** (multiplying inputs by a constant changes the effective integer bit width, can degrade accuracy or make CRT recovery fail) and give a scale-invariant formula derived from the CRT uniqueness condition at no extra cost, with accuracy comparable to accurate mode. If you adopt CS scaling, use that formula (the AO values / weights have wildly different magnitudes across blocks).
* Why it can save ~1 modulus here [G]: AO rows are strongly peaked on a few grid points of the block (n_eff << K = 4*4096), so ||a||_2 ||b||_2 << K max|a| max|b|; with n_eff ~ 100-300 vs K = 16384 the bound on P loosens by 6-7 bits, but q is only half of the leftover bits, so that is ~+3 bits of q at equal n (about one modulus saved at equal accuracy, ~10 % of the GEMM time). Measure the distribution of ||a_i||_2/(sqrt(K) max|a_i|) on real BPA blocks first.
* Error bound structure for deciding n (Theorem 2 of 2602.02549 [V]): |AB - C| <= t|A| v(2^beta')^T + t 2^alpha' v^T|B| + ..., t = 1/sqrt(2^5 (P-1)). So error is **proportional to row-sum |A| times column-max** (componentwise-ish, not just normwise) and each extra ~8-bit modulus multiplies P by ~2^8, shrinking t by 2^4 (**i.e. each modulus buys ~4 bits of relative accuracy in the bound, not 8**; the observed bits-per-modulus in practice is better than the bound; GEMMul8 table of effective bits is in its README [V]).
* **Global scale across blocks (your next-step item 2):** not needed for correctness: per-row scaling means each block yields an exactly reconstructed FP64 block result (recon cost ~5 % of GEMM by my estimate [G]), and summing FP64 block results is fine. Accumulating residues across blocks under one global per-AO scale would work (residue sums are in the ring) but costs bits in blocks where an AO is small; gains only the (small) reconstruction cost. Not recommended until profiling shows reconstruction > 15 % of the time.

**2b. K limit.** Ozaki-II assumes K <= 2^17 = 131072: with symmetric residues |x| <= 128 and p1 = 256, INT32 wrap at K = 2^17 is harmless because rmod(+-2^31, 256) = 0 [V, 2508.03984]. Your `MAX_K = 120000` is slightly conservative; raise to 131072 when the first modulus is 256 (MGGA block K = 4*4096 = 16384 is nowhere near; matters only for K-build shapes with large nL*nocc). For K > 2^17 the paper only says "block the matmul" [V].

**2c. Conversion kernel.** GEMMul8 replaces fmod by an FMA chain with precomputed reciprocals and INT32 mod by mulhi+conditional subtract [V]. Your NVRTC kernels already bake moduli in as compile-time constants (compiler emits mul-shift), so that trick is already in effect. What remains:
* Fuse B-formation into the quantizer: `_vmat_block_crt` materializes `b = cupy.empty((4,nao_sub,ng))` in FP64 (write + read of 8 B/element on a 320 GB/s T4, with `_scale_ao` and `cupy.multiply` launches) and then quantizes it. Instead pass `ao` and `wv` into `quant` and compute `ao_k * w_k` inside (per row/component) [G].
* (Idea withdrawn/caveated after review.) Reusing the quantized ao planes for the rho contraction (`C_occ^T ao`) is NOT recommended as is: the CRT path deliberately keeps rho in FP64. Fixed-point psi = C^T ao loses relative precision in low-density tails, which feeds tau and grad rho into wB97M-V's iso-orbital indicator and into large vrho/vtau values; the 1.8e-7 FP32-rho figure (xc_error_split) is a floating-point result and does not carry over. Only try it with a tail / low-density regression test.
* Reconstruction: GEMMul8 sums the UINT8 residues U_i weighted by constants split into two FP64 pieces so the sum is exact, with P as a double-double, one kernel for all i (O(n) per element) [V]. Your Garner reconstruction is O(n^2) modular multiplies per element. Estimated share is small for K=16384 blocks (~5 %) [G] but it grows for the K-build (K1/K2 outputs per block) and for small K; port only if the profile shows > 10 %. On T4 the FP64 FMAs of that scheme run at 1/32 rate, so keep the integer path there (int64 two-limb).
* Planes/streams: issue the n modulus GEMMs as **one strided-batched INT8 GEMM** (`cublasGemmStridedBatchedEx`, 8I inputs, 32I compute) instead of n calls: for M = N ~ 300-600 a single 128x128-tile GEMM launches only 6-25 CTAs on T4's 40 SMs (underfilled), batching over n=8-11 planes fills the chip [G]. Add achieved INT8 TOPS per call to `profile_xc_kernels.py` as the notes already plan.

**Gain.** 1.3-2x on the CRT GEMM path [G]; the uncertain part is how far current kernels are from ~60 % of 130 TOPS, which only an nsight run answers.

---------------------------------------------------------------------------------------------------

## 3. K-build on CRT: static residue DF tensor, thin-K1 problem, SYRK for K2  (rank 3)

**Facts from the code** (`df_jk.get_jk`, read): per aux-block `cderi (L,nao,nao)`: K1 `Lij,njk->nikL` is [L*nao, nao] x [nao, nocc] with **N = nocc = 61 (BPA)**, output rhok; K2 `nikL,njkL->nij` is [nao, nocc*L] x [nocc*L, nao]. FLOPs equal (2*naux*nao^2*nocc each). For BPA ~ 9.5e10 FLOP/K build [G, naux ~ 1000-1200].
* Your Ozaki-I result: K1 only 1.6x at S=6 and 0.95x at S=8 (RESULTS): because K1 is a thin GEMM where each cderi element is used for only 2*61 MACs, so quantizing/reading the n planes costs as much as the products. Arithmetic: native FP64 on T4 needs 15 FLOP/B * 0.32 TB/s = 4.9 TF to be memory-bound, far above 0.25, so FP64 native is compute-bound; the INT8 path with n=8 and **pre-quantized planes** needs 8 B/element read (same bytes as FP64) for 8*122 = 976 INT8 ops per element -> 122 ops/B -> bandwidth-limited at ~39 TOPS = 30 % of peak [G]. Without pre-quantization the quantize kernel adds the FP64 read + n-byte write per element per K build, doubling traffic.
* GEMMul8 exposes exactly this optimization: `GEMMUL8_SKIP_SCALE_A/B` reuses scaling/residue work for operands that do not change [V, README https://github.com/RIKEN-RCCS/GEMMul8]; and it has **SYRK/SYR2K/SYMM** emulation [V] (supports CUDA >= 12.9; Turing not mentioned -> unconfirmed; do not build it on the runtime, the classifier refused; use it only as a reference design).

**Sketch.**
1. In `df_jk.py` quantize the (static) DF tensor block once per SCF (after `dfobj.build`) into `n_max` residue planes in int8 with one scale per aux-function row; store planes instead of FP64 `_cderi` blocks **only for the K path**. Memory: 11 planes = 11 B vs 8 B/element (1.4x) -> the Tamoxifen 7.4 GB tensor would not fit. Options: keep only `n_max = 8` planes (precision ladder tops out at 8 for K; the FP64 final cycles use the FP64 `_cderi`, i.e. both copies) or keep residues on-device and `_cderi` FP64 in pinned host memory (`dfobj.loop` already streams blocks) [G]. The K tensor in "mix_inside_kernel" mode halves this but is an approximation (E3).
2. K2 symmetric: for `dm_factor_mode == 0` (hermi, same factor on both sides) K2 = R R^T is symmetric; compute only upper-triangle output tiles (SYRK-style tile loop over 128x128 tiles using the batched INT8 GEMM). Exact 25 % of total K FLOPs (E6a already planned; Huang explicitly did not do it because their emulation lacked SYRK [V]).
3. Reorder to make K1 thicker: for a block of aux functions compute K1 for **both** the full-range and long-range K simultaneously (two DF tensors, same C_occ) sharing the quantized C_occ planes. Small.
**Gain.** T4 K 28 s -> ~10-14 s [G]. Risks: memory (above); a quantization scale per DF row: DF rows (aux functions) with huge dynamic range across ij (core-like aux functions) lose bits relative to the row-max; verify with fixed-density test (Vk error vs FP64) at n = 6..10.

---------------------------------------------------------------------------------------------------

## 4. XC loop launch-overhead removal  (rank 4, the A100/H100 item)

**Measured** (RESULTS): one A100 `nr_rks` = 0.49 s wall but 0.245 s GPU time; ~100k Python calls per XC evaluation; plan cache already took it to 0.43 s.
**Options (all exact).**
1. CUDA graphs: shapes (nao_sub, ng, xctype) repeat across blocks and SCF cycles (the cuTENSOR plan cache relies on this); CuPy supports stream capture (`stream.begin_capture()`; `graph.launch()`), so capture the per-block "rho -> libxc input -> vmat" pipeline once per distinct shape and replay with updated pointers [G; pointer updates via cudaGraphExecKernelNodeSetParams or by capturing against fixed workspace buffers into which inputs are copied]. cuTENSOR plans are capturable in principle; cuBLAS under graph capture needs a user-set workspace (`cublasSetWorkspace`) [G; the cuBLAS page I read only covered the workspace requirement for emulation reproducibility].
2. Batch over blocks with cuBLAS grouped GEMM (`cublasGemmGroupedBatchedEx`): I could not verify supported types (FP64/INT8) or the minimum CUDA version from the docs I could fetch [unverified]; CUTLASS grouped GEMM only on H100+/Blackwell per Huang [V]. Treat as a research probe: one grouped call = all blocks of a given AO-subset size.
3. Fuse the MGGA rho path: ~14 small kernels/block (RESULTS) into one custom kernel `rho,sigma,tau` from `[c0..c3]=C^T[ao0..ao3]` (one GEMM with stacked AO, as already planned) + one elementwise kernel. This is the item in NOTES step 3; I add only the CUDA-graph route.
**Gain.** A100 SCF -15-25 %, H100 more (higher FP64 TC, even more launch-bound) [G]. Zero effect on accuracy. This is the only large item that helps A100/H100 with no precision trade-off.

---------------------------------------------------------------------------------------------------

## 5. VV10: lagging, symmetric pairs, far-field tiles  (rank 5)

Source facts: exact kernel and cost structure confirmed in code (`numint._vv10nlc`, `vv10.cu`: omega/kappa per point, pair kernel computes U, W, E sums over all NLC grid pairs). Terco (JorgeG94/terco PR #91, https://github.com/JorgeG94/terco/pull/91 [V]) screens with a single `VV10_RHO_TOL` shared by energy and gradient, and reports VV10 ~25 % of GPU gradient time (15 atoms) [V]. No source I found gives distance-cutoff results for VV10 [searched; negative].
* **5a. Lagged NLC potential (SCF-level, exact at convergence) [G].** V_nlc depends smoothly and weakly on D. Re-evaluate the O(N_grid^2) pair sum only when |D - D_nlc_last| exceeds a threshold tied to the current gradient (e.g. recompute when |dD| > 0.3 * |g|-scaled tolerance, or every 2nd-3rd iteration), reuse V_nlc otherwise; always recompute at the end and for the convergence check. Different from E2 (defect correction for rounding noise, which failed): this is a deterministic smooth functional, error first order in |dD| only for the *lagged* iterations. Saves ~50 % of NLC calls. Test: count iterations and final dE vs lag policy. In `rks.get_veff` there is already `dm_last/vhf_last`; extend with an `nlc_cache`. Gain: A100 NLC is 39 % of SCF -> ~ -15 %; T4 -4 %.
* **5b. Symmetric pair evaluation** (~1.5x fewer ops) is already NOTES step 4; U_i/W_i/E_i are symmetric-friendly only for E; W and U use rho_j weights: a tile (i-block, j-block) computing both i and j accumulations in one pass needs atomics or two accumulators; fine in shared memory.
* **5c. Far-field tiles [G].** Phi -> -3/(2 w_i w_j (w_i+w_j) R^6) for R^2 w >> kappa: asymptotically separable apart from (w_i+w_j). Sort NLC points spatially (Morton) into tiles of 128; for tile pairs whose separation exceeds R_c use a centroid monopole (+dipole) with tile-mean omega, kappa; error O((tile radius/R)^2). Savings depend on molecule extent: 50-100 atoms (~15-25 A) probably skips 30-60 % of tile pairs at R_c ~ 10 A [pure guess; measure Phi tail magnitude on tamoxifen]. Gate: dE_nlc < 1e-9 Eh; gradients need the same approximation. Medium effort.
* **5d. Interpolation-in-q (Roman-Perez & Soler; Sabatini, Gorni, de Gironcoli rVV10, PRB 87, 041108 (2013), abstract only [S])** makes the kernel separable in (q_i, q_j) so the pair sum becomes N_q^2 convolutions -> FFT on a uniform grid. This is the standard solid-state route; for atom-centered Becke grids you would have to map density to a uniform grid (accuracy for core regions is the problem) or use DF-style Gaussian expansion. I rate it unsuitable for 1e-9 Eh molecular work; no result found for Gaussian-basis codes.
* **5e. Exactness negative result [G, algebra].** Exact VV10 kernel 1/(g g'(g+g')) with g = w R^2 + kappa has Laplace representation int e^{-a(t1+t3)-b(t2+t3)} dt, with a = w_i R^2 + kappa_i: the Gaussian exponent multiplies R^2 by (w_i(t1+t3) + w_j(t2+t3)), a sum, so exp(2 a_ij r_i.r_j) does not factor into low-rank (i)x(j) pieces. No exact GEMM reformulation exists; only the w_i ~ w_j approximation of Laikov does.

---------------------------------------------------------------------------------------------------

## 6. Occupied-projected XC/NLC Fock + direct-minimization SCF  (rank 6, research-grade but exact)

**Idea [G].** For MGGA/GGA the Vxc matrix V = sum_g ao^T (W o ao) costs nao^2 ng. The *applied* operator V C_o = ao^T (W o (ao C_o)) = ao^T (W o psi_o) costs nao*nocc*ng, i.e. nocc/nao = 0.14 of the dense GEMM (the same trick you already used in `grad/rks.py`, "occupied factor"). The orbital gradient g_ov = C_v^T (F C_o) needs only F C_o; the energy needs only rho; the final gradient needs only D. A solver that never forms the full F therefore needs: XC/NLC GEMM FLOPs x0.14 per iteration; K already scales with nocc; J is cheap.
**Where.** `scf/gdm.py` (geometric direct minimization with L-BFGS; `kappa_gradient`, `pseudo_canonicalize`, `approximate_optimal_diagonal_hessian`) currently call `get_fock` -> full F. Changes: (i) a `get_veff_occ(dm, C_o)` returning V C_o via the existing per-block AO/weights loop (one extra GEMM with N=nocc instead of nao); (ii) gradient from F C_o; (iii) the preconditioner/diagonal Hessian needs F_oo and F_vv diagonals: F_vv diag = sum_mu... needs the full V diagonal in the MO basis; approximate with the last full-F eigenvalues refreshed every ~5 iterations. DIIS cannot be used (it extrapolates full F); plain GDM typically needs more iterations than DIIS.
**Gain estimate.** If GDM needs 1.5-2x iterations: vmat GEMM share (the `vmat contractions ~3.5 s` of 11 s XC on A100; GEMM 95 % on T4) drops ~ (1.5..2) * 0.14 = 0.2-0.3 of baseline => XC GEMM 3-5x on T4 [G]. On A100 the XC is launch-bound, so no gain there (do #4 first). Risk: convergence robustness of GDM for MGGA with VV10 (non-convex landscapes) and loss of DIIS; also the CRT/INT8 path then multiplies: thin N = nocc GEMM has the same thin-GEMM memory-bound problem as K1 (section 3). Hence do #1-#3 first; revisit if XC remains the bottleneck.

---------------------------------------------------------------------------------------------------

## 7. cuBLAS 13.4 Ozaki-II auto-selection (rank 7, one afternoon on H100/A100)

* Press release reprinted by PC Watch (Shibaura Inst. of Technology, https://pc.watch.impress.co.jp/docs/release/atpress/2144360.html [S]): CUDA Toolkit 13.4 (Sept 2026) adopts Ozaki scheme II for DGEMM and ZGEMM, supported on Ampere and later, auto-selected when faster than Ozaki-I; B200 up to ~175 TFLOPS DGEMM. Turing not mentioned (consistent with cuBLAS fixed-point emulation table "8.x, 9.0, 10.x, 11.0, 12.x", CUDA 13.0u2+ [V, cuBLAS docs]).
* Controls [V, cuBLAS docs]: `cublasSetEmulationStrategy` (DEFAULT/PERFORMANT/EAGER, env `CUBLAS_EMULATION_STRATEGY`), dynamic mantissa (default), fixed mantissa bit count via `cublasSetFixedPointEmulationMaxMantissaBitCount` (default 55), offset, workspace <= 8 GB.
* Action: on H100 (INT8 ~24x FP64 TC) run `gemm_precision_bench.py` K/XC shapes with fixed 31/39/47 bits; compare with your own Ozaki-II. On A100 your own RESULTS show Ozaki-I emulation 2-4x slower than native, so expect <= break-even at 47+ bits.

---------------------------------------------------------------------------------------------------

## 8. FP32-class GEMM with outside-TC accumulation (rank 8)

* Already in notes: Ootomo & Yokota (arXiv 2203.03341) and your E1 result (in-TC accumulation = RZ error; accumulating outside every kc K-steps restores FP32 accuracy but the cuBLAS-call loop only reaches 0.5-2 TFLOP/s).
* New pointer [G, from memory, not verified here]: CUTLASS ships a fast-accurate 3xTF32 GEMM example (examples/27_ampere_3xtf32_fast_accurate_tensorop_gemm) and `OpMultiplyAddFastF32/F16/BF16` mainloop operators; test it on A100 directly against your "hand-split" error (1.6e-5..2.8e-5) before writing a kernel. T4 has no TF32; the analogue is split-FP16 (3 products, FP32 accumulate), 65/3 = 21.7 TF vs 8.1 TF FP32 = 2.7x peak, but FP16's 5-bit exponent requires per-row/column scaling like Ozaki, and the in-TC accumulation (RZ) problem applies (needs the same fused outside-TC accumulation kernel). INT8-CRT at FP32-class precision (n ~ 8, 130/8 = 16 TF-eq peak) is **comparable to / below split-FP16 at peak** but has exact INT32 accumulation and shares the code with the FP64 path: measure, do not assume CRT wins. Recommend only for A100/H100 early cycles (where FP64 TC = FP32 SIMT, so only TF32 TC can win).

---------------------------------------------------------------------------------------------------

## 9. Laikov DF-VV10 (rank 9, approximate functional)  [V: https://arxiv.org/html/2511.04169v2 read in full text form]

* Formulas [V]: kernel approximated with w_1 ~ w_2 (eta = rho/omega^{3/2}, mu = kappa/omega), expansion around fixed length scale beta = 16 bohr^2 (sqrt(beta) = 4 bohr, "system-independent"), tanh smoothing alpha = 4; E6 = sum of **7 bilinear terms** U^{(n)}_{ij} = q_i^T u^{(n)} q_j with q_0 = eta, q_1 = eta mu/beta, q_2 = eta (mu/beta)^{1/2}, q_3 = eta (mu/beta)^{-3/4} and u_n(r) = beta^n/(r^2+beta)^{3+n}, n = 0,1,2. Noble-gas dimers: bond lengths ~1 % (He2 3.3 %), well depths off by up to ~14-18 % (Eq. 12 and 29) [V]. No refit; no timings published ("a fraction of the overall cost") [V].
* v2 detail [V]: densities q_i fitted in an **atom-centered Gaussian aux basis**, q = S^-1 p with p_l = sum_k w_k b_l(r_k) q(r_k) (one GEMV over grid x aux), the six-dimensional integrals u^{(n)}_{kl} = <b_k|u_n|b_l> are analytic Coulomb-like Gaussian integrals (special function U_mn evaluated by double-exponential quadrature). After that U_ij = q_i^T u q_j = dense aux-basis GEMV/GEMM: **the only GEMM-able exact-ish VV10 reformulation I found**, nothing remains O(N_grid^2). The matrix u^{(n)} (naux x naux, static per geometry) is built once per SCF like the 2-center metric.
* Gradient requires derivative integrals of the same kind. Because the answer differs from VV10 by up to 18 % in dimer well depths, wB97M-V (parameters fitted with exact VV10) would no longer be wB97M-V; keep as opt-in `nlc='laikov'` research branch, never a default [G, with Laikov's own numbers as evidence].
* Effort: new Gaussian integral code for U_mn (use `gpu4pyscf/lib/gint` int2c2e-style machinery as template: `df/int3c2e_bdiv.py::int2c2e` exists) + fit + grid-response gradient: weeks.

---------------------------------------------------------------------------------------------------

## 10. Exchange alternatives and integral evaluation (rank 10-11): why not now

* **THC/ISDF for exchange.** Lee, Lin, Head-Gordon (arXiv 1911.00470, text read [V]): THC-RI-K cost O(nao * N_IP^2) with N_IP = c_ISDF * N_aux; recommended c_ISDF = 6.5 for wB97X-V [V]; errors ~1 kcal/mol class RMSD in W4-11 atomization energies [V]; crossover vs RI-K at about 100-200 electrons for exchange on CPU, "obtaining the THC factorization takes most of the time" [V]; THC-RI-K is not variational (collapses at small c_ISDF) [V]. Hillers-Bendtsen & Martinez (arXiv 2508.19212, TeraChem, GPU, text read [V]): 2-4x faster K than **integral-direct** (not vs DF), THC built in <8 s for (H2O)100, relative-energy errors "below 7.0e-? Eh" (exponent garbled in the PDF text extraction; kcal/mol-level) on proteins up to 3000 basis functions, **density-difference ansatz with a full-precision reference iteration** [V].
  Arithmetic for our case [G]: BPA nao=623, nocc=61, naux ~1000-1200: occ-RI-K = 2 x 2*naux*nao^2*nocc ~ 1e11 FLOP; THC-RI-K with N_IP = 6.5*naux ~ 7000: 2*nao*N_IP^2 ~ 6e10 plus factor application and the one-off ISDF build. So no FLOP win at 33-57 atoms; the 3x INT8/FP32 accelerated occ-RI-K is better and exact. Rank low; only for >=200-atom systems where K dominates (and then as ACE-like approximate preconditioner with exact refinement).
* **COSX / sn-LinK / seminumerical exchange.** No GPU4PySCF implementation; approximate (grid error ~1e-5..1e-4 Eh, needs refinement iterations). I did not fetch any sn-LinK or COSX paper in this session; claims from memory are not included. Not recommended against an exact INT8 occ-RI-K.
* **Integrals as GEMM / tensor cores.** Asadchev & Valeev matrix-formulated McMurchie-Davidson ERI on GPUs, claimed up to 15x efficiency for high-l [ii||ii] [S snippet only, not read]. Relevant only to 4-center/3-center evaluation of **high-angular-momentum** shells; in our workload DF tensor build is 5 % of A100 SCF (1.2 s of 24.6 s, RESULTS) and it is needed twice (full+LR). Rank low. A search-result snippet [S] says Huang et al.'s related-work section states no prior ERI work used GEMM units.
* **XC on tensor cores / TPU.** I found no literature on reduced-precision XC quadrature on tensor cores beyond yours (searches negative; Yasuda 2008 single-precision GPU XC is known to me from memory but not re-verified here). Pederson et al. "Large scale quantum chemistry with TPUs" (arXiv 2202.01255, JCTC 19, 25 (2023), ar5iv text read [V]): FHI-aims NAO basis, **only the O(N^3) density-matrix step runs on TPU** (Newton-Schulz S^-1/2, hole-particle canonical purification, SUMMA matmuls in JAX); H/S built on CPU; precision: FP32 early iterations, emulated 57-bit FP64 later (11x slower matmuls), switching on relative density/energy change with eps = 5e-7; S^-1/2 always in double and cached; ~10 TFLOPS per core matmul, 93 % of single-core peak [V]. Takeaways: (i) validates the staged/ladder SCF; (ii) no TPU code for XC/integrals exists in that line; TPU-based Gaussian DFT with ISDF/THC: searches found none (GS-DFT arXiv 2609.31483 is JAX but TPU use not stated [S]).

---------------------------------------------------------------------------------------------------

## 11. Dead ends and corrections to the prior plan (worth knowing before spending GPU time)

1. **INT4 CRT on T4.** Turing has INT4 tensor cores (260 TOPS). Moduli must be <= 16 (~3.7 bits) vs 8 bits for INT8, so you need ~2.2x more moduli for the same precision, exactly cancelling the 2x peak; no cuBLAS INT4 GEMM (WMMA s4 only). Skip [G].
2. **FP8 Ozaki-II** (Uchino, Ozaki, Imamura arXiv 2603.10634 [S]; GEMMul8 README: FP8 needs ~10-11 moduli for 55-bit vs 14-15 for INT8 [V]; "Ozaki 2.5" arXiv 2609.09095 model-only projections for Rubin [V, abstract, "every result is a model projection pending measurement"]): needs FP8 TCs (Hopper+); T4/A100 have none. Irrelevant to your target hardware. Relevance: its observation that *residue conversion* ("deconstruction") on integer pipes is the real bottleneck confirms section 2c; its "convert-once residue workspaces" = section 3.
3. **Complex/Karatsuba Ozaki-II** (2512.08321 [S]) irrelevant (real GEMMs).
4. **Global fixed-point scale across XC blocks** (NOTES step 2): not needed (section 2a).
5. **K-only emulation as end-to-end speedup.** Huang et al. speed up only K and get 3.0-4.6x on RTX 4090/6000 Ada but only 1.37x on H100 [V]; on our T4 K is ~32 % of the mixed-SCF time and XC dominates, so K-only caps SCF gain at ~1.3x.
6. **NTT.** unchanged from notes (no true convolutions).
7. **Nepal & Wang (arXiv 2603.29975, read [V])** is LSMS (multiple-scattering, ZGEMM/ZTRSM, GB200), not Gaussian DFT, and reports **no analysis of dynamic range, scaling or k**; 31-bit/10-moduli mode failed to converge their SCF, 47+ bits fine, 1.7x from GEMMul8. Use only as a weak "robust energies under emulation" precedent (their variational-robustness argument is the same as yours).
8. **E2-like ideas for VV10:** unlike defect correction of rounding noise, the lagged NLC potential (5a) is first-order smooth in dD. Do not conflate; test separately.

---------------------------------------------------------------------------------------------------

## 12. Suggested order of work

1. Profile `crt_gemm.py` on T4 with achieved INT8 TOPS, quantize/GEMM/reconstruct split at BPA XC (K=16384 blocks) and K shapes; decide among 2a (CS scaling), 2c (fusion, batching moduli).
2. Implement the energy-keyed ladder (#1) on top of existing `XC_PRECISION='crt'`/`K_PRECISION='crt'`, replace stage 1b/FP64 polish; gate with BPA + tamoxifen energy 1e-9 and gradient 1.2e-7.
3. In parallel (independent, helps A100/H100): CUDA-graph / fused XC (#4) and lagged VV10 (#5a) + symmetric VV10 (#5b).
4. K-build residue caching and K2 SYRK (#3).
5. Only then: occupied-projected Fock (#6), cuBLAS 13.4 probe (#7), fused 3xTF32 (#8), Laikov branch (#9).

## Source list (what was actually read)
* [V] Uchino, Ozaki, Imamura, "High-Performance and Power-Efficient Emulation of Matrix Multiplication using INT8 Matrix Engines", arXiv:2508.03984 (SC'25). https://arxiv.org/html/2508.03984v2
* [V] "Error Analysis of Matrix Multiplication Emulation Using Ozaki-II Scheme", arXiv:2602.02549. https://arxiv.org/html/2602.02549v1 (first 100k of 120k chars read)
* [V] Kawakami & Takahashi, "Improved Scaling for Fast Mode of Ozaki Scheme II", arXiv:2606.29129 (abstract). https://arxiv.org/abs/2606.29129
* [V] GEMMul8 README. https://github.com/RIKEN-RCCS/GEMMul8
* [V] Huang, Shao, Hammond, arXiv:2601.08077v3. https://arxiv.org/html/2601.08077v3
* [V] NVIDIA cuEST docs: gemm_emulation, scalar_emulation (float-float ~45-48 bits, default off on cc 8.0/9.0/10.0, for VV10 and PCM kernels; JIT first compile up to ~1 min). https://docs.nvidia.com/cuda/cuest/
* [V] Nepal & Wang, arXiv:2603.29975. https://arxiv.org/html/2603.29975v1
* [V] Laikov, "Nonlocal van der Waals density functional made faster", arXiv:2511.04169v2. https://arxiv.org/html/2511.04169v2
* [V] Lee, Lin, Head-Gordon, arXiv:1911.00470 (PDF text). [V] Hillers-Bendtsen & Martinez, arXiv:2508.19212 (PDF text).
* [V] Pederson et al., arXiv:2202.01255 (ar5iv). [V] terco PR #91 (VV10 gradient). [V] cuBLAS docs (emulation controls; page truncated before GemmGroupedBatchedEx).
* [V abstract] Ozaki 2.5, arXiv:2609.09095. [S] Uchino et al. FP8 Ozaki-II 2603.10634, complex Ozaki-II 2512.08321, PC Watch press release on CUDA 13.4, rVV10 (PRB 87, 041108), Asadchev-Valeev matrix MD, GS-DFT 2609.31483.
* Repo files read: `benchmarks/dft/NOTES_next_steps.md`, `PLAN_stage2_algorithms.md`, `RESULTS_mixed_precision.md`, `gpu4pyscf/lib/crt_gemm.py` (header, first 80 lines), `gpu4pyscf/dft/numint.py` (`_vv10nlc`, `_vmat_block_crt`), `gpu4pyscf/df/df_jk.py::get_jk`, `gpu4pyscf/scf/gdm.py` (index).
