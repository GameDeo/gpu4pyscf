# Research report 7: the report-6 items implemented and measured

Date 2026-11-05.
- **Hardware and system:** Colab T4 (session `t4r6`, stopped at the end); BPA wB97M-V/def2-TZVP, def2-universal-jkfit, grids
  (99,590)/(50,194), `crt` scheme + multi-fidelity SCF, against the tight FP64 reference.
- **Logs:** `experiments_r6/logs/`.
- **Tags:** [M] measured, [E] estimate.

## 0. Bottom line

| BPA, T4 | SCF | gradient | total | dE | max\|dg\| |
|---|---|---|---|---|---|
| start of this round (same session, `r6_noprof.log`) | 20.6 s | 6.2 s | 26.9 s | 5e-11 | 1.8e-7 |
| **new defaults, 3 runs (`r6_final.log`)** | **18.8 / 19.3 / 19.8 s** | **5.75 / 5.99 / 5.87 s** | **24.5–25.7 s** | ≤1.8e-10 | ≤1.7e-7 |

- **BPA gain:** about 1.5–2 s (6–7 %) [M].
- **Taxol on a T4:** the exact operator now runs in 0.6 GB of GPU memory instead of ~70 GB stored, at 139 s for the
  full-range exchange + J and 116 s for the long-range exchange per evaluation [M]. A full Taxol SCF still needs an
  INT8-stored surrogate for the inner loop (§4, not built).
- **Precision schedule:** neutral, left off [M].
- **Not built:** fused lagged-Γ derivative pass, VV10 low-rank far field.

## 1. Trace (nsys, `r6_nsys.txt`, `r6_nsys_per_stage.txt`) [M]

- **Busy share:** of 27.2 s profiled span, the GPU was busy for 22.9 s (84 %). The idle 16 % (4.3 s) is host/launch
  overhead; report 6 estimated ~10 %.

| kernel class | s | | per stage (busy) | s |
|---|---|---|---|---|
| CRT quantize + reconstruct | 4.70 | | DF J/K gradient | 4.65 (Rys deriv 3.17, FP64 GEMM Γ 1.07) |
| Rys integrals | 4.21 | | fine XC + fused XC gradient | 2.93 |
| FP64 GEMM | 2.99 | | fine XC, 2 outers | 2.55 (incl. 0.40 FP64 AO screening) |
| INT8 GEMM | 2.66 | | exact K, 3 outers | 1.90 |
| other (AO eval, screening, grid weights, decompress) | 2.99 | | surrogate K stage 1 | 1.58 |
| cupy elementwise | 1.45 | | fine NLC | 1.49 (vv10_sym 0.70) |
| FP32 GEMM | 1.21 | | | |
| VV10 | 0.94 | | | |
| cuTENSOR | 0.83 | | | |
| eigensolver | 0.69 | | | |
| libxc | 0.19 | | | |

Report 6's estimated shares were close:
- **FP64-bound work:** ~32 % measured vs the 35–40 % estimate.
- **Memory-bound passes:** ~26 % measured vs ~25 %.
- **INT8 tensor cores:** 10 % measured vs ~10 %.

## 2. What was built and what it measured

| item (report 6) | implementation | result [M] |
|---|---|---|
| 4.4 gradient, Γ build | `crt_gemm.ozaki1_gemm_nt`: Ozaki scheme I with balanced base-2^7 INT8 digits. One INT8 GEMM per digit group over the concatenated K axis gives exact int32 sums, so there is no CRT reconstruction. `df/grad/rhf.py::_gamma_oz1` | Γ = C(ZCᵀ) (K = nocc = 61): DF J/K gradient −0.55 s in the A/B runs, but those ran without cuTENSOR (§5), so the Γ-level figure is skewed. Gradient unchanged (1.7e-7) |
| 4.4 gradient, FP32 tier | `gvhf-rys/gen_f32.py` generates an FP32 copy of the Rys derivative kernel (`libgvhf_rys_f32`). Shell pairs are taken into the FP32 tier in ascending Γ weight up to a fraction of the total | FP32 kernel ≈3.5x faster than FP64. Extra gradient error vs fraction: ~0 at 1 %, 3e-7 at 3 %, 2.5e-6 at 10 %, 2e-5 at 30 %, 2.3e-4 all-FP32. So only ~1 % is free (−0.4 s in the skewed A/B runs) |
| default (crt `grad_stage`, every `crt` gradient, not only multi-fidelity) | `grad_gamma_ndig=6`, `grad_fp32_frac=0.01` | correct environment: gradient 6.24 s → 5.75–5.99 s, i.e. **−0.25 to −0.5 s** (with the XC-gradient changes) |
| 4.1 precision schedule | `schedule=True` with cheap/medium levels per checkpoint | cheap grid (75,302) at checkpoint 1 → **4 checkpoints**, SCF 30 s (worse). Fewer moduli only (9/10) → neutral within ±0.7 s noise. **Off by default.** Inexact checkpoints save less (~0.4 s each) than one extra checkpoint costs (~4 s) |
| 4.2 fine XC | not the single quantisation: the TN-only INT8 GEMM on Turing needs both layouts, so it saves no memory traffic. Built instead (all exact): FP32 AO shell screening (`nr_eval_gto.cu::_screen_index`, 0.05 safety margin); the 4 density components in one transposed quantisation + one INT8 GEMM with per-row exponents (`PlanesT(per_slab=True)`); Y quantised once for V·C and the 3 gradient products; derivative AO components selected by index map (no gather copies); fresh AO buffers (no copy) | fine XC per checkpoint 1.48 → 1.19–1.25 s; fine XC + gradient 3.04 → 2.90–3.03 s |
| 4.5 diagonalisation-free inner step | `inner_eig='rotate'`: occupied canonicalisation + first-order rotation into the projected reference virtuals + Löwdin. Rotation residual: 2e-1 → 8e-5 → 1e-7 → 3e-10 → 7e-13 in four calls | same cycles and accuracy on BPA (5 runs); eig 0.75 → 0.56 s. Removes the per-cycle nao³ term; Taxol estimate −15 to −40 s [E]. The default is `'auto'` (rotate only when nao > 1500), because it is tested on one molecule and the BPA gain is only 0.2 s. Added after the T4 runs, untested on GPU: a guard \|ε_o − ε_v\| ≥ 0.05 in the step. The final 3-run measurement in §0 used 'rotate' |
| bug found | the projected-gradient norm was a difference of two large traces (cancellation floor ~4e-5) | the first rotation runs stalled at that floor (445 cycles); fixed by projecting first |
| 1.4 / 4 direct checkpoints | `df/direct_k.py::DirectDFK`: exact occupied DF exchange + J in two streamed integral passes, no stored tensor. Uses the DF build's own half metric (`_decompose_j2c`); an explicit J⁻¹ amplifies rounding by the condition number of the singular erf metric (825 dropped eigenvalues on BPA) | water dimer: = stored = CPU pyscf to 1e-14 (FR and LR). BPA: dE_K 4e-11 with 15/11 moduli. End-to-end BPA (`exact_k='direct'`): dE 4.9e-10, max\|dg\| 1.7e-7 |
| gradient fit reuse in direct mode | the last direct pass already holds J⁻¹(P\|ij); rotated to the canonical orbitals it becomes the gradient's fit cache | gradient 1.7e-7, no long-range tensor build (was 3.7 s) |

## 3. Taxol on the T4 (direct exact operator) [M]

Taxol: nao 2228 (2538 sorted Cartesian), naux 6598, nocc 226. **Timing and memory only:** `r6_taxol_k.py` uses random
S-orthonormal occupied orbitals. The cost does not depend on the orbitals; accuracy was checked on the water dimer and
BPA only.

| one exact evaluation | time | peak GPU pool |
|---|---|---|
| full-range K C_occ + J | 139 s | 0.61 GB |
| long-range K C_occ | 116 s | 0.50 GB |

Breakdown of the full-range pass:

| step | time |
|---|---|
| INT8-CRT half transforms X = Cᵀ(P\|μν) | 37 + 28 s |
| FP64 T = XC and KC += XZ | 11 + 11 s |
| integrals | 14 s |
| J | 7 s (now done from the compressed pairs) |
| unpack | 4 s |

- **Inherent cost:** 2e13 operations per half-transform pass (naux·nao²·nocc). The T4 emulates FP64 at ~2.5 TFLOP/s; an
  A100's FP64 tensor cores would need ~2–3 s per pass.
- **Remaining headroom [E]:**
  - T and KC via CRT: −15 s.
  - quantise straight from the compressed pairs, no unpack and no zero pairs: −20 s.
  - Realistic: ~70–90 s per full-range evaluation.

**Change of plan vs report 6:** the inner-loop surrogate for Taxol is now INT8-stored DF tensors, not ISDF/THC.
- The ISDF fit of the AO pair space would itself cost ~1e14–1e15 operations at Taxol size.
- INT8 storage reuses the existing DF build and the increment principle.

**What still blocks a full Taxol SCF on a T4 (not built):**
1. **Inner-loop J and surrogate K need stored tensors that don't fit:** ~31 GB (SVP-RI, mixed operator) and ~31 GB
   (universal-jfit, Coulomb) in FP64.
   - Plan: INT8 storage with per-row scales, 3.9 GB each. Use them only for increments (J(δD), K_s(δD)); the increment
     principle tolerates ~4e-3 element error.
   - The INT8 B·ΔC product maps directly onto INT8 tensor cores.
2. **The stage-1 exact DF build (used for J) must be skipped.**
3. **The coarse-grid FP32 AO cache:** ~5–10 GB at Taxol size; use a smaller grid or recompute instead.
4. **Projected Taxol SCF + gradient on a T4 [E]:** 3 exact checkpoints × ~4 min (FR + LR), plus the inner loop and the
   gradient. About 15–20 min today, ~10 min with the direct-K headroom above.

## 4. Not built (and why)

- **4.3 lagged-Γ fused derivative pass:**
  - It needs a new Rys kernel that evaluates values and derivatives in one pass; the derivative kernel is separate code
    (2,200 unrolled lines).
  - At BPA the tensor is stored, so there is no value pass at the checkpoints. At Taxol scale it would save about one
    derivative pass per gradient (~1–2 min) [E].
- **4.6 VV10 low-rank far field:** VV10 is 0.93 s of kernel time per run on BPA (fine + gradient). The possible gain
  (~0.5 s) did not justify the error-control work in this round.
- **INT8 surrogate tensors for Taxol (§3):** the next step for goal 3.

## 5. Notes and caveats
- **Behaviour changes beyond the multi-fidelity path:**
  - **FP32 AO shell screening** in `libgdft` affects every DFT run. The 27 tests above passed, but
    `dft/tests/test_grids.py` (the dedicated screening regression test) and the wider `dft/tests` were **not run**. Run
    them next session; `libs75.tgz` skips the rebuild.
  - **Ozaki-I Γ build and 1 % FP32 tier** are on for every `crt`-scheme gradient.
- **Build script:** `colab/build_t4.sh` now builds `gvhf_rys_f32`. Without that library the FP32 tier switches itself
  off.
- **Runs without cuTENSOR:** `r6_sched.log`, `r6_sched2.log` and `r6_grad_tiers*.log` ran without cuTENSOR on the
  library path, so contractions used cupy (J 0.7 → 4.7 s). Their accuracy columns are valid; use only their relative
  timings. All other timings use the full environment (`/content/run.sh`).
- **Run-to-run noise** on the Colab T4 is ±0.7 s on the SCF.
- **Tests (T4):** `test_multifidelity`, `test_xc_crt`, `test_df_rks_grad`, `test_exc_contract_first` all pass (27).
- **Reuse:** compiled sm_75 libraries are kept at `/tmp/claude-0/libs75.tgz` (outside the repo) for the next session.
