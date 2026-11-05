# Research report 8: round 7: standard-threshold convergence, Taxol on one T4, quantised surrogates

Date 2026-11-05.
- **Hardware:** Colab T4 (sessions `t4r7`, `t4r7b`, high-RAM `t4hm`).
- **Systems:** BPA wB97M-V/def2-TZVP (def2-universal-jkfit), grids (99,590)/(50,194), `crt` scheme with multi-fidelity
  SCF, compared against the tight FP64 reference (`reference/bpa_tight.json`).
- **Logs:** `experiments_r7/logs/`. Session `t4r7` was reclaimed by Colab mid-round and its logs (stock baseline,
  round-start defaults, per-(l_i,l_j) tier probe, first `dft/tests` run) were lost; their numbers below are transcribed
  from the session output and marked (t).
- **Tags:** [M] measured, [E] estimate.

## 0. Bottom line

| BPA, T4 | SCF | gradient | total | dE | max\|dg\| |
|---|---|---|---|---|---|
| stock gpu4pyscf FP64 DF, pyscf defaults (conv_tol 1e-9) (t) | 196.3 s | 18.4 s | 214.6 s | 4.1e-11 | 4.2e-7 |
| round-6 defaults (t) | 19.1 s | 6.2 s | 25.3 s | 1.0e-10 | 1.7e-7 |
| **round-7 defaults, 3 runs (`r7_final.log`)** | **15.3 / 15.6 / 16.1 s** | **5.9 / 5.8 / 6.1 s** | **21.2 / 21.4 / 22.2 s** | 2.3–2.5e-10 | 4.5–4.6e-7 |

- **BPA:** 10x faster than stock gpu4pyscf on the same GPU, at the stock's own accuracy (gradient error 4.6e-7 vs
  4.2e-7; energy error below the 1e-9 convergence threshold) [M]. The 16 s target is **not** met: 21–22 s, a gap of
  5–6 s (section 6).
- **Taxol (nao 2228) on one T4:** full wB97M-V/def2-TZVP SCF + nuclear gradient completed (section 4). Standard
  gpu4pyscf cannot run it on this machine (its FP64 DF tensors alone are ~2 x 85 GB).
- **Quantised surrogate tensors:** negative result with an identified mechanism (section 3). The working large-system
  path uses FP32 surrogate tensors in pinned host memory.

## 1. Standard-threshold convergence: 2 exact evaluations on BPA [M]

The multi-fidelity default tolerance was `0.1*sqrt(conv_tol)`, ten times tighter than pyscf's own `sqrt(conv_tol)`. The
bar is now the stock SCF at pyscf defaults (row 1 above).

| variant (BPA, T4, gradient included) | outer \|g\| | total | dE | max\|dg\| |
|---|---|---|---|---|
| round-6 defaults (inner_factor 1e-2, tol 3.2e-6) (t) | 7.7e-3, 7.7e-5, 6.6e-7 | 25.3 s | 1.0e-10 | 1.7e-7 |
| inner_factor 1e-3 only (t) | 7.7e-3, 1.9e-5, 1.2e-7 | 27.1 s | 1.2e-10 | 1.6e-7 |
| inner_factor 1e-3 + tol sqrt(conv_tol) = 3.16e-5 (t) | 7.7e-3, 1.9e-5 | 21.9 s | 2.2e-10 | 4.6e-7 |

- **Surrogate-limited contraction:** about 400x per outer step (7.7e-3 to 1.9e-5). The 100x of round 6 was the
  inner-loop tolerance, not the surrogate.
- **New defaults:** `inner_factor=1e-3`; `conv_tol_grad` = `mf.conv_tol_grad` or `sqrt(mf.conv_tol)`.
- **Behaviour change:** multi-fidelity results are now converged to the same orbital-gradient threshold as a stock
  pyscf SCF, not ten times tighter. For the old behaviour, set `conv_tol_grad` explicitly.

## 2. Other BPA changes [M]

| item | result |
|---|---|
| stored exact K as K C_occ (`df_jk.get_kc_occ`): X = B C, T = C^T X, KC = X T | the second GEMM drops from nao^2 naux nocc to nao naux nocc^2; T = C^T B C is the gradient's occupied-pair fit. `grad fit_oo` 1.15 s to 0.46 s (t); with the cached metric decomposition (next row) 0.02–0.16 s |
| `_cholesky_eri` keeps `(aux_coef, aux_sorting)` on the `Int3c2eOpt` | the fit reuses the DF build's J^-1/2 instead of a second j2c integral + Cholesky/eigh |
| FP32 AO-shell screening made decision-exact | values within 0.05 of the cutoff are re-evaluated with the original FP64 expression, so the screening index is identical to upstream. This fixes `test_numint::test_sparse_index` (the only failure of the round-6 FP32 screening in `dft/tests`: 178 passed, 1 failed (t)). Now `test_numint` + `test_grids` pass, 53/53 (`r7_test_numint.log`) |
| FP32 Rys tier by (l_i, l_j) group (t) | the FP32 error is not localised: single s–s groups add up to 1.4e-5, p–s up to 9e-6, with no speed gain per group. The FP32 tier stays at 1 % of the Gamma weight |
| Gamma-weighted skipping (drop the lowest-weight shell pairs, `GRAD_TIERS['skip']`) (`r7_skip.log`) | dropping 1e-4 of the weight costs 1.7e-4 in the gradient, 1e-2 costs 1.5e-2, and only the 1 % level saves time (0.6 s). Not viable; the option is kept for diagnostics only |
| stage-1 tolerance 3e-3 (`r7_mf2.log`) | one stage-1 cycle fewer; neutral within noise |

Tests on T4 (`r7_tests.log`): `test_multifidelity`, `test_xc_crt`, `test_df_rks_grad` 22/22 pass.

## 3. Quantised surrogate tensors: why they fail, and what works [M]

**Goal.** At Taxol size the dense FP32 surrogate (SVP-RI, unpacked) would need ~73 GB and the stored exact tensors
~85 GB each. The plan was INT8 per-pair-scaled DF tensors (`df/int8_df.py`): ~4–5 GB each, INT8 GEMMs.

**Operator accuracy is fine in isolation** (`r7_int8_ops.log`, `r7_dbg6b.log`):

| tensor | rel. error K_s / J_s (BPA, physical density) | increment error |
|---|---|---|
| K, INT8, SVP-RI, wB97M-V operator | 2.6e-3 | 3.5e-3 |
| J, INT8, universal-jfit | 1.5e-2 | 8e-3 (random dC), 8.5e-2 (D - D_guess) |
| J, INT16 | 4.5e-5 | 5e-5 / 2.2e-4 |

J needs more bits than K: the fitted coefficients rho_L = B D of a physical density are large and coherent, while
J_p = sum_L B_Lp rho_L cancels. A symmetric J^-1/2 fit instead of the Cholesky one does not help (2x worse).

**In the SCF every quantised variant diverged** (stage-1 |g| 4e5 with INT8 K; inner loop diverging with INT16 J).

**Mechanism.** The overlap matrix of def2-TZVP is nearly singular:
- lambda_min(S) is 7.2e-6 on BPA (33 eigenvalues below 1e-3) and 3.4e-6 on Taxol (118 below 1e-3).
- The rounding noise of a quantised AO-pair tensor is unstructured in the AO basis. The orthogonalisation
  X = S^-1/2 amplifies it by up to 1/lambda_min along near-linearly-dependent AO combinations. Exact (and
  FP32-rounded) operators are physically small there; quantised ones are not.
- Evidence (`r7_dbg8`, `r7_dbg12`): after one stage-1 cycle with INT16 J, the density is orthonormal and has the right
  122 electrons, but its elements reach 3.1e4. On that density the INT16 J error is 0.84 (vs 4.6e-5 on the guess
  density).
- A second, independent bug found on the way: a CRT build of the Coulomb tensor with 8 moduli leaves a systematic
  error that the same cancellation amplifies ~20x. The tensors now build with >= 11 moduli.

**What was tried** (`r7_A*.log`, BPA, outer |g| per checkpoint):

| surrogate (exact checkpoints unchanged) | outer \|g\| | contraction |
|---|---|---|
| dense FP32 K + exact J (default) | 7.7e-3, 1.9e-5 | ~400x |
| INT8 K + INT16 J increments, no filter | 7.7e-3, 22, 11, ... | diverges |
| same + spectral filter lambda > 1e-3 (`inc_filter`) | 7.7e-3, 2.5e-3, 3.7e-4, 5.1e-5, 9.6e-6 | ~7x |
| INT16 J increments only (dense K), filter 1e-3 | 7.7e-3, 7.8e-5, 9.6e-7 | ~100x |
| INT8 K increments only, filter 1e-3 | 7.7e-3, 2.6e-3, 3.7e-4, 5.2e-5, 1.0e-5 | ~7x |
| INT16 K, FP32 compute, filter 1e-3 | 7.7e-3, 2.3e-4, 2.2e-5 | ~30x |
| FP32-stored K, no filter | 7.7e-3, 1.9e-5 | ~400x (= dense) |
| FP32-stored K with filter 1e-3 | 7.7e-3, 2.3e-4, 2.2e-5 | ~30x: the filter itself costs rate |
| **FP32 J + FP32 K (`inc_store='f32'`), host-streamed or on device** | 1.0e-2, 9.5e-5, 1.2e-6 | **~100x, 3 checkpoints** |
| ACE reference K (`ref_k='ace'`, rank-nocc from the exact K C_occ) | same as the dense reference | works |

- **Converged answer:** every converging variant reaches the same answer (dE <= 5e-10, max|dg| <= 4e-7). The
  surrogate only sets the rate.
- **Rule:** anything that enters the inner Fock matrix must carry FP32-level, structured rounding. Quantised tensors
  are usable only for increments behind a spectral filter, and at a large cost in contraction rate.
- **Plan change:** the INT8 surrogate plan of reports 6 and 7 is replaced by FP32 surrogate tensors in pinned host
  memory, streamed per evaluation (`surrogate_host`).

**Remaining gap between FP32 J_s and exact J** (100x vs 400x). The surrogate J uses universal-jfit; the exact J uses
universal-jkfit. With exact J at the checkpoint as the reference, the rate is the same (`r7_A5.log`), so the
increment response of the smaller fit basis is what limits it. At Taxol size a universal-jkfit FP32 J would be
~30 GB of host memory.

## 4. Taxol (C47H51NO14, nao 2228, 1.33M significant AO pairs) on one T4 [M]

Configuration:
- `inc_store='f32'`: J_s with universal-jfit, 19.1 GB; K_s with SVP-RI, 19.6 GB; both pinned host memory in 2 GB
  chunks, built in ~50 s each.
- `exact_k='direct'`: exact K and J at the checkpoints in two streamed integral passes; INT8-CRT half transforms.
- `coarse_cache_frac=0.1`; a high-RAM T4 VM (50 GB host memory).

The two failed attempts:
- A single 17 GB pinned allocation fails on this VM; 4 GB chunks work up to 48 GB.
- The first checkpoint ran out of GPU memory in the direct-K metric step (3 x 2.7 GB naux x nocc^2 arrays). It is now
  applied in column blocks in place. The occupied-pair fit (2.7 GB per operator) is kept only at the checkpoint
  predicted to be last.

TAXOL_RESULTS

## 5. Not done this round
- **A100 (goal 4):** no run this round. The new library code (decision-exact screening, FP32 Rys tier) needs an sm_80
  rebuild (~30 min); the FP32 host-streamed surrogate is T4-memory-specific; the A100's 40/80 GB holds the surrogate
  tensors on the device.
- **Direct-K efficiency:** quantise straight from compressed pairs; T and KC via CRT. These are the next ~2x on the
  Taxol checkpoint (report 7 section 3).
- **Fused lagged-Gamma derivative pass and VV10 low-rank far field:** as in report 7.

## 6. The 16 s BPA target: where the remaining time is (round-7 defaults, `r7_final.log`)

| stage | s |
|---|---|
| fine XC + XC gradient (last checkpoint) | 2.9 |
| K exact (2 checkpoints, incl. the LR tensor build) | 2.2 |
| K_s stage 1 (incl. the surrogate tensor build) | 2.0 |
| DF build (exact FR tensor) | 1.5 |
| fine XC (first checkpoint) | 1.3 |
| fine VV10 (2) | 1.1 |
| coarse XC (stage 1 + inner) | 1.5 |
| eig, J, DIIS, setup, misc | 2.5 |
| **SCF** | **~15.5** |
| nuclear gradient (DF J/K derivative integrals 4.9 s, of which Rys 3.2) | 5.9 |

Getting to 16 s needs about 5.5 s more:
- **Gradient (~6 s):** dominated by FP64 Rys derivative integrals; FP32 does not work beyond 1 % of the Gamma weight
  (section 2). The remaining options are kernel-level: fuse the FR and LR passes (shared pair data), or a fused
  value + derivative pass.
- **Stage 1 + the two tensor builds (~5.5 s):** a single build that serves both the exact FR tensor and the surrogate
  would save ~1.5 s [E].
- **GPU idle:** the round-6 trace measured 16 % idle (host/launch). CUDA graphs for the inner loop are the remaining
  lever.

## 7. Behaviour changes
- **Multi-fidelity tolerance:** `conv_tol_grad` now defaults to pyscf's `sqrt(conv_tol)`, and `inner_factor` to 1e-3
  (section 1).
- **Stored exact K:** computed as K C_occ, with the fit reused for the gradient (`occ_k=True`).
- **`exact_k`:** defaults to `'auto'` (direct when the stored FP64 tensors would exceed 45 % of free GPU memory).
- **Surrogate defaults:** quantised and FP32-stored surrogates are opt-in (`inc_store`). The default remains the dense
  FP32 surrogate + exact J.
- **`libgdft` screening:** FP32 with an FP64 re-check at the cutoff; the index is identical to upstream.
