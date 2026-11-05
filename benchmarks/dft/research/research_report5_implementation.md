# Research report 5: multi-fidelity SCF implemented, measured; gradient analysis

Date 2026-11-05. Implements the architecture of `research_report4_novel_algorithms.md`. All numbers are measured on a Colab
**T4** (session `t4mf`) unless marked [G] (projection).
- System: Bisphenol A, wB97M-V/def2-TZVP, def2-universal-jkfit, grids (99,590)/(50,194).
- Reference: tight FP64 energy and gradient in `reference/bpa_tight.json`.
- Baseline: the current production `crt` scheme measured in the same session, with the same NLC-gradient precision fix
  applied to both (see 3.3).

---------------------------------------------------------------------------------------------------

## 0. Result

Final defaults, including the SCF-tensor fitting coefficients for the gradient (3.4, built). Both T4 columns use FP32 VV10
in the gradient.

| BPA | T4: current `crt` | **T4: multi-fidelity** | A100: current FP64 | **A100: multi-fidelity** |
|---|---|---|---|---|
| SCF | 37.4 s | **20.0 s** (1.87x) | 6.51 s | **4.53 s** (1.44x) |
| gradient | 13.7 s | **6.1 s** (2.25x) | 2.00 s | **0.88 s** (2.27x) |
| total | 51.1 s | **26.1 s** (1.96x) | 8.51 s | **5.41 s** (1.57x) |
| dE vs tight FP64 ref | 1.1e-10 Eh | 5.5e-11 Eh | −5.3e-10 | −6.1e-10 |
| max gradient error | 1.1e-7 Eh/bohr | 1.7e-7 Eh/bohr | 3.3e-7 | 1.7e-7 |

- **Counts:** 3 exact (outer) evaluations, 9 inner and 8 stage-1 cycles on both GPUs. The current T4 scheme uses 17 full
  Fock builds; the A100 FP64 run uses 12.
- **Outer residuals:** 7.7e-3 → 7.7e-5 → 7e-7.
- **History:** the original FP64 code took 249.7 + 47.1 = 296.8 s on this T4; now 26.1 s (11.4x).
- **Gap:** the unoptimised A100 (10.95 + 2.5 s, handoff) is still 1.9x faster than the T4. The A100 itself is now 1.57x
  faster than its previous best plain-FP64 pipeline (A100 multi-fidelity numbers are from the second run in the session;
  the first included JIT of the new kernels: 7.2 + 1.2 s).
- **A100 headroom:** the FP64 path of `nr_rks_vco` is untuned (0.32 s per fine evaluation vs 0.196 s for the production
  fused full-matrix loop), and the fused gradient adds 0.6 s there.

---------------------------------------------------------------------------------------------------

## 1. What was built

### `gpu4pyscf/scf/multifidelity.py` (new)
Enable with `mf.multifidelity = True` (or a dict of options), or `bench_wb97mv.py --multifidelity '{}'`.

- **Driver: stage 1** (cheap operator only) → outer loop.
  - Each outer step applies the exact operator to the occupied orbitals only: W = F(D_ref) C_occ.
  - The defect is folded into Δ̃ = W CᵀS + S C Wᵀ − S C(CᵀW)CᵀS.
  - Inner DIIS runs on the surrogate F_s = h + J + V_xc^coarse − ½K_s + Δ̃.
  - The occupied orbitals are finally canonicalised with the exact operator, so `mo_energy[:nocc]` is valid for the gradient.
- **`SurrogateK`:** one DF tensor of the mixed operator α·erf/r + hyb·erfc/r in def2-SVP-RI. It is gpu4pyscf's own
  `lr_factor`/`sr_factor` DF and is built with 8 INT8 moduli: the tensor is fixed, so its precision does not affect exactness.
  - Stage 1 uses FP32 absolute K.
  - Inner steps use K_s(D_ref) (accurate) + FP32 increments 2Σ_L[Y Xᵀ + X Yᵀ + Y Yᵀ], where Y = B ΔC and ΔC is the
    Procrustes-aligned orbital increment.
- **`CoarseXC`:** semilocal XC on the (30,110) inner grid from cached FP32 AO values (0.5 GB).
  - Stage 1: FP32 absolute.
  - Inner steps: ρ = ρ_ref + δρ, with δψ = AᵀΔC in FP32 and ψ_ref in FP64.
  - libxc runs pointwise in FP64; δV = Φᵀ diag(u − u_ref) Φ in FP32.
- **`FastEig`:** FP32 syevd + Ogita–Aishima refinement with cluster threshold and Cholesky-QR. Correct, but **slower**
  than cuSOLVER FP64 at nao ≈ 600 on a T4 (63 vs 46 ms), so the default stays FP64.
- **`gemm16_nt`:** FP16 tensor-core GEMM with FP32 output (cublasGemmEx).

### `gpu4pyscf/dft/numint.py: nr_rks_vco` (new)
E_xc and V_xc·C_occ without any nao×nao matrix: V C = Σ_k A_k Y_kᵀ.
- Y₀ = u₀ψ₀ + Σ_x u_x ψ_x and Y_x = u_x ψ₀ + ½u₄ψ_x, with ψ_k = A_kᵀC built in one RawKernel.
- One grouped block pass: AO values of ~0.8 GB of blocks are kept while libxc runs once per group, so each AO block is
  evaluated once.
- `with_grad=True` also returns the XC nuclear gradient from the same pass (section 3.1).

### Hooks
- `scf/hf.py` selects the driver.
- `df/grad/rks.py` uses the fused XC gradient when the density matches the last outer evaluation.
- `scf/mixed_precision.grad_stage` now also sets the final VV10 precision for the NLC gradient, plus an optional INT8 switch
  for the DF J/K-gradient contractions (off, see 4).

### Correctness checks (`experiments_r4/test_mf_components.py`, water dimer TZVP, T4)
- `nr_rks_vco` vs `nr_rks @ C`: 1.5e-15 (FP64), 3.4e-11 (INT8).
- Fused XC gradient vs `get_exc`: 1.9e-15 (FP64), 2.8e-8 (INT8).
- Multi-fidelity energy vs plain tight SCF: 6e-14 (FP64), 3e-11 (INT8).
- Gradient: 9e-9 / 6e-8.

On BPA every run above passes the energy (≤1e-8) and gradient (≤1e-6) gates.

---------------------------------------------------------------------------------------------------

## 2. Where the time goes now (BPA, T4, final defaults)

| SCF part | s | | gradient part | s |
|---|---|---|---|---|
| fine XC, 2 outers (V·C_occ) | 3.20 | | DF J/K gradient (FR+J, LR) | 7.35 |
| fine XC + fused XC gradient, last outer | 3.82 | | NLC gradient (VV10 FP32) | 0.95 |
| exact K at outers (incl. LR tensor build) | 2.69 | | h1e / overlap / W / nuc | 0.47 |
| surrogate K, stage 1 (incl. surrogate build) | 1.96 | | XC gradient | 0 (fused) |
| fine NLC, 3 outers | 1.73 | | | |
| exact DF build | 1.24 | | | |
| coarse XC (inc 0.73, FP32 abs 0.66, accurate at refs 0.59, set_ref 0.27) | 2.25 | | | |
| eig 0.76, J 0.71, diis 0.31, nlc coarse 0.34, K_s inc 0.37, K_s ref 0.45 | 2.94 | | | |
| setup (grids, AO cache, h/S) 0.61, init guess 0.06 | 0.67 | | | |

About 10.7 s of the SCF is "exact" work (3 fine XC + NLC + K evaluations and both exact DF builds); about 8 s is the
17 cheap cycles plus setup.

---------------------------------------------------------------------------------------------------

## 3. Gradients

### 3.1 New: the XC gradient is the derivative of the occupied-only Fock product (implemented, exact)
For D = C Cᵀ (occupation-weighted C) the contract-first XC gradient (`grad/rks.py::_exc1_block`) is, per AO μ and direction n:

  exc1[μ, n] = Σ_i C_μi G_n[μ, i],   G_n = Σ_k (∂_n A_k) Y_kᵀ

with exactly the same Y_k (and the same INT8-quantised Y operand) as V·C = Σ_k A_k Y_kᵀ. Here ∂_n A_0 = ∂_nφ and ∂_n A_x = ∂_n∂_xφ.
Derivation:
- the term (D·aow) collapses to C (Cᵀ aow) = C·(Σ_k wv_k ψ_k);
- the term (D φ) collapses to C ψ₀;
- the τ term collapses to C (wv₄ ψ_x).

Adding them gives Y₀ for ∂_nφ and Y_x for ∂_n∂_xφ.

So the gradient needs no extra libxc call, no extra ρ, and no second AO pass. It rides on the final exact evaluation, which
by construction is at the converged density.
- **Measured cost:** +2.2 s inside the last outer step (deriv-2 AOs, three extra GEMMs) against 5.1 s for the standalone
  XC gradient.
- **Accuracy:** 2.8e-8 (INT8) vs FP64.

### 3.2 NLC gradient
`_get_denlc` ran outside the precision context, so the FP64 VV10 pair kernel was used (5.7 s on T4). With the scheme's FP32
VV10 (the SCF already uses it; the gradient error is unaffected, 1.2e-7) it takes 0.95 s. This fix also speeds up the
existing `crt` scheme (its gradient 18.3 → 13.7 s, included in the baseline above).

### 3.3 DF J/K gradient: the remaining 7.35 s
Per operator (full range + J; long range):
1. Recompute the raw 3c integrals and form j3c_oo = (P|ij): FP64, a 90 GFLOP contraction.
2. Apply J⁻¹.
3. Build the pseudo-density Γ^P_μν = (C Z^P Cᵀ)_μν: FP64, a 90 GFLOP contraction with reduction length nocc = 61.
4. Run the FP64 Rys derivative kernel (`sum_ejk_int3c2e_ip1`).

All four steps are FP64-bound on a GPU with 1/32-rate FP64.

**Measured negative:** INT8 for the two contractions is slower (8.6 vs 7.3 s) and inaccurate at 11 moduli (5.2e-6 Eh/bohr).
- In step 1, J⁻¹ amplifies the rounding error; this is the same effect that forced 15 moduli in the DF build.
- In step 3, the reduction length is only 61, too short for INT8 to pay.
- Kept as `mixed_precision` option `jk_grad_crt`, off by default.

### 3.4 Gradient ideas not built yet (exact unless stated; T4 estimates [G])

| idea | mechanism | est. saving |
|---|---|---|
| **built:** fitting coefficients from the SCF tensor | R = J⁻¹(P\|ij) = aux_coef·(CᵀBC) from the stored DF tensors at the end of the SCF (`df/grad/rhf.py::precompute_fit_oo`, used by `_jk_energy_per_atom` when the operator and orbitals match). Removes the raw 3c recompute, the 90-GFLOP contraction and the J⁻¹ solve. Gradient vs full recomputation: 2e-9 | **measured:** gradient 8.7 → 6.1 s, +1.0 s at the end of the SCF |
| Γ-weighted Rys screening | skip (shell pair, aux shell) blocks with max\|Γ\|·Schwarz bound < 1e-12: Γ is known before the kernel runs | small at 33 atoms, large at 100+ |
| MO-transformed derivative driver | contract C inside the Rys kernel (half-transformed derivative integrals) instead of materialising Γ in AO-pair space: removes the 90 GFLOP short-reduction FP64 GEMM and the pair compression | ~1.5 s (kernel work) |
| double-float Rys | Rys roots/weights and recurrences in FP32 pairs (~48 bits) on non-FP64 GPUs | ~2x of the Rys part |
| lagged-Γ fused integral pass + FP32 increment (only when the exact K is integral-direct, i.e. large systems where B is not stored) | the last outer's integral pass computes values and derivatives together, contracted with Γ of the previous outer; the correction g(Γ_final − Γ_prev) is linear and small, so it runs in FP32 | removes the separate FP64 derivative pass |
| `mix_inside_kernel` model (opt-in, dE ~1e-6) | one operator instead of two: halves the DF gradient, the exact K at outers and the DF memory | ~3.6 s gradient + ~1.5 s SCF |
| geometry optimisation: carry the fidelity defect and the gradient defect between steps (manifold mapping) | only every few steps needs the full exact gradient | per-step amortisation |

Summary: the XC part of the gradient is solved by the fused identity. The DF part is FP64 integral and contraction work; on a
T4 it needs either kernel work (double-float / MO-driver / screening) or the model change.

---------------------------------------------------------------------------------------------------

## 4. Negative and neutral results this round (measured)

| what | result |
|---|---|
| FP16 exchange increments on BPA (they worked on the water dimer) | **diverged**. FP16 noise ≈1e-3·\|K(δD)\|, about 0.017\|g\| at the inner tolerance, above the required 0.01\|g\|. The inner loop stalls and DIIS wanders. The dimer hid it because its K is small. FP32 increments: identical to exact. |
| FP32 syevd + Ogita–Aishima (2 steps) | correct after adding the cluster threshold (without it the virtual vectors became garbage and the convergence measure was wrong), but 63 ms vs 46 ms for FP64 syevd at nao 623 |
| single-pass grouped `nr_rks_vco` | 1.9 → 1.6 s per fine evaluation (one AO pass); the two INT8 quantisations of the AO block (rho: transposed; V·C: row-scaled) remain the cost |
| occupied-only fine XC vs full vmat | 2.12 → 1.6 s per evaluation (not the 3.2x of the FLOP count: AO quantisation, not GEMM, dominates) |
| INT8 for the DF J/K-gradient contractions | slower and inaccurate (3.3) |
| inner grid (20,50) / (30,110) / (40,146) | 4 / 3 / 3 outer evaluations; (30,110) is fastest (21.5 vs 24.0 vs 25.0 s) |
| coarse XC as FP32 increments | 0.19 → 0.08 s per inner cycle (kept) |

---------------------------------------------------------------------------------------------------

## 5. Are we at the limit?

**Not at the algorithmic limit, but at the T4's FP64 wall for the parts that remain.**
- **Exact evaluations (~10.7 s SCF):** 3 fine XC/NLC/K evaluations plus the two exact DF builds. A fourth evaluation would
  already cost more than the whole inner loop. Ideas for fewer or cheaper exact evaluations:
  1. **Exact evaluations on increments [G]:** for outers k ≥ 1, V(D_k)C_k = V(D_{k−1})C_{k−1} + Σ A δY. δY is formed exactly
     pointwise from the stored u, ψ and the aligned ΔC. Only the small δψ and δY GEMMs are needed, and they can use fewer
     INT8 moduli (7 instead of 11; FP16 is too coarse because E_xc needs ρ to ~1e-10). K(D_k)C_k likewise from the stored
     X = B C_{k−1}. Estimated −1.5 to −2 s, at the cost of 2.2 GB of stored ψ.
  2. **Anderson/secant acceleration of the outer map [G]:** the outer contraction is ~100x per step; a secant update of the
     defect (manifold mapping) could reach the tolerance in 2 outer steps for some systems.
  3. **Energy-only single points [M on CPU, report 3]:** 2 outer evaluations suffice (E error ~ g²).
- **Cheap cycles (~8 s):** each costs J 0.034 + K_s 0.04 + coarse XC 0.08 + eig 0.045 + DIIS 0.02 + Python glue. The next
  step is one CUDA graph per inner cycle (all shapes are fixed), which would mainly help A100/H100.
- **Gradient (8.7 s):** see 3.4.

**Parity of the T4 with the unoptimised A100 (13.5 s total) needs a further ~2x.** On the SCF side, items 1–2 above might
give ~4 s [G]. On the gradient side, ~3 s [G] needs kernel work (MO-transformed derivative driver, double-float Rys). The
remainder is FP64-bound integral evaluation that INT8 cannot express. Being honest: with the present tools the T4 lands at
~1.5–2x of the unoptimised A100 for SCF + gradient. For energy-only single points (2 outers) it is closer [G].

---------------------------------------------------------------------------------------------------

## 6. A100 (sm_80 libraries built on the T4 host; session stopped after the run)

| component (A100, multi-fidelity, 2nd run) | s |
|---|---|
| fine XC + fused gradient (last outer) | 0.92 |
| fine XC, 2 outers | 0.64 |
| surrogate K, stage 1 (incl. build) | 0.49 |
| exact K, 3 outers | 0.33 |
| exact DF build | 0.30 |
| fine NLC, 3 outers | 0.30 |
| coarse XC (inc + FP32 abs + refs) | 0.53 |
| J 0.15, eig 0.15, setup 0.15, grad fit_oo 0.18, rest | ~0.7 |

The A100 is now glue- and FP64-path-bound. The next steps there:
- a fused FP64 tensor-core V·C kernel, which should reach the 0.196 s of the full-matrix loop or less;
- a CUDA graph for the fixed-shape inner cycle.

## 7. Files
- `gpu4pyscf/scf/multifidelity.py`, `gpu4pyscf/dft/numint.py` (`nr_rks_vco`, `_half_transform`, `_BUILD_Y_SRC`),
  `gpu4pyscf/scf/hf.py`, `gpu4pyscf/scf/mixed_precision.py`, `gpu4pyscf/df/grad/rks.py`, `gpu4pyscf/df/grad/rhf.py`.
- Experiments: `experiments_r4/test_mf_components.py`, `r5_variants.py`, `r5_grad_profile.py`; benchmark
  `bench_wb97mv.py --multifidelity`.
