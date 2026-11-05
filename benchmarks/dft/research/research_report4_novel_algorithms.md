# Research report 4: reformulating the heavy math (multi-fidelity, increment-precision SCF)

Date 2026-11-05. Builds on reports 1-3 and `HANDOFF.md`.
Labels: **[M]** measured this round (GPU = T4 session `t4r3`, BPA wB97M-V/def2-TZVP/DF, crt scheme; CPU = PySCF 2.14 emulation of the
algorithm, counts and accuracy only, no hardware timing); **[G]** projection or derivation, not built. Scripts, logs, JSON:
`research/experiments_r4/`.

---------------------------------------------------------------------------------------------------

## 0. Bottom line

The heavy math is not cheap to do *per evaluation* on a T4. The large cut is to stop doing exact evaluations except where
they matter. Each component below has a measured check:

1. **Fidelity-ladder defect correction** (nonlinear multigrid / FAS idea, round 3) is generalised from "coarse grid" to *every*
   expensive component. The inner SCF runs on a cheap surrogate Fock matrix; the exact operator is applied only at 3-4 outer steps.
   The fixed point is exact. **[M, CPU]** final energies equal a tight plain SCF to 1e-12 Eh (water dimer).
2. **The exact operator is needed only as F·C_occ, never as an nao×nao matrix.** The fidelity defect is compressed ACE-style onto the
   occupied space: Δ̃ = W CᵀS + S C Wᵀ − S C (CᵀW) CᵀS. Outer contraction is unchanged **[M, CPU]**. FLOP cut in the
   fine XC GEMMs: 3.2x **[M, GPU, op count]**, but only 1.26x in an unfused prototype **[M]** (AO quantization dominates; see 2.2).
3. **Inner exchange surrogate = one merged FR+LR DF tensor** (0.15 I + 0.85 G folded into a single tensor; 1.7x fewer rows), **evaluated
   on increments in FP16**: K_s(D) = K(D_ref) + K_fp16(D − D_ref). **[M, CPU]** Identical outer convergence to exact K. The *same* FP16
   arithmetic used directly stalls at |g| ≈ 6e-4. **[M, GPU]** FP16 tensor cores evaluate a full-size BPA exchange increment in 17 ms;
   the current INT8-CRT K_FR takes 0.30 s and FP64 0.85 s.
4. **Memory.** Error-bounded integer storage of the DF tensor needs **11.6 bits/element at an absolute step of 1e-10 (5.5x smaller
   than FP64), dE_K = 3e-10 Eh** **[M, GPU]**. The exact tensor is only touched 3-4 times (as K·C_occ), so it can also be streamed
   or recomputed instead of stored. Projected T4 capacity: ~42 atoms TZVP today → ~75 (compressed) → ~110 (integral-direct exact
   part) **[G]**.
5. **Projected T4 BPA time:** SCF 36.6 s **[M, this run]** → ~21 s with only the algorithm (current kernels) → ~14 s with two kernel
   changes → ~10 s with fixed-point AO evaluation **[G]**. With the gradient ideas, total ~15-19 s vs the unoptimised A100's 13.5 s:
   **parity is within reach but not shown.** A100 gain is modest, ~1.6-2.5x **[G]**, because it becomes glue/launch-bound.
6. **The gradient is the open research problem.** After the SCF work it is ~1/3 of the T4 total; the only levers found are reuse,
   scheduling and known precision tricks (section 5).

---------------------------------------------------------------------------------------------------

## 1. Where the time and the FLOPs are (T4, BPA; `r4_gpu.json`) [M]

Unit costs at the converged density, final-stage precision (11 INT8 moduli):

| unit | time | note |
|---|---|---|
| fine XC `nr_rks` (99,590), 1.06 M points | 2.12 s | rho 0.82 s + vmat 0.86 s + AO/libxc/glue |
| coarse XC (50,194) / (20,50) | 0.46 / 0.10 s | 217 k / 41 k points |
| NLC (50,194) / (30,110) / (10,26) | 0.50 / 0.14 / 0.019 s | VV10 FP32 kernel |
| K full-range / long-range / J | 0.30 / 0.15 / 0.03 s | INT8-CRT K |
| `mf.eig` (generalized eigh, 623) | 0.10 s | FP64 syevd alone: 0.048 s |
| SCF (crt scheme, 17 cycles) | 36.6 s | E = −731.64940451 |

XC work per grid level, MGGA (`R1`): Σ nao_sub² ng and the occupied-only alternative 2 Σ nao_sub ng nocc:

| grid | points | mean nao_sub | vmat TFLOP | V·C_occ TFLOP |
|---|---|---|---|---|
| (99,590) | 1,063,936 | 373 | 1.24 | 0.39 |
| (50,194) | 217,344 | 395 | 0.29 | 0.085 |
| (20,50) | 40,704 | 463 | 0.073 | 0.019 |

In the current scheme the SCF is ~9 fine evaluations × 3 s + ~8 coarse × 1 s. Exchange is ~0.45 s on *every* cycle.

---------------------------------------------------------------------------------------------------

## 2. The architecture: multi-fidelity, increment-precision SCF

### 2.1 Fidelity ladder (FAS-style defect correction across components)
Exact Fock F(D) = h + J(D) + V_xc^fine(D) + V_nlc^fine(D) − ½K(D). Surrogate F_s(D) = h + J(D) + V_xc^coarse(D) − ½K_s(D) + Δ̃,
with Δ̃ frozen per outer step and F_s(D_ref) ≡ F(D_ref) on the occupied space. Iterate F_s to a loose inner tolerance (1e-2 |g_true|),
then evaluate the exact residual at the new D_ref and rebuild Δ̃.

Each fidelity axis contributes to the surrogate's Jacobian error and so to the contraction per outer, never to the fixed point:
- grid: coarse (20,50) semilocal XC;
- NLC: frozen;
- exchange: merged and/or truncated aux tensor;
- precision: FP16 on increments.

This is Brandt's full approximation scheme (FAS) with a "coarse level" made of cheaper operators rather than a coarser
discretisation of the same operator. Space mapping and manifold mapping from engineering optimisation are the same structure.

### 2.2 Occupied-projected defect (exact operator only as F·C_occ)
At the fixed point only F·C_o matters (stationarity: (1 − S D/2) F C_o = 0). Given W = (F − F_cheap)(D_ref) C_o, with C_o S-orthonormal:

  Δ̃ = W CᵀS + S C Wᵀ − S C (CᵀW) CᵀS

- Δ̃ is symmetric and has rank ≤ 2 nocc.
- Δ̃ C = W exactly, because CᵀW = CᵀΔC is symmetric.
- So F_s(D_ref) C = F(D_ref) C and the fixed point is unchanged.

Δ̃ differs from the full Δ only in the virtual–virtual block, which enters the orbital Hessian at order |Δ_vv|/gap. The XC defect
is small (|ΔV| ≈ 2e-3 vs gap ≈ 0.3 Eh, round 3), so this costs ~1e-2 in contraction.
The residual is also computed from W alone: ‖F D S − S D F‖ = ‖2(W CᵀS − S C Wᵀ)‖.

**[M, CPU]** Water dimer, def2-TZVP, inner seed = surrogate (the first run had a bug, see 4):

| inner exchange | outer residual history | final E − E_ref (tight plain SCF) |
|---|---|---|
| exact DF K | 1.1e-2, 6.0e-5, 5.5e-7, 4.2e-9 | 0 (1e-12) |
| merged FR+LR tensor (226 rows vs 381) | 1.1e-2, 6.0e-5, 5.8e-7, 4.4e-9 | 1e-12 |
| exact K, FP16 increments | 1.1e-2, 5.9e-5, 5.5e-7, 4.3e-9 | 1e-12 |
| merged def2-SVP-RI tensor (152 rows, 40 %), FP16 increments | 1.2e-2, 1.0e-4, 1.0e-6, 9.3e-9 | 1e-12 |
| merged lmax ≤ 1 aux (100 rows, 26 %), FP16 increments | 9.4e-2, 1.1e-3, 1.0e-5, 9.4e-8 | 1e-12 |
| merged s-only aux (28 rows, 7 %), FP16 increments | 3.4e-1, 6.9e-3, 1.6e-4, 5.0e-6, 2.0e-7, 8.7e-9 | 1e-12 |

Every run: 8 stage-1 cycles plus 4-6 inner cycles per outer. E_ref = −152.866927614179 (plain DF SCF on the same grids,
conv_tol 1e-11). The larger first-outer residual of the truncated runs comes from stage 1 converging on the cruder surrogate.
The contraction per outer stays ~100x down to 26 % of the rows.

Cost: the fine XC needs only V·C_o = Σ_k φ_k Y_kᵀ, with Y built from the half-transformed orbitals that rho already computes
(Y₀ = v₀ψ + v₁·∇ψ, Y_x = v₁ₓψ + ½ v_τ ∂ₓψ).
- FLOPs 1.24 → 0.39 TFLOP per fine evaluation **[M]**.
- Unfused cupy prototype on T4 **[M]**: 0.68 s (GEMM 0.39 + Y build 0.29) vs 0.86 s for the production vmat, relative error
  5.6e-10. Only 1.26x: the INT8 path is bound by quantising the AO operand (~3 ms per 4096-point block vs ~0.4 ms of memory
  traffic), and rho and vmat each quantise the AO values separately.
- The win needs one AO quantisation per block shared by both GEMMs. Rho reduces over AOs and V·C_o over grid points, so the same
  row-scaled AO planes serve both if the AO row exponents are folded into C (c′_μi = 2^{e_μ} c_μi). That needs a non-TN INT8
  layout (cuBLASLt) or a transposed copy.
- Projected fine XC: 2.12 → ~1.2 s **[G]**.
- Exact exchange at outers: K·C_o = Σ_L X^L (X^Lᵀ C_o) needs only the K1 half: ~0.23 s for FR+LR **[G]**.

### 2.3 Exchange surrogate
Merged tensor: M = hyb·I + (α − hyb)·G with G = L⁻¹ J_erf L⁻ᵀ is positive definite, so B′ = M^{1/2} B gives
K_s = Σ_L B′^L D B′^L: one tensor instead of two, 1.7x fewer rows (381 → 226 on the dimer). This is the cuEST single-tensor
approximation (rel. K error 6e-5); here its error is removed by the defect.

Aux truncation **[M, CPU, fixed runs, table 2.2]**:
- def2-SVP-RI, merged: 40 % of the exact rows, no loss in outer count.
- lmax ≤ 1, merged: 26 %, ~100x contraction per outer.
- s-only: 7 %, 6 outers instead of 4.
- Inner exchange FLOPs and surrogate memory scale with the rows: 2.5-4x on top of FP16.
*Correction:* cc-pVDZ-JKFIT (232 functions) is **not** smaller than def2-universal-jkfit (226), so the first "cheap aux" run
proved nothing about cost.

**THC/ISDF exchange surrogate: negative [M, CPU].** At c = 2/4/8 interpolation points per AO the relative K error is 0.30/0.044/0.0024
(in line with the literature's c ≈ 8.5 for < 1 kcal/mol). The outer contraction is only ~2.5-7x (9-15 outers). THC errors are spread
over the whole response, whereas aux-truncation errors sit in directions that matter little for the valence response. What counts
for a surrogate is the accuracy of the response, not of the energy.

### 2.4 Mixed-precision iterative refinement inside the SCF
K is linear in D, so K_s(D) = K_s(D_ref) + K_s(δD) with δD = D − D_ref. Rounding errors of a low-precision evaluation scale with
|δD| (small) instead of |D|.
- **[M, CPU]** FP16 operands with FP32 accumulation give convergence identical to exact.
- The absolute form K_fp16(D) stalls at |g| ≈ 6e-4 for all 15 outers (`dimer_aux.log`).
- Same principle as LU-based iterative refinement in numerical linear algebra (low-precision correction, high-precision residual),
  here applied to the Fock-build increment.

**[M, GPU]** T4 FP16 tensor cores, BPA-size exchange increment (naux 1904, nao 623):

| precision | rank-64 factor | rank-128 factor | TFLOP/s |
|---|---|---|---|
| FP16 | 17 ms | 38 ms | 10-11 |
| FP32 | 66 ms | 120 ms | 3 |
| FP64 | 0.85 s | 1.7 s | 0.22 |

Not measured: factorising δD each inner step. The cheap form aligns C to C_ref and uses
δD = 2[(ΔC)Cᵀ + C_ref ΔCᵀ], with X_ref = B C_ref cached from the outer.

Extensions **[G]**:
- The coarse-grid XC in the inner loop works the same way: ρ = ρ_ref + δρ (FP16 GEMM), v(ρ) pointwise in FP64,
  δV = Φᵀ diag(w δv) Φ in FP16. Projected 0.10 → ~0.03 s per cycle.
- Stage 1 must use the increment form against the initial guess with periodic refreshes, because the FP16 floor (6e-4) is
  next to the stage-1 target (1e-3).

### 2.5 Eigensolver: FP32 syevd + Ogita–Aishima refinement [M, GPU]
FP64 syevd (623): 48 ms.
- FP32 syevd: 8 ms, occupied density error 3.9e-7.
- One refinement step (FP64 GEMMs only): 18 ms, 4.6e-10.
- Two steps: 28 ms, 1e-12.

Only the occupied subspace converges; near-degenerate virtuals are not refined, which SCF does not need. In the full `mf.eig`
(0.10 s, generalized problem) the orthogonalisation can be done once per geometry.

### 2.6 Fixed-point integer AO evaluation feeding the RNS GEMM [G, not prototyped]
On a T4 FP64 runs at 1/32 rate. The INT8-CRT path needs ~40-48-bit integers, not doubles. The AO value N r^l Σ c_k e^{−α_k r²} can
be computed in 48-bit fixed point, DSP-style:
- exp(−x) = 2^{−k} · e^{−f}, with a 64-entry table and a degree-5 polynomial, ~30-40 int32 ops on Turing's separate INT32 pipe;
- the per-block row exponent comes from an analytic distance bound instead of a row-max pass;
- residues are written straight to the int8 planes, so AO values are never materialised in FP64.

Estimate: ~6e8 primitive exponentials per fine evaluation at ~2e11/s ≈ 3 ms, against AO 0.46 s plus quantise ~1 s now.
Projected fine XC ~0.6 s. The same kernel serves the deriv-2 AOs of the gradient. A100/H100 do not need it.

### 2.7 Memory
- FP64 packed DF tensor: 12.2 nao³ bytes per operator (24.4 for FR+LR) → T4 limit nao ≈ 790 (~42 atoms TZVP).
- **Error-bounded integer storage [M, GPU]**: B rounded to an absolute step s.

  | step s | dE_K (FR) | bits/element (per element / shared exponent per 16) |
  |---|---|---|
  | 1e-8 | 1.2e-7 | – |
  | 1e-9 | 3.6e-9 | 9.1 / 14.5 |
  | 1e-10 | 3.0e-10 | 11.6 / 17.5 |
  | 1e-11 | 1.6e-11 | 14.3 / 20.7 |

  For comparison, FP32 storage gives dE_K = 4.5e-8. At s = 1e-10, FR+LR needs ~4.4 nao³ bytes. The stored integers already *are*
  the fixed-point operand of the CRT GEMM (global scale), so the K1 quantisation pass disappears. The surrogate can read the same
  integers truncated to their top bits.
- **Exact tensor not stored [G]**: exact K enters only as K·C_o 3-4 times. Integral-direct X = (μν|P) C_o followed by the metric on X
  costs 2 naux² nao nocc = 0.27 TFLOP for BPA (report 3's "1.8 nao⁵" is a typo for nao⁴). Plus 3c integrals per outer, chunked
  over occupied orbitals.
- Capacity on 16 GB (12 GB for tensors) **[G]**:

  | storage | bytes | max nao | ~atoms TZVP |
  |---|---|---|---|
  | current FR+LR FP64 | 24.4 nao³ | 790 | 42 |
  | error-bounded FR+LR integers | 4.4 nao³ | 1400 | 75 (BPA at QZVPP fits) |
  | integral-direct exact + FP16 merged surrogate (full rows / lmax1 rows) | 3.05 / ~1.4 nao³ | 1590 / 2050 | 84 / 108 |

  XC and VV10 have no memory wall.

### 2.8 Gradient fused with the last outer [G]
The last outer step is, by construction, an exact evaluation at the converged density. Two reuses follow:
- **XC gradient:** needs Σ_g ∂φ_μ Y₀ᵢ + Σ_g ∂∂φ_μ Y_y,i. Same Y and same quantised B-operand as V·C_o, plus deriv-2 AO rows in the
  same pass. 5.1 s → ~1.5 s **[G]**.
- **DF-K gradient:** its intermediates (X^L = B C_o → Z^P_ij) come out of the outer K·C_o. Removes the gradient's
  `fill_int3c2e` + transformation DGEMMs, ~3.5 s **[G]**.

---------------------------------------------------------------------------------------------------

## 3. Projected timelines (BPA)

T4 counts from the CPU runs (4 outers including the final check, ~14 inner + 8 stage-1 cycles) and the unit costs of section 1:

| scenario | SCF | how |
|---|---|---|
| now | 36.6 s [M] | crt scheme, 17 cycles |
| A: algorithm only (current XC kernels, FP16-increment K, merged tensor) | ~21 s [G] | 2.2 DF + 8×0.32 + 14×0.30 + 4×3.0 |
| B: + shared-quantisation V·C_o (fine XC ~1.2 s), FP32+OA eigh, increment coarse XC | ~14 s [G] | 2.2 + 22×0.18 + 4×2.0 |
| C: + fixed-point AO→RNS (fine XC ~0.6 s), Anderson-accelerated outers (3) | ~10 s [G] | 2.2 + 3.5 + 3×1.4 |

T4 gradient 14 s [M] → ~4.5-5 s [G] (2.8 + dynamic-precision FP32 Rys for small shell triples: 3.3 → ~0.8 s).
**T4 totals:** B ≈ 19 s, C ≈ 14.5 s, vs the unoptimised A100's 13.5 s and today's T4 ~51 s.

**A100** [G]: fine evaluations become cheap (V·C_o ~0.1 s, K·C_o ~0.07 s) and the inner loop is eigh/DIIS/Python-bound, ~0.04 s per
cycle. SCF 5.5 → ~2.5-3 s, gradient 2.0 → ~1.5 s: 1.6-2x. More needs the inner loop captured as one CUDA graph (fixed shapes,
which the multi-fidelity loop has).

---------------------------------------------------------------------------------------------------

## 4. Evaluated and rejected (or corrected) this round

| idea (source field) | outcome |
|---|---|
| THC/ISDF exchange as inner surrogate (THC: Martinez/Parrish; ISDF: Lin) | [M] contraction 2.5-7x/outer at c ≤ 8 → 9-15 outers. Worse than aux truncation. |
| Low-rank incremental K: K(D_ref) + K(trunc δD) | [M] δD has nearly full rank 2·nocc (14-20 of 20 at 1e-1 relative); no saving. |
| FP16 exchange in absolute form | [M] stalls at 6e-4 (the reason the increment form is needed). |
| torch `_int_mm` (cuBLASLt) instead of legacy `gemmEx` | [M] 16 vs 37 TOPS (XC vmat shape), 2.6 vs 14.7 (V·C_o), 26.7 vs 21.1 (K2), 14.1 vs 14.5 (K1): no win. |
| "Cheap aux" = cc-pVDZ-JKFIT | correction: 232 vs 226 functions, not cheaper. |
| Occupied-defect runs, first version | bug: inner DIIS seeded with the full fine Fock. Rerun with surrogate seed: unchanged (table 2.2). `r4_cpu_defect.py` occ mode is superseded. |
| DEIM / hyper-reduction of the XC nonlinearity (model order reduction) | [reasoning] inner XC is already cheap (coarse grid); fine evaluations must be exact; the reduced-basis matrices Φᵀdiag(wu_k)Φ each cost a full vmat and depend on geometry. Only worth revisiting for MD trajectories. |
| ISDF of the XC quadrature (pair-product interpolation points per block) | [reasoning] the point–function Gram (Φ_Sᵀ Φ per block) costs about as much as the vmat it replaces; even at 25 % rank the net gain ≤ 1.3x; MGGA needs ∇φ∇φ pair spaces too. |
| Sum factorisation / CP-rank of the weighted potential on tensor-product blocks (spectral elements) | [reasoning] w·v is not separable near nuclei; break-even needs CP rank < ~100 on 16³ blocks; Becke–Lebedev grids are already pruned. |
| Fourier / random-feature long-range exchange | [reasoning] k-points needed for erf(0.3r)/r on a 20-bohr box: ~3000-25000 > naux 1900. |
| VV10 exponential-sum or Román-Pérez–Soler interpolation | [reasoning] kernel has two local parameters per point; after the ladder VV10 runs only 3-4 times (~1.5 s). |

---------------------------------------------------------------------------------------------------

## 5. Gradient: the open problem

T4 BPA gradient 14 s [M, handoff]: DF J/K 7 s (FP64 Rys `int3c2e_ip1` 3.3 s, `fill_int3c2e` 0.5 s, DGEMMs ~3 s), XC 5.1 s, NLC 1.07 s.
- Reuse and scheduling (2.8) and dynamic-precision FP32 Rys (Luehr–Martinez style) are the only levers found. Both are known
  techniques or scheduling, not new math.
- Things that did not work out on paper:
  - A defect correction needs a second evaluation, so it does not help a single gradient.
  - Compressing Γ^P in the aux index does not shrink the integral work.
  - McMurchie–Davidson as batched GEMMs has short contractions (~10-35), where INT8-CRT does not pay.
- Candidate research directions:
  - fixed-point/block-floating-point Rys recurrences on the INT32 pipe;
  - double-float Rys roots and weights;
  - manifold-mapping (multi-fidelity) gradients across geometry-optimisation steps, where they pay because many gradients are needed.

---------------------------------------------------------------------------------------------------

## 6. Novelty check (WebSearch, standard mode; "not found" means not found in our search)

| component | source field | prior art found | not found |
|---|---|---|---|
| fidelity-ladder defect SCF | nonlinear multigrid (Brandt FAS); space/manifold mapping | FAS for real-space KS grids (Beck 1999) | component-wise surrogates in a Gaussian DF SCF |
| occupied-projected defect Δ̃ | ACE (Lin 2016), PW codes | ACE for exchange, ACE+ISDF (Hu–Lin–Yang) | ACE compression of a fidelity defect |
| merged FR+LR tensor as surrogate | cuEST single-tensor RS-K | as an approximation | as an exactly-corrected surrogate |
| FP16 increments (iterative refinement) | numerical LA (Haidar, Carson–Higham) | tensor-core S⁻¹ factorisation (Habib–Finkelstein–Niklasson 2024) | Fock-build increments |
| FP32 eigh + Ogita–Aishima | numerical LA (Ogita–Aishima 2018) | – | not searched |
| error-bounded DF-tensor integers feeding RNS | scientific data compression (SZ, PaSTRI for 4-index ERIs) | PaSTRI (GAMESS, 4-index) | 3-index DF tensor on GPU |
| fixed-point AO evaluation | DSP fixed-point arithmetic | – | not searched |
| THC-K (rejected as surrogate) | – | TeraChem GPU THC-J/K (2025), molecular ISDF (2019) | – |

---------------------------------------------------------------------------------------------------

## 7. Implementation order (each step gated: energy ≤ 1e-8 Eh, gradient ≤ 1e-6 Eh/bohr vs `reference/bpa_tight.json`)

1. Multi-fidelity driver in `scf/mixed_precision.py`: outer exact F·C_o, Δ̃, inner surrogate (coarse XC, frozen NLC, exact K at first).
   Uses the existing kernels plus a V·C_o block function; outer count and energy checked on T4 against the CPU numbers.
2. Merged FR+LR surrogate tensor + FP16-increment K (cupy fp16 matmul), aligned-ΔC factorisation.
3. FP32+Ogita–Aishima eigensolver in the inner loop; increment-form coarse XC.
4. Shared-quantisation XC pass (rho + V·C_o + optional gradient rows); B-free K·C_o for large systems.
5. Fixed-point AO→RNS kernel (T4/Ada/Blackwell-consumer option); dynamic-precision Rys for the gradient.
Risks: the CPU counts come from a water dimer and Vitamin C, not BPA; DIIS behaviour of the inner loop at BPA size; stage-1
robustness for difficult SCF cases (metals, small gaps), where the Δ_vv term and ACE-type compression are less benign.

## 8. Data index
`experiments_r4/`:
- GPU: `r4_gpu.py/.json/.log` (R1-R6), `r4_gpu2.py/.json/.log` (G2-G3).
- CPU: `r4_cpu_auxK.py` (main emulation; fixed version) with `dimer_aux_v2.log`, `dimer_aux_v3.log`, `vitc_aux_v2.log`;
  `r4_cpu_thc.py` + `dimer_thc.log`; `r4_cpu_defect.py` + `dimer.log`, `dimer_aux.log` (first versions, inner-seed bug).
