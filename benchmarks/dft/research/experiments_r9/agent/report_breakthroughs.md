# Round 9 (agent): cheaper exact checkpoints, fewer checkpoints, local exchange, gradient redundancy

**Setup.** All results here are CPU emulations with pyscf 2.14, using HF rather than wB97M-V.
- The exact operator is DF def2-universal-jkfit, for both J and K.
- The surrogate is def2-svp-ri DF-K plus exact J.
- The outer/inner protocol is the one in `experiments_r8/r8_gridj_cpu.py`, with inner_factor 1e-3.
- Molecules (def2-TZVP): phenol (nao 253, nocc 25) and BPA (nao 623, nocc 61).

Tags: [M] measured, [E] estimated, [U] untested.

**Scripts** (in this directory):
- `a1_incremental_x.py`: fitted-B stored increments, model-energy prediction, energy-weighted-density check.
- `a3_direct_raw.py`: the direct_k algorithm with raw integrals and M^-1.
- `a2_local_sparsity.py`: BPA Pipek-Mezey screening.
- `a4_taxol_extrap.py`: extrapolation of the screening results to Taxol.

Logs are in `logs/`.

**BPA fp64 baseline:** g = 2.10e-2, 9.05e-5, 7.96e-7, 7.6e-9 (contraction 232 / 114 / 104).

## 1. Model-energy finalization: skip the certifying checkpoint [M on HF; Taxol E]

**Mechanism.** The inner-loop model is Ẽ(D) = E_x(D_k) + E_s(D) − E_s(D_k) + tr Δ_k (D − D_k). Its value at the converged inner density predicts E_x(D_{k+1}).

**Measured prediction error:**
- On 10 points it stayed ≤ 0.12·g_k·g_{k+1}. The points covered phenol, BPA, a jfit surrogate, a crude def2-svp-aux surrogate (ratios ~18) and a loose inner loop.
- Examples:
  - phenol 0→1: 6.1e-8 (factor 0.064)
  - phenol 1→2: 4.4e-12 (factor 0.12)
  - BPA 0→1: 1.2e-7 (factor 0.063)
  - BPA 1→2: 8.5e-12 (factor 0.12)

**Effect on the nuclear gradient.** Taking the energy-weighted density from the model (F_s + Δ) changes the gradient by ≈0.005·g_{k+1}.

**Taxol [E].**
- With g₂ measured at 1.6e-5, the error is ~8e-10.
- If g₂ is only known to be below the tolerance, the bound is 1.6e-9, which is over a 1e-9 budget.
- So this needs a cheap certificate of |g| (section 2's FP16 increment, or a ~7-moduli direct pass). Saving: up to −243 s of 730 s at Taxol.

## 2. Incremental exact checkpoint with stored half-transform X [M]

**Mechanism.**
- Keep X = (P|mn)C between checkpoints and add δX = (P|mn)δC, with δC Procrustes-aligned.
- Update T in FP64, then compute KC = 2 Σ X_P Z_P.

**Measured on phenol (raw / direct emulation):**

| storage | increment | E error | \|g\| error | verdict |
|---|---|---|---|---|
| fp64 | fp32 | 7e-12–9e-12 | ≤3e-10 | works |
| fp64 | fp16 | 1.3e-7 | 8e-8 | certificate only |
| **fp32** | – | **2.1e-7 at k=0** | – | **fails**: M⁻¹ (cond 8e8) cancellation |

- Rule: increment precision × |δC| ≲ 1e-10.
- Observed |δC|: 1e-3 at k=1, 1e-5 at k=2, 4e-8 at k=3.

**Cost and memory [E].**
- Memory: 22.5 GB FP64 per operator at Taxol, 45 GB for FR+LR. That needs host memory or an A100-80GB.
- Time: K exact from 730 to ~500 s.
- Combined with section 1: ~350–460 s.

## 3. Low-rank ΔD/δC: negative [M]

- No rank decay: phenol keeps 25/25 singular values above 1e-10, BPA 61/61.
- Truncating at τ = 1e-8 already gives an energy error of 3.6e-9.
- Use reduced precision for the increments, not reduced rank.

## 4. Direct-K metric step [E]

- Z = W(WᵀT) on naux × nocc² costs 6.4e12 FP64 flops per operator at Taxol, ~26–32 s on a T4.
- Two exact savings:
  - T_P is symmetric, so working on the packed i ≤ j columns gives an exact 2x.
  - On the full-range Cholesky path a triangular TRMM gives another ~2x.
- Estimate: −100 to −140 s at Taxol.
- Applied in round 9: `direct_k.py` now uses the packed columns and has a `metric` timer.

## 5. Localized (PM) exchange sparsity [M on BPA, E for Taxol]

**BPA, flops kept versus Schwarz-only screening:**

| threshold | flops kept | dE |
|---|---|---|
| 1e-10 | 82 % | 2.7e-10 |
| 1e-9 | 77 % | 5e-9 (over budget) |

- There is no (i, j) sparsity.
- Taxol extrapolation: ~70 % kept at 1e-10.
- Verdict: not worth block-sparse kernels at ≤113 atoms.

## 6. Gradient

- The Rys derivative kernel already uses translational invariance; there is no redundant centre derivative.
- [U] Skipping one-centre triples: at most 1–3 %.
- MD/Hermite reformulation and a fused FR+LR pass remain the best options.

## 7. Outer acceleration and COSX surrogate

**Outer acceleration.** Anderson or Broyden on the defect [U] cannot remove the third checkpoint, because that checkpoint is a certificate. Section 1 addresses it instead.

**SGX grid sizes:**

| molecule | level 0 points | level 1 points |
|---|---|---|
| phenol | 10.6k | 45.1k |
| BPA | 25.9k | 107.5k |
| Taxol | 87k | 359k |

Level 0 still contracts 773 / 496 / 285x per step.

## Recommendations
1. Metric symmetry (done) and TRMM.
2. Model-energy finalization, with a cheap certificate.
3. Stored FP64 X with incremental checkpoints, where host or A100 memory allows.
4. Defer local exchange.
