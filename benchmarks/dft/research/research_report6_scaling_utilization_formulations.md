# Research report 6: scaling, hardware utilisation, and new formulations for the remaining cost

Date 2026-11-05. Analysis only; nothing in this report was implemented.

Tags used throughout:
- **[M]** measured (earlier reports or this session's CPU runs).
- **[E]** estimate from component timers and FLOP or byte counts.
- **[G]** projection.

Novelty tags:
- **known**: an existing technique.
- **new combination**: known pieces put together in a way we did not find published.
- **own construction**: a quick search found no precedent. This is not proof that none exists.

---------------------------------------------------------------------------------------------------

## 1. Scaling: the stored DF tensor is the wall

### 1.1 Sizes (CPU, pyscf) [M]

| system | atoms | nao | nocc | naux (univ-jkfit) | naux (SVP-RI) | fine grid points |
|---|---|---|---|---|---|---|
| BPA def2-TZVP | 33 | 623 | 61 | 1567 | 1040 | 1.06 M |
| BPA def2-QZVPP | 33 | 1449 | 61 | 1567 | 1040 | 1.06 M |
| Taxol def2-TZVP | 113 | 2228 | 226 | 5598 | 3690 | 3.63 M |

- **Significant AO pairs** (|S| > 1e-11): 71 % for BPA, 43 % for Taxol.
- **Calibration:** the overlap proxy gives BPA 2.0 GB, but the measured tensor is 2.98 GB, so the proxy is scaled by about 1.5x below.

### 1.2 What has to be stored, and what happens today [E]

| | BPA TZVP | BPA QZVPP | Taxol TZVP |
|---|---|---|---|
| exact full-range tensor | 3.0 GB [M] | 12–16 GB | ~70 GB |
| exact long-range tensor (stored at outer steps) | 3.0 GB | 12–16 GB | ~70 GB |
| surrogate SVP-RI tensor | ~2 GB | 8–11 GB | ~45 GB |
| coarse-grid FP32 AO cache | 0.5 GB [M] | ~1.2 GB | ~5–10 GB |
| **fits a T4 (16 GB GPU, 12.7 GB Colab host)?** | yes | **no** | **no** |

**What the code does when it doesn't fit** (`df/df.py`):
- The tensor stays on the GPU only if it is below 40 % of free memory, or below 95 % of the computed budget on the other path.
- Otherwise the full tensor goes to pinned host memory (`empty_mapped`). On a Colab T4 host (12.7 GB) that allocation fails.

So **neither Taxol/TZVP nor BPA/QZVPP can run on a T4 today, with either the current scheme or the multi-fidelity one.** This, not speed, is the first scaling answer. An A100-40GB also cannot hold Taxol's tensors.

### 1.3 Time scaling, if memory were solved [G, ranges]

Measured basis scaling is much flatter than FLOP exponents suggest. Vitamin C with `crt` went from 28.7 s to 58.7 s between TZVP and QZVPP (nao 420 → 924, HANDOFF §3.5). On that basis:
- **BPA/QZVPP on T4:** about 2–3x BPA/TZVP, so roughly 50–80 s for SCF + gradient.
- **Taxol/TZVP:** by component:
  - Exchange (occupied-RI, ~naux·nao²·nocc): 30–170x BPA, depending on how much pair sparsity can be exploited.
  - Fine XC: 15–45x.
  - Eigensolver: 20–46x per call. syevd at nao 623 is latency-bound, so cubing overstates the growth.
  - Altogether, a T4 would need roughly 4–10 minutes. Exchange and the per-cycle O(N³) diagonalisation would dominate.

### 1.4 Path to native Taxol / QZVPP on a T4 (the user's goal 3)

The multi-fidelity design already isolates the parts that need the exact tensor: only the 3 outer evaluations. So:
1. **Never store an exact tensor.** Make the outer evaluations integral-direct: stream 3c blocks, half-transform them immediately to X = (P|μν)C, and accumulate.
2. **Surrogate exchange in O(N²) memory.** Use an ISDF/THC factorisation of the SVP-RI operator:
   - Inner-loop K becomes pure GEMMs on about 8·nao interpolation points: X 160 MB and a Coulomb matrix of about 1.3 GB at Taxol.
   - Its ~1e-4 error does not matter, because the outer defect removes it. ISDF-K is otherwise limited by exactly that error.
   - (new combination: ISDF/THC-K is known [Lu/Ying, Lin, Hu/Yang; TeraChem THC-K 2025]; using it as a defect-corrected surrogate we did not find.)
3. **Diagonalisation-free inner loop** (§4.5): removes the O(N³)-per-cycle term.
4. **Fused derivative pass at the last outer** (§4.3): makes the gradient's integral work nearly free once the outer steps are direct.

The estimated memory at Taxol is under 10 GB. Each item is substantial work. Items 1 and 2 are the minimum for Taxol to run at all.

---------------------------------------------------------------------------------------------------

## 2. Hardware utilisation (BPA, T4, multi-fidelity, 26.1 s) [E]

The timers in report 5 add up to almost the full wall time, so host and launch time sits inside them and cannot be separated without a trace. The breakdown below is estimated from FLOP and byte counts:

| where the time is spent | ≈ s | share | notes |
|---|---|---|---|
| FP64 units (1/32 of the T4's FP32 rate) | 9–10 | 35–40 % | Rys derivative kernel, 3c integrals of the DF builds, Γ build, FP64 syevd, FP64 AO evaluation, libxc |
| memory-bound passes | 5–7 | ~25 % | quantise and unpack of the AO blocks and DF tensors, residue writes |
| FP32 SIMT | ~4 | ~15 % | VV10 pair kernel, coarse XC, surrogate K increments |
| INT8 tensor-core GEMMs | 2–3 | ~10 % | all the CRT products |
| host, Python and launch gaps | 2–3 | ~10 % | 17 cheap cycles of small ops, plus driver glue |

Anchors:
- **Fine XC:** the INT8 GEMM floor for one evaluation is about 0.19 s (6.5e11 ops × 11 moduli at 37 TOPS), against 1.6 s measured. The tensor cores are busy about 12 % of that time.
- **A100:** the production `nr_rks` measured 0.196 s wall vs 0.195 s GPU-busy [M]. But the new `nr_rks_vco` and the multi-fidelity driver have never been gap-profiled. My estimate is that 30–40 % of the A100's 4.5 s SCF is host- or launch-bound (many 10–50 ms pieces).

**In short:** the T4 is rarely idle, but most of its time is spent on its slowest units (FP64) and on memory passes. The INT8 tensor cores, its fastest unit (130 TOPS), are busy only about 10 % of the time.

One `nsys` trace on a T4 (about 10 minutes of credits) would turn this table into measured numbers. It has not been run.

---------------------------------------------------------------------------------------------------

## 3. Can the integrals be put onto INT8 or FP16 GEMMs? The rank argument

A GEMM needs a factorisation through a shared inner index: (μν|P) = Σ_r U(μν, r) V(r, P).

- **Near field:** the pair × aux matricisation of the 3c tensor is essentially full rank. Its rank is about naux; that is why the metric J is ill-conditioned and why DF works at all.
- So any exact GEMM route needs inner dimension r ≥ ~naux. The cheapest such route is a quadrature over r₁ (pseudospectral/COSX, r = grid points), which costs more than Rys and is not exact.
- **Exp-sum separations of 1/r** (Σ_k w_k e^{−s_k r²}, ~50 terms) factorise per Cartesian direction, but leave a Hadamard product over (pair, aux, k) that is about 10x more pointwise work than Rys. Rejected.
- **Monomial-feature GEMMs of the Rys polynomials:** 5–20x more operations at about 14x the FP64-equivalent rate (INT8 at 14 moduli). This is break-even at best and very complex. Rejected.
- **Number-theoretic transforms (NTT):** these give exact integer convolution, but only convolution-structured work benefits. That would be uniform-grid J, which is not our bottleneck. Rejected.

What is left is a split by structure:
1. **Far field is low rank.** Multipole expansions give rank about (p+1)² for well-separated pair/aux clusters, so those blocks become small GEMMs. The far-field share of (significant atom pair, aux atom) triples (separation > pair radius + 6 bohr) is 38 % for BPA and 70 % for Taxol [M, CPU]. These are triple counts at atom level, not work fractions. Far-field Γ and fitting coefficients are small, so FP16 or FP32 precision is enough there. (known for RI-J as MARI-J [Sierka/Hogekamp/Ahlrichs 2003]; for the DF-K tensor build and gradient on tensor cores, new combination.)
2. **The pointwise near-field core cannot use tensor cores**, because it has no reduction dimension. Its lever is the T4's FP32 SIMT, which is 32x the FP64 rate:
   - plain FP32 for blocks whose a-priori contribution is small;
   - double-float (FP32 pairs built from error-free FMA transformations, ~48 bits) for large blocks.
   - (known as dynamic precision for Fock builds [Luehr/Ufimtsev/Martínez 2011]; the gradient version is §4.4.)
3. **The transforms around the core** are GEMMs with short reductions: Γ = C Z Cᵀ (K = nocc), contraction with C, Cartesian → spherical. INT8 CRT lost here because the Garner reconstruction per output costs as much as a K = 61 product. Ozaki-I on FP16 tensor cores avoids reconstruction (§4.4).

**Conclusion:** the near-field integrals themselves cannot become INT8 GEMMs. What can change is how much of them is near field (grows in our favour with size), the precision of the core, and where the transforms run.

---------------------------------------------------------------------------------------------------

## 4. New formulations for the remaining cost

### 4.1 Precision follows the residual: an inexact defect schedule (new combination)
- **Idea:** in defect correction (as in inexact Newton [Dembo–Eisenstat–Steihaug]), outer step k only needs a relative accuracy of about c·r_k. Only the certifying last evaluation must be exact.
- **Current residuals:** 7.7e-3, 7.7e-5, 7e-7. So:
  - outer 1 can use the (75,302) XC grid (~40 % of the points), the coarse NLC grid, and 7–8 moduli;
  - outer 2 can use 9 moduli;
  - only the last outer needs full precision.
- **Fixed point:** still exact. The converged density is certified by an exact W, and inexact earlier steps only change the path.
- **Risk:** if an inexact outer lands near tolerance, a 4th full evaluation is needed, which costs more than the whole inner loop. The expected saving below includes that risk.
- **Precedents:** loose-early thresholds and dynamic integral precision are known [Q-Chem variable thresholds; Luehr 2011]. Varying grid, aux and moduli per outer inside an exactified surrogate scheme we did not find.

### 4.2 One AO quantisation, generated in-kernel at the bit width actually consumed (own construction)
Fine XC currently quantises each AO block twice: per grid point for ψ = CᵀA, per AO row for V·C = Σ A Yᵀ.

- **Two-sided scaling:** give the block one rank-1 power-of-two scaling, A = diag(s) Ã diag(t), balanced Sinkhorn-style.
  - For ρ: ψ = diag(t)·(diag(s)C)ᵀ Ã.
  - For V·C: diag(s)·Ã·(diag(t)Y)ᵀ.
  - The scalings fold into the small operands C and Y, so a **single** integer Ã serves both GEMMs. The error bound loses at most 1–2 bits against per-axis scaling.
- **Analytic scales:** s and t come from basis-function magnitude bounds on the block, so the AO kernel writes int8 residues directly and never stores FP64 AO values.
- **Precision of AO generation:** with 11 moduli the operand precision is only q ≈ 32 bits. The AOs can therefore be evaluated in double-FP32 (~48 bits) instead of FP64.
- **Effect:** removes the FP64 AO write, one of two quantise passes, and the re-read. Est. 1.6 → 0.7–0.9 s per fine evaluation, and the same factor on the deriv-2 pass for the fused gradient and on NLC-grid ρ.
- **Precedents:** two-sided power-of-two scaling is standard in Ozaki-II [Ozaki/Uchino/Imamura]. Sharing one quantisation between the A and Aᵀ products, and generating the residues in the producing kernel at the consumed width, we did not find.

### 4.3 Lagged-Γ fused derivative pass with an FP32 ΔΓ correction (own construction)
- **Identity:** g(Γ) is linear in Γ, so g(Γ_final) = g(Γ_prev) + g(Γ_final − Γ_prev).
- **Construction:**
  - When the exact outer steps are integral-direct (§1.4), the last outer's value pass also evaluates the derivative integrals. This shares Rys roots and recurrences; a derivative is +1 angular momentum on the bra, about +60 % cost.
  - These derivatives are contracted with Γ_prev, which is known from the previous outer.
  - Then one FP32 pass contracts ΔΓ = Γ_final − Γ_prev, which is ~1e-4 relative. That pass is screened by ‖ΔΓ_block‖, so most blocks are skipped.
  - The error is about 1e-7 × 1e-4 × |g terms|, far below 1e-6 Eh/bohr.
- **Effect:** removes the separate full FP64 derivative pass.
- **Limits:** at BPA the exact tensor is stored, so the outer steps have no value pass. There it costs an extra value pass and is about break-even. It pays at Taxol scale or for QZVPP, where outer steps must be direct anyway: est. gradient 3c work drops 3–5x there.
- **Precedents:** not found in a quick search.

### 4.4 Gradient at BPA size: Γ-weighted precision tiers, bits allocated by reverse water-filling (new combination)
- **Bounds:** Γ is known before the derivative kernel runs. Each (shell pair, aux shell) block gets a bound b = ‖Γ_block‖·(derivative Schwarz-type bound).
- **Tiers:** FP64, double-FP32, FP32, or skip.
- **Allocation:** choose tiers to minimise cost subject to Σ (ε_tier·b)² ≤ τ², the reverse water-filling rule from rate–distortion theory, with τ = 1e-7 Eh/bohr.
- **Short-reduction Γ build:** do Γ = C Z Cᵀ as Ozaki-I on FP16 tensor cores: 9-bit slices, products exact in FP32 accumulation, 6 slices / 21 products. Est. 0.45 → 0.1–0.15 s per operator. (Ozaki-I on tensor cores is known [Mukunoki, Ootomo]; using it where CRT failed is ours.)
- **Effect:** est. DF J/K gradient 4.7 → 1.8–3 s on T4. Dynamic precision itself is known (TeraChem); the Γ-weighted gradient tiers and the water-filling allocation are the new part.

### 4.5 Diagonalisation-free inner loop (known pieces, new use)
- **Rotation step:** replace syevd in inner cycles with a preconditioned occupied-subspace rotation: κ = −F_vo/(ε_v − ε_o), 2–3 steps, then Cholesky re-orthonormalisation.
- **Cost:** GEMMs of nao²·nocc. Increments are low precision, so FP16 tensor cores or FP32 suffice; the occupied block stays FP64.
- **Gradient:** the RKS gradient needs only occupied energies, which we already canonicalise. So no full diagonalisation is needed after the initial guess.
- **Effect:** BPA −0.5 to −0.7 s; Taxol −15 to −40 s.
- **Rejected alternative, FP32 eigensolver:** at nao 623 syevd is latency- and BLAS2-bound, so FP32 lost [M]. At nao ≥ 2000, FP32 syevd plus Ogita–Aishima refinement with CRT GEMMs should win (~5x), but the rotation step is better still.
- **Precedents:** pseudo-diagonalisation [Stewart 1982], curvy steps, SP2 purification on tensor cores [Niklasson et al.] are known.

### 4.6 VV10 on tensor cores via low-rank far field (new combination)
- **Interpolation:** interpolate the kernel's two per-point parameters (ω₀, κ) on a 2D node set, Román-Pérez–Soler style but for original VV10, not the modified rVV10. The pair sum becomes a few smooth radial kernel sums.
- **Far field:** in hierarchical-matrix form, well-separated block pairs are low rank, so FP16 tensor-core GEMMs work there. The near field stays direct in FP32.
- **Effect:** est. NLC (fine evaluations + gradient, 2.7 s) → 1.2–2 s. Moderate confidence; error control needs a test.

### 4.7 Orchestration (known)
- **CUDA graphs:** one graph per inner cycle (all shapes are fixed) and per outer pass.
- **Effect:** T4 −0.5 to −1 s; A100 SCF 4.5 → ~3–3.5 s together with a tuned FP64 tensor-core V·C kernel.

---------------------------------------------------------------------------------------------------

## 5. What it adds up to (BPA, T4) [G]

Savings are grouped by the cost they attack, so nothing is counted twice:

| cost (now) | ideas | after |
|---|---|---|
| fine XC 7.0 s | 4.1 + 4.2 (+ increment outers, report 5) | 3–4.5 s |
| exact K + DF builds at outers 3.9 s | 4.1, K increments | 3–3.5 s |
| DF J/K gradient 4.7 s | 4.4 (4.3 only at scale) | 1.8–3 s |
| NLC fine + gradient 2.7 s | 4.1 (coarse at outer 1), 4.6 | 1.2–2 s |
| cheap cycles, eig, setup, glue, fit 7.8 s | 4.5, FP16 coarse increments, 4.7 | 5–6.5 s |
| **total 26.1 s** | | **~14–19.5 s** |

- **Parity:** the unoptimised A100 does SCF + gradient in 13.5 s. Even the optimistic end only gets close, so **parity on BPA is probably still not reached**. The structural reason is the FP64-bound integral core (§3).
- **A100:** ~3.5–4 s, down from 5.4 s.
- **Beyond BPA:** the larger payoff of this list is in §1.4. Items 1–4 there make Taxol and QZVPP runnable on a T4 at all, and the relative gains grow with size (far field 38 % → 70 %, diagonalisation N³ removed).

## 6. Suggested order
1. **nsys trace on T4** (cheap): turns §2 into data and guides the rest.
2. **4.2 + 4.1** (largest BPA cut, self-contained).
3. **4.4** gradient tiers.
4. **§1.4 items 1–2** (direct outer steps + ISDF surrogate), then a Taxol run.
5. **4.3, 4.5, 4.6.**

Sources for the precedent checks:
- https://manual.q-chem.com/6.3/sect_incfock.html
- https://arxiv.org/pdf/1505.07851
- https://arxiv.org/pdf/2504.08009
- https://agenda.isc-hpc.com/media/2026/slides_pdf/14-15_ISC2026_Uchino_08.pdf
