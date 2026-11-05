# Research report 3: fundamental restructuring of wB97M-V + DF (T4 parity with A100, memory scaling)

Date 2026-11-05. Builds on `research_report1_literature.md`, `research_report2_flop_reduction.md`, `HANDOFF.md`.
Legend: [M] measured this session, [Lit] read in literature (abstract level unless stated), [G] my estimate/derivation.
All [M] numbers below are **CPU PySCF 2.14 algorithm-behaviour experiments** (counts of expensive evaluations, ranks); they carry no
hardware timings. Scripts and logs: `research/experiments_r3/`. GPU confirmations are listed in section 6 (T4 session `t4r3`).

---------------------------------------------------------------------------------------------------

## 0. Bottom line

1. **The cost is a product: (number of full-fidelity evaluations) x (cost of each) + fixed per-cycle glue.** After the INT8 work the T4 SCF
   (39.6 s) is 17 cycles; 9 of them are full-grid. Making each GEMM cheaper no longer pays much; reducing the *number* of full-fidelity
   evaluations does. This session's CPU experiments show it can be cut by ~3x (13 -> 3-4 fine XC evaluations) with an exact fixed point.
2. **Can the T4 match the unoptimized A100 (10.95 s SCF + 2.5 s gradient)?** Not with what I can identify, honestly. Projected T4 BPA
   (section 4, all [G], to be confirmed on GPU): SCF ~22-26 s exact, ~17-20 s with opt-in approximations; gradient 14 s -> ~8 s;
   i.e. **~1.6-2x slower than the unoptimized A100**, not parity. The gap that remains is physics (T4: 320 GB/s vs 1555 GB/s, 0.25 vs 19.5
   FP64 TFLOP/s, INT8 recon overhead) plus the gradient, which is now >50 % of the total. Parity becomes realistic only when work is
   *amortised across a trajectory* (geometry optimisation / MD: reuse Delta and DF/AO caches, 1 outer iteration per step, ~6-8 s SCF [G]).
3. **Memory**: the DF tensor is 12.2 nao^3 bytes (FP64, packed, per operator). On a 16 GB T4 that is ~nao 1000 (~52 atoms TZVP) for one tensor,
   ~nao 800 for FR+LR. FP32 storage (2x), a single transformed tensor (A1), and host streaming (K arithmetic intensity = nocc FLOP/byte, PCIe-bound
   but balanced with INT8 compute at ~90 atoms) move the wall to ~100 atoms. Details in section 5.

---------------------------------------------------------------------------------------------------

## 1. Experiments run (CPU, PySCF 2.14, wB97M-V DF def2-universal-jkfit)

### A. Numerical rank of AO blocks (idea: low-rank/ID compression of the collocation matrix per grid block) -- NEGATIVE
`experiments_r3/rank_test.py`: BPA def2-TZVP, 12 random Morton-ordered 4096-point blocks, stacked [phi, d_x phi, d_y phi, d_z phi], singular values vs
the largest. Rank / nao_sub: **0.40-0.55 at 1e-6, 0.55-0.75 at 1e-8, 0.61-0.87 at 1e-10**, 0.8-0.95 at 1e-12. Accuracy gates need >=1e-9, so
the (r/nao_sub)^2 FLOP saving of a factored vmat is ~0.4-0.7 at best, before ID conditioning losses. Dead; only the coarse stage (1e-6) could use it,
and the coarse stage is not where the time is. (Smaller/more compact blocks lower the rank but also GEMM efficiency.)

### B. Defect-corrected dual-grid SCF (E3 of report 2) -- POSITIVE, strongly
`dualgrid_cpu.py`, `dual_ace_cpu.py`. Surrogate F_s(D) = h + J + K(D) + Vxc_coarse(D) + Delta, Delta = (Vxc_fine - Vxc_coarse)(D_ref) frozen; iterate
the surrogate to a loose tolerance, re-evaluate the true fine Fock at the new D_ref, repeat (inexact Newton with the coarse-grid Jacobian).

| system | baseline fine XC evals to |g|<1e-7 | dual-grid: stage-1 coarse cycles | outer (fine) evals | |g_true| sequence |
|---|---|---|---|---|---|
| water dimer, SVP, coarse (50,194)/(30,110) | 13 | 7 | **3** | 8.7e-4 -> 4.7e-7 -> 6e-10 |
| water dimer, SVP, **coarse (20,50) both grids (26x fewer points)** | 13 | 7 | **3** | 6.8e-3 -> 7e-6 -> 1.5e-8 |
| Vitamin C (20 atoms), SVP, coarse (20,50) | >=14 (killed at cycle 11, |g|=8e-5) | 10 | **4** | 1.9e-2 -> 1.5e-4 -> 7e-7 -> 8e-9 |

Each outer step contracts the true residual by 100-1800x: the grid defect is smooth, so the coarse Jacobian is excellent (even a 26x smaller grid).
Energy: after the first inner solve the *energy* is already exact to ~1e-12 (E error ~ g^2: g=8.7e-4 gave 7.8e-8, g=7e-6 gave <1e-12), so an
**energy-only single point needs just 2 fine evaluations**; SCF+gradient needs 3 (the final_grad_factor requirement).

### C. Freeze the whole NLC (VV10) potential at the reference density -- POSITIVE
Inner cycles use coarse semilocal XC only; V_nlc enters through the frozen part of Delta. Outer count unchanged (3, water dimer; 4, Vitamin C).
**VV10 evaluations drop from every cycle (~18) to the fine outer evaluations (3-4)** (+ the stage-1 coarse ones, which can be dropped too).

### D. Frozen ACE exchange as the K surrogate -- NEGATIVE for wB97M-V
Inner cycles use the ACE operator built from the reference orbitals (K frozen). Contraction only ~0.4-0.5 per outer: **10 outer / 10 fine XC / 17 exact
K** versus 3 / 3 / 23 with exact K inside. 100 % long-range exchange makes K far too sensitive to the density to freeze. ACE only pays if K is applied to the
*current* orbitals (E4: K C_o, ~0.6x K), never as a frozen operator.

### E. Small-basis K surrogate (basis projection P: big -> def2-SVP, K_s = P^T K_small(P D P^T) P, with exact defect at D_ref) -- MIXED
Water dimer, big def2-TZVP / small def2-SVP: contraction ~10x per outer (vs ~100-200x with exact K): 6 fine evals instead of 4, but only 6 exact K builds
instead of 27. Not a win at BPA scale (K = 0.3 s/eval on T4, a fine evaluation 3 s), **but the trade reverses where K dominates (N^4): >=100 atoms**,
which is also where K memory hurts. Keep as the large-system option, not for BPA.

---------------------------------------------------------------------------------------------------

## 2. What the experiments change in the ranking (relative to report 2)

* **E3 (dual-grid) is confirmed and is the structural change**: it removes ~9 of 13 fine XC evaluations and ~14 of ~18 NLC evaluations.
* **After E3 the per-fine-evaluation savings (E4 occupied-projected V C_o, E6 SYRK tau, E8 one AO pass) matter less** (only 3-4 fine evals left: E4 now
  saves ~3.5 x 0.9 s ~ 3 s, not 7-13 s). Their rank drops; E4's PC-DIIS machinery (Hu-Lin-Yang, arXiv:1708.06485 [Lit]: DIIS on a projected commutator,
  validated for hybrids) is still the right DIIS formulation if built, but it is no longer first.
* **The new dominant cost is the number of K builds and fixed glue**: stage 1 (7-10 cycles) + inner solves (~16) = ~23-37 exact K evaluations, more than the
  baseline's 13-17. Remedies: (i) loose inner tolerance (inner_tol = 0.1 g_true gives 5 outer but only ~14 inner cycles; 1e-2 is the sweet spot: 4 outer / 38 K at
  water dimer TZVP); (ii) cheaper K inside the loop (E: small-basis surrogate; A1 single transformed tensor halves K; K C_o); (iii) a **stage 0** in a small basis
  (SVP converged in ~0.1 s cycles, project up) replacing stage 1; (iv) cut glue per cycle (eigh, DIIS, launches: ~0.15-0.25 s now).

---------------------------------------------------------------------------------------------------

## 3. New/refined directions (ranked for T4 BPA; exact unless stated)

| # | idea | mechanism | status |
|---|---|---|---|
| N1 | **Defect-corrected multi-fidelity SCF** (per-component fidelity ladder: coarse semilocal XC every inner cycle, NLC frozen, K exact; refresh Delta at each outer; always finish on a true fine evaluation) | inexact Newton with a cheap, smooth Jacobian model; fixed point identical to fine SCF | **validated on CPU (B, C)**; build in `mixed_precision.py` / `dft/rks.py::get_veff` |
| N2 | Energy-only single points: 2 fine evals (second-order energy) | E error ~ g^2 | validated (B) |
| N3 | Amortisation across geometry steps: carry Delta, DF/AO/grid caches, extrapolated density | Delta is smooth in geometry; first outer already near 1e-4 | [G]; test with an optimisation trajectory |
| N4 | K inside the loop: K C_o (ACE-style applied to current orbitals), single transformed tensor (A1), small-basis surrogate for >=100 atoms | see D, E | partially validated |
| N5 | Stage 0 (small basis, super-coarse grid) | replaces 5 s of stage 1 | needs T4 timing; earlier SVP-seed test was negative under the old cost model |
| N6 | Gradient: FP32/double-float contraction of the XC gradient products (K=550, FP64 1.8 s on T4), FP32 Rys with density-weighted threshold, j3c_oo from the SCF tensor | gradient is 14 s = >50 % of total after N1 | GPU tests in section 6 |
| N7 | FP32 storage of the DF tensor (2x memory, 2x less quantizer traffic, 8-9 moduli suffice) | error floor ~6e-8 per element, random | GPU accuracy test needed |
| N8 | Fused AO->quantize (no AO materialisation), double-float AO evaluation | T4 FP64 is 1/32 rate, AO eval is ALU-bound | speculative |

Considered and dropped this round: low-rank AO blocks (measured, A), frozen ACE exchange (measured, D), empirical-cubature/ECSW sparse XC grids
(MOR literature has the machinery [Lit]; nothing found in DFT; unnecessary because 26x-coarser plain grids already give a contraction of 1e-2..1e-3),
Fourier-space LR-K (same information as the aux basis), tensor-train/QTT (non-ring-native, discussed in notes).

---------------------------------------------------------------------------------------------------

## 4. Projected T4 timeline, BPA (all [G]; unit costs from the HANDOFF profile: full XC 1.9 s, coarse XC 0.6 s, K 0.5 per cycle (FR+LR), NLC 0.45 full / 0.2 coarse, DF build 2.2 s)

Baseline check: 8 x (0.6+0.5+0.2+0.1) + 9 x (1.9+0.5+0.45+0.1) + 2.2 = 40 s (measured 39.6).

| step | SCF s | note |
|---|---|---|
| now | 39.6 | |
| N1 (stage 1: 7 x 0.75; outer 4 x 3.1; inner 16 x 0.75; DF 2.2) | ~32 | dual-grid alone, K-bound |
| + E6/E8/E4-type per-eval cuts on the 4 fine XC (1.9 -> ~1.1) | ~29 | |
| + cheaper in-loop K (A1/K C_o: 0.5 -> 0.3) | ~24 | |
| + stage 0 / better glue (0.75 -> 0.45 per cheap cycle) | ~19-22 | |
| + opt-in grid (SG-3), FP32 Rys in DF build | ~17-19 | |
Gradient: 14 s -> N6 (FP32 contraction, FP32 Rys, j3c_oo reuse, DF grad 7 -> ~3.5; XC grad 5.1 -> ~2) -> ~7-8 s.
**Total T4 ~ 26-30 s exact, ~22-25 s with opt-ins versus 13.5 s unoptimized A100 (1.6-2.2x slower).** Everything here is arithmetic on measured unit costs; the
two numbers that decide it are the real cost of a cheap inner cycle and the FP32 gradient accuracy (section 6).

---------------------------------------------------------------------------------------------------

## 5. Memory scaling on a 16 GB T4 (TZVP-like: nao ~ 18.9 per atom, naux ~ 3.05 nao, nocc ~ 0.098 nao) [G, formulas checked against BPA: 2.95 GB]

| representation | bytes | max nao on T4 (12 GB for tensors) | atoms (TZVP) |
|---|---|---|---|
| FP64 packed B, one operator | 12.2 nao^3 | 995 | ~52 |
| FP64, FR + LR (current mixed_outside) | 24.4 nao^3 | 790 | ~42 |
| FP32 B, one operator (N7) | 6.1 nao^3 | 1250 | ~66 |
| single transformed tensor, FP32 (A1+N7) | 6.1 nao^3 | 1250 | ~66 |
| host-streamed FP32 (64 GB host RAM) | 6.1 nao^3 on host | 2190 | ~116 |
| half-transformed X = B C_o (recomputed integrals) | 2.4 nao^3 | 1710 | ~90 | but the aux-metric solve costs ~1.8 nao^5 FLOP -> infeasible; dropped |

Streaming: K1+K2 fuse per aux slab (K = sum_L X^L X^L^T), arithmetic intensity = nocc FLOP/byte (~61 for BPA, ~190 at 100 atoms); T4 INT8-CRT
(~3.4 TFLOP/s FP64-equivalent) needs 56 GB/s, PCIe3 gives ~12 GB/s: **transfer-bound**, ~5 s/K at 90 atoms FP64, ~2.7 s FP32; with N1+small-basis surrogate
only ~4-6 exact K builds per SCF are needed. The Colab T4 host has 12.7 GB RAM, so host streaming cannot be demonstrated there.
XC and VV10 need only block buffers (no memory wall); crt int8 planes obey the existing budgets.

---------------------------------------------------------------------------------------------------

## 6. GPU confirmations to run on T4 (session `t4r3`; results appended below as they arrive)

1. Real cost of a cheap inner cycle (XC on a (20,50)-like grid, K, J, eigh, DIIS) on BPA -> checks the 0.45-0.75 s assumption.
2. torch `_int_mm` (cuBLASLt) vs the legacy `gemmEx` 37 TOPS on the XC/K shapes.
3. FP32 vs FP64 XC-gradient contraction error on BPA (gate 1e-6 Eh/bohr).
4. FP32 storage of the DF tensor: energy shift on BPA.

(Results: see section 7 once filled in.)

---------------------------------------------------------------------------------------------------

## 7. GPU results

(pending)

---------------------------------------------------------------------------------------------------

## 8. Sources

[Lit] Hu, Lin, Yang, Projected Commutator DIIS for hybrid functionals, arXiv:1708.06485 (abstract); Lin, ACE, arXiv:1601.07159 (report 2); Sharma-White-Beylkin
pseudospectral/ISDF exchange, arXiv:2207.04636 (abstract); NNLS reweighting/pruning of THC grids, arXiv:2604.03899 (abstract); aMECP+aCOSx hybrid analytic-numerical Fock,
arXiv:2411.01526 (abstract: coarse grids suffice when the analytic part is extracted); GPU multigrid GPW in GPU4PySCF, arXiv:2603.24881 (abstract); Huang-Shao-Hammond,
arXiv:2601.08077 (report 2); cuEST docs (density fitting, GEMM emulation); empirical cubature/ECSW (Farhat; Hernandez; arXiv:2308.03877, abstract); Laikov arXiv:2511.04169 (report 1);
QUOTR quasi-Newton SCF (PCCP 2024, abstract). Searches for XC low-rank/ISDF-on-grid, defect-corrected multigrid SCF, frozen-NLC lag found **no prior work** (negative literature result).
