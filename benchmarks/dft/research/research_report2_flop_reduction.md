# Research report 2: FLOP-level / architectural changes for wB97M-V + DF on T4-class GPUs (and A100/H100)

Date 2026-11-04. Scope: new items only. Not repeated (see `research_report.md`): Ozaki-II tricks, cuBLAS emulation, THC/ISDF K, Laikov DF-VV10,
occupied-projected Fock as a *concept* (re-ranked and corrected below), lagged NLC potential, SYRK for K2, TF32 3-split.
Repo state read: HEAD f01d858 (fused XC loops merged, small-basis-seed experiment recorded as NEGATIVE, XC-grad CRT note).

Legend: [V] = I read the passage in the primary source (PDF/HTML text) in this session. [S] = search snippet / abstract page only.
[G] = my own derivation or guess (arithmetic shown). [M] = measured in this repo (from the RESULTS/NOTES files). "Exact" = same fixed point as
the current code to the 1e-8 Eh / 1e-6 Eh/bohr gate. "Opt-in" = changes the answer.

---------------------------------------------------------------------------------------------------------

## 0. Cost model used for every estimate (T4, Bisphenol A, nao 623, nocc 61, naux ~1900, 1.06 M XC points, 217 k NLC points)

Measured: SCF 48.7 s (XC 26, K 10 incl. DF build 2.3, NLC 10, rest 2.7); gradient 16.8 s steady (DF J/K ~8 incl. Rys ip1 3.3, XC 5.6, VV10 3.3). Total 65.5 s.

Unmeasured, **assumed** (the repo does not store per-stage cycle counts; `mf.cycles_coarse` is only logged): 6 coarse-grid cycles + 6 full-grid
cycles (12-cycle FP64 baseline [M]; the seed experiment saved "2 coarse + 1 full" [M]). Per-cycle costs chosen so that the total is 46-49 s:

| cycle type | XC | K | NLC | other | total |
|---|---|---|---|---|---|
| coarse (50,194) grid, 8 moduli | 0.65 | 0.55 | 0.80 | 0.2 | 2.2 s |
| full grid, 11 moduli | 3.5 | 0.65 | 0.80 | 0.2 | 5.2 s |

6 x 2.2 + 6 x 5.2 + 2.3 (DF build) = 46.7 s. Inside a full XC evaluation (3.5 s): vmat INT8 GEMM+quantize+reconstruct ~2.2-2.5 s (35.8 ms per
K=61440 block at 12 moduli x 69 blocks [M, round 3]), rho GEMMs 0.3, AO evaluation 0.36-0.46 (both loops) [M], libxc/elementwise 0.2.
Sensitivity: with 8 full cycles instead of 6 every "fewer full cycles" gain below grows by ~1/3; with 4 it shrinks by ~1/3.

Verdict that frames everything: **after the INT8 work the T4 profile is no longer GEMM-bound in the FLOP sense; what is left is (i) the nao^2*ng vmat
GEMM that exists only because we assemble a full Fock matrix, (ii) per-cycle work that is repeated at full cost although it barely changes
(fine-grid XC, NLC, K), and (iii) a few exact-but-legacy code paths in the gradient.** The ranked list attacks those three.

---------------------------------------------------------------------------------------------------------

## 1. Ranking table (gain x 1/effort); T4/BPA numbers, [G] unless marked

| # | item | class | SCF / grad saving (s) | end-to-end | effort | risk to 1e-8 / 1e-6 gate |
|---|---|---|---|---|---|---|
| E1 | VV10 gradient contraction through the contract-first path that XC grad already has (`grad/rks.py::get_nlc_exc`) | exact | grad -2.5 | -3.8 % | XS (hours) | none (identical to 1e-13) |
| E2 | Coarse NLC grid in the coarse SCF stage (`mixed_precision._coarse_grids` only swaps `mf.grids`, not `mf.nlcgrids`) | exact | SCF -3.4 | -5 % | XS | none (final stage re-converges) |
| E3 | Dual-grid (defect-corrected) SCF: surrogate Fock = coarse-grid XC/NLC + frozen (fine - coarse) defect; 2-3 fine evaluations instead of ~6 | exact if validated | SCF -10..-14 | -15..-21 % | S-M | must measure defect smoothness first |
| E4 | Occupied-projected Fock F C_o (XC V C_o, K C_o, ACE-style reuse) with DIIS on a Fock reconstructed from F C_o + lagged F_vv | exact | SCF -7..-13.5 (already net of 0-2 extra cycles) | -11..-21 % | M-L | 0-2 extra cycles; occupation reordering in small-gap systems |
| E5 | DF-JK gradient: (i) finite-difference check whether FR+SR vs the SCF's FR+LR is a real gradient inconsistency; (ii) build j3c_oo from the cached SCF DF tensor instead of re-running the FP64 3c2e fill | exact | grad -2.4 (LR pass may be slower) | -3.7 % | M | possible ~1e-6 Eh/bohr inconsistency (unverified) |
| E6 | Symmetric (SYRK-tile) tau-term GEMMs in the INT8 vmat; fuse B-build into the quantizer | exact | SCF -2..-2.5 | -3.5 % | M | none |
| E7 | Increment-in-W vmat: V(D)=V(D0)+ao^T (W(D)-W(D0)) ao, INT8 moduli sized by |dW| | exact | SCF -2..-3 (overlaps E4) | -3..-4 % | S-M | accumulation of quantization error |
| E8 | One AO evaluation per block and cycle (rho -> libxc -> vmat in a single block pass) | exact | SCF -1.5 | -2 % | S | none |
| A1 | cuEST-style single-tensor range-separated DF-K: K = sum_C (B Z)_{mu i C}(B Z)_{nu i C} with Z Z^T = 0.15 I + 0.85 L^-1 L_erf L^-T | opt-in | SCF -4.7, grad -3.5 | -12 % | M | error ~E3's 7e-7 Eh [G], must measure |
| A2 | SG-3 pruned grid (DE2 radial, Lebedev partitions) instead of PySCF's (99,590) nwchem-pruned | opt-in | SCF -10, grad -2.6 | -19 % | S-M | tens of micro-Eh vs default grid |
| A3 | Dynamic-precision FP32 Rys for 3c2e fill and ip1 (density-weighted threshold) | opt-in (controllable) | grad -2.6, SCF -0.7 | -5 % | M-L | threshold-controlled |
| - | rejected / not applicable (section 5) | | | | | |

Combined (exact set E1-E8, overlaps removed): 65.5 s -> roughly 35-42 s on T4 (1.6-1.9x) [G; see section 4 for the arithmetic].
With A1+A2 on top ~25-30 s [G, very rough].

---------------------------------------------------------------------------------------------------------

## 2. EXACT / SAFE items

### E1. VV10 gradient: use the contract-first contraction (rank 1, exact, ~2.5 s of 16.8 s)

What (verified in the repo, [V] by reading code): `grad/rks.py::get_nlc_exc` (the default, no-grid-response branch) still ends with the *legacy* loop
`vmat_tmp = _gga_grad_sum_(ao_mask, wv); exc1[mask] += einsum('nij,ij->ni', vmat_tmp, dm_mask)`, i.e. three FP64 GEMMs of size nao_sub^2 x ng per block.
`get_exc` already replaced exactly this pattern by `_exc1_block(ao, wv, (c_occ[idx],), xctype)` (5 GEMMs, applies D as C(C^T x), 0.14 of the dense
FLOPs, results identical to 1e-14 [M]). `profile_grad.py` already sets `VV10_PRECISION='fp32'`, so the FP32 pair sum (~0.35-0.4 s at N=217 k,
scaled from 0.69 s at N=303 k [M]) is not what costs 3.3 s.
Arithmetic: legacy FLOPs = 3 comps x 2 x nao_sub^2 x N_nlc = 6 x 550^2 x 2.17e5 = 3.9e11 FP64 -> 3.9e11 / 0.2 TFLOP/s = 2.0 s, plus the rho and AO(deriv=2)
evaluation. Contract-first: 2 x nao_sub x nocc x N per product x ~3 products = 6 x 550 x 61 x 2.17e5 = 4.4e10 -> 0.22 s. Saving ~1.8-2.6 s (the
profile figure 3.3 s minus pair sum 0.4 minus AO/rho 0.4 minus 0.3).
Sketch: in `get_nlc_exc` build `c_occ = mo_coeff[:, occ] * sqrt(occ)` as in `_get_exc_task`, and call `_exc1_block(ao_mask, wv, (c_occ[mask],), 'GGA')`;
reuse `_blocked_weights`. The same one-line change applies to `get_nlc_exc_full_response` (grid_response=True branch), which is not on the default path.
Risk: none. Effort: XS.

### E2. Coarse NLC grid in the coarse stage (exact, ~3.4 s)

What: `scf/mixed_precision.py::_coarse_grids` replaces only `mf.grids`; `mf.nlcgrids` stays (50,194) in the coarse stage, so the NLC cost (10 s = 21 % of the
SCF) is paid at full price in the 6 cycles whose only job is to reach |g| < 1e-3. Use e.g. (30,110) (3.9k vs 9.7k points per atom => N_nlc x0.4).
Arithmetic: NLC call 0.8 s = pair sum 0.35 (N^2 => x0.16 = 0.06) + rho/vmat/Morton/AO 0.45 (N => x0.4 = 0.18) = 0.24 s. Saves 0.56 s x 6 coarse cycles = 3.4 s.
Why exact: the coarse stage is only a starting density; the final stage re-converges on the production NLC grid (same logic as the existing coarse XC grid [M]).
Correction to a statement in the task brief ("Richardson/correction for NLC"): Richardson extrapolation in the grid size is **unsound** (quadrature error is
not an asymptotic series in the point count); the defect correction of E3 is the sound version.
Sketch: add `with _coarse_nlc(mf, opt['coarse_nlc_grid'])` next to `_coarse_grids` (copy `mf.nlcgrids`, `reset()`, set `atom_grid`). Effort XS. Check: SCF
cycle counts of the final stage must not change.

### E3. Dual-grid / defect-corrected SCF (exact if the defect is smooth; est. -10..-14 s; rank 3)

What: F_surrogate(D) = J + K + [Vxc_coarse(D) + Delta_xc] + [Vnlc_coarse(D) + Delta_nlc], with Delta = (F_fine - F_coarse)(D_ref) frozen at the last
reference density. Iterate the cheap surrogate to convergence; re-evaluate Delta at the new density (one fine XC + one fine NLC evaluation); stop when the
*true* orbital gradient at the reference density is below tolerance. The fixed point is identical to the fine-grid SCF because Delta is re-evaluated at
the end (surrogate == true Fock at D_ref); only the path changes.
Why it should work here and why it is NOT the failed experiment E2 [M]: E2 corrected *FP32 rounding noise*, which decorrelates between densities (ratio 1.1 for
Vxc). The fine-minus-coarse grid defect is the difference of two deterministic quadratures of the same smooth functional, so its D-dependence
should be ~|dD|/|D| times its size: defect ~1e-5 (guess, coarse (50,194) vs (99,590) for a MGGA), |dD|/|D| ~1e-3 after the coarse stage => surrogate
error ~1e-8 in the Fock, far below conv_tol_grad*0.1 = 3e-6. Analogous published structure: TeraChem's THC-HF/DFT uses "density-difference ansatz with a
full-precision reference iteration" (F = F_ref(D_ref) + [F_approx(D) - F_approx(D_ref)]) [V, arXiv 2508.19212, text read in the previous report; abstract page [S] here];
ORCA runs SCF on progressively finer COSX grids and uses the third grid for the final energy/gradients [V, ORCA 6.0 manual numint page] (but without
defect-corrected iterations). **Precision rules for the surrogate cycles:** (i) the coarse-grid XC and the K in the surrogate cycles must run at the *final* modulus count (11), not the coarse stage's 8:
at 8 moduli the decorrelating INT8 noise (~1e-6 relative) enters the surrogate, which is exactly the E2 failure mode; (ii) re-price: coarse XC at 11 moduli ~0.8 s instead of 0.65 s, so a surrogate cycle is ~2.4 s; (iii) the reported
energy comes from the true fine evaluation at the converged reference density (not from the surrogate); (iv) reset or rebuild the DIIS history whenever Delta is refreshed (the stored Fock vectors belong to the old surrogate).
Arithmetic (cost model of section 0): current full stage 6 x 5.2 = 31.2 s. New: 6 surrogate cycles x 2.4 s (coarse XC 0.8 + K 0.65 + coarse NLC 0.24 + 0.2 + ...
~1.9-2.4) = 11.5-14.5 s plus 2 fine evaluations x (XC 3.5 - 0.8 + NLC 0.8 - 0.24 = 3.3 s extra) = 6.6 s -> 18-21 s. Saving ~10-13 s (-21..-28 % of the SCF);
with E2's NLC already removed from the coarse stage, the extra over E2 is ~8-11 s. With 8 current full cycles the saving is ~14-17 s; with 4, ~6-8 s.
Sketch: new `scheme='crt'` option `dual_grid=True`; wrap `mf.get_veff` (dft/rks.py::get_veff) with an object that (a) evaluates coarse XC+NLC through
the existing `_coarse_grids` machinery, (b) adds the stored Delta, (c) refreshes Delta (fine `nr_rks` + `nr_nlc_vxc`) when the SCF callback reports
|g| < threshold or |D - D_ref|_F > 2e-3, (d) always refreshes before the convergence test. Energy: E_xc is likewise corrected by the (fine - coarse) energy
defect at D_ref plus the first-order term tr[Delta (D - D_ref)]. Files: `scf/mixed_precision.py` (run_fast_stage / final stage), `dft/rks.py::get_veff`.
Effort S-M. **Go/no-go experiment (30 min, T4, no new code beyond a script):** take D1, D2 from two consecutive cycles at |g| ~ 1e-3 and 1e-4; compute
Delta(D) = Vxc_fine(D) - Vxc_coarse(D) for both; require |Delta(D2) - Delta(D1)| / |Delta(D1)| < 0.05 (and the same for the NLC potential and the energy defects).

### E4. Occupied-projected Fock F C_o with DIIS (exact; est. -9..-15 s; rank 4 by effort, rank 2 by size)

Correction of a claim in `research_report.md` section 6: it says "DIIS cannot be used". That is wrong. The SCF fixed point only requires F_ov = 0.
The DIIS error vector FDS - SDF with D = C_o C_o^T needs only F C_o; F_oo = C_o^T (F C_o) comes for free; the nuclear gradient needs only D and the
energy-weighted density C_o eps_o C_o^T, i.e. F_oo. So build F C_o exactly, reconstruct a working Fock F~ = F_lag + [terms fixed by F C_o] (with F_lag the
last fully built Fock, refreshed every ~4 cycles or taken from the coarse stage), run DIIS on F~ and diagonalise F~. A stale F_vv only acts as a stale
preconditioner; at convergence F~ C_o = F C_o so the fixed point and the occupied eigenvalues are exact. Virtual eigenvalues need one final full build only if
requested (TDDFT etc.).
FLOPs for the XC part (per MGGA block, ng points, 4 AO components): rho: c_x = C_o^T ao_x, 4 GEMMs of 2 nao_sub nocc ng. Current vmat: K = 4 ng, output
nao_sub^2: 2 nao_sub^2 4 ng. Replacement V C_o: reuse c_x from rho; (V C_o) = ao_0 (sum_x w_x c_x) + sum_x ao_x (w_0 c_0 delta_x0-type terms + w_tau c_x): again 4 GEMMs of
2 nao_sub nocc ng with long K = ng (reconstruction cost amortised, unlike K1/XC-grad where K = 61 hurts [M: round-3 notes]).
Ratio vmat : rho : V C_o = nao_sub : nocc : nocc = 550 : 61 : 61 => 9.0 : 1 : 1 units; total XC GEMM work (1 + 9) -> (1 + 1) = 5x fewer FLOPs.
Applied to the measured share: vmat ~2.2-2.5 s of the 3.5 s full XC -> V C_o ~0.5 s; full XC 3.5 -> ~1.8 s
(AO evaluation 0.4, rho 0.3, V C_o 0.5, libxc/elementwise/launch 0.3-0.5). The residue planes of the AO are **not** shareable between rho and V C_o: rho contracts over mu (scale per grid point), V C_o contracts over g
(scale per AO row); sharing would need rank-1 scaling, which costs bits in the tails. So the AO is quantised a second time (this is already in the 0.5 s). The XC totals below use 1.8 s instead of 1.6 s. K part (occ-RI-K): K1 `Lij,njk->nikL` unchanged; K2 (nao x nocc*L x nao,
2 nao^2 nocc naux) replaced by (K C_o)_{mu j} = sum_{iL} B_{mu iL} G_{iL,j}, G = B contracted with C_o: cost 2 nao nocc^2 naux + 2 nao nocc^2 naux = 0.2 of K2 =>
K build 1 + 1 = 2 units -> 1 + 0.2 = 1.2 units = 0.6x. For wB97M-V both FR and LR tensors, so K 0.65 -> 0.4 s per cycle.
ACE connection [V, Lin, arXiv:1601.07159, JCTC 12, 2242 (2016), text read]: ACE builds W = K C_o once per outer iteration, applies K~ = -W (C_o^T W)^-1 W^T afterwards at
O(nao nocc) cost and "provides correct eigenvalues when self-consistency of the orbitals is reached"; oversampling nocc' > nocc speeds up the outer loop. In a dense-diagonalisation Gaussian
code there is one Fock build per cycle, so ACE alone gains nothing -- its benefit appears only when several *inner* cycles share one K build, which is exactly
the E3 surrogate loop (6 surrogate cycles x 0.65 s K = 3.9 s -> 2-3 outer K builds x 0.4 s ~1 s).
Arithmetic (stand-alone, no E3): per full cycle 5.2 -> 5.2 - 1.7 (XC) - 0.25 (K) = 3.25 s; 6 full cycles save 11.7 s; coarse cycles save ~0.3 s each (1.8 s); extra cycles
if the lagged F_vv slows DIIS: +0..2 full cycles = +0..6.5 s (at 3.25 s each) => net -7 .. -13.5 s. Together with E3 the two do **not** add: E3 already removes 4 of 6 fine evaluations; E4 then makes
each of the remaining 2 cost 1.8 instead of 3.5 s and shrinks the K and AO share of every surrogate cycle: combined SCF ~27-31 s (from 46.7), see section 4.
Caveat: `get_occ` / the aufbau occupation uses the eigenvalues of F~; a stale F_vv can reorder orbitals around the HOMO-LUMO gap in small-gap systems (diradicals, stretched bonds) -- keep occupations fixed by orbital
index after the first cycles, and do the final full build whenever the gap is < ~0.05 Eh.
Sketch: `dft/numint.py`: new `nr_rks_vc(ni, mol, grids, xc_code, mo_coeff, mo_occ)` returning (exc, V C_o) with a CRT path next to `_vmat_block_crt`
(extra moduli ladder as for rho); `df/df_jk.py::get_jk`: new `get_k_occ` returning K C_o (and the ACE factor W); `dft/rks.py::get_veff` returning an "occupied Fock" object;
`scf/hf.py` DIIS: error vector from F C_o, working Fock F~ = F_lag + ...; final full build for the reported orbitals only when `mf.mo_energy` virtuals are requested.
Risk: DIIS on a partially lagged F can stall near convergence if F_lag is too old (refresh rule needed); keep the final cycle a full Fock build as a safety check (3.5 s).
Effort M-L (2-3 weeks incl. tests).

### E5. DF-JK gradient restructuring (exact, ~2.4 s)

Facts from `df/grad/rhf.py::_jk_energy_per_atom` [V, code read]: for every operator the routine (1) re-runs the FP64 `int3c2e_evaluator` fill to get j3c, (2) contracts it
twice with C (nao^2 nocc naux each) to j3c_oo, (3) solves with the metric to dm_oo, (4) builds, per aux block, the nao x nao x dk tensor Gamma = C dm_oo C^T
(two FP64 contractions, K = nocc), (5) calls `sum_ejk_int3c2e_ip1` (Rys). `grad/rks.py::energy_ee` calls this routine **twice** (full-range with J, then the short-range
operator with factor hyb - alpha), and the SCF builds hyb*K_FR + (alpha - hyb)*K_LR [V, `df_jk.get_k`, mix_outside_kernel]. Two observations:
(a) **Possible correctness finding (not yet tested):** the SCF energy model is hyb*K_FR + (alpha-hyb)*K_LR, the gradient differentiates alpha*K_FR + (hyb-alpha)*K_SR. With per-operator DF metrics
K_SR^DF != K_FR^DF - K_LR^DF, so the gradient may not be the exact derivative of the DF energy the SCF minimised; the mismatch is of the size of the DF-fit error of the SR part, which could be ~1e-6 Eh/bohr, i.e. the gate itself.
The repo's "max gradient error 1.2e-7" is measured against a tight-convergence reference **from the same code**, which has the same mismatch, so it cannot reveal it. Decide with a finite-difference test of the SCF energy on a small
molecule (e.g. water dimer, wB97M-V/def2-TZVP, central differences of E with conv_tol 1e-12 vs the analytic gradient). If a mismatch shows, the fix is to differentiate FR + LR (`omega>0`; `int3c2e_evaluator`/`int2c2e_ip1_per_atom` take either sign),
but note the code comment "Prefer computing the SR part" suggests SR was chosen for screening efficiency (short-range integrals are more sparse), so an LR ip1 pass may be *slower* than the SR one; do not count a speed gain from this.
(b) j3c_oo is C^T (L^-T cderi) C: the SCF already holds cderi = L^-1 j3c for both operators (full and LR), so with FR+LR the Rys fill (~0.7-1 s per operator) could be replaced by a C-contraction of the cached tensor (~0.3 s per operator, one K1 pass).
This is only M effort, not S, because the SCF tensor lives in spherical, contracted AOs with its own aux ordering while the gradient evaluator works in the Cartesian, decontracted, aux-sorted layout (`aux_coeff`, `aux_sorting`, `pair_addresses`):
the cached tensor must be mapped through `ctr_coeff`/the pair index tables before the oo-transform. Memory: the SCF tensors (2 x 2.9 GB for BPA) must stay resident or be streamed from host (`dfobj.loop` already streams); fall back to the current path when they were freed.
Arithmetic: steps (1)-(2): 2 operators x (fill ~0.8 + 9.0e10/0.2e12 = 0.45) = 2.5 s -> 2 x 0.3 = 0.6 s; saving ~1.9 s. Step (4): 2 x (9.0e10 + 1e10) / 0.2e12 = 1.0 s FP64 (K = 61: CRT does not pay [M]); FP32-hi/lo
or FP16-split GEMM would be 4-6x faster but only at the 1e-6 gradient gate's edge -- not recommended; the sum of changes ~2.4 s of the 8 s (0.5 s of it is [G]-uncertain because the fill time per operator was not profiled separately).
Sketch: `df/grad/rhf.py`: new `j3c_oo_from_cderi(dfobj, C)` (with the AO/aux layout mapping above); `grad/rks.py::energy_ee`: second call with `omega=+omega`, `k_factor=alpha - hyb` only if the finite-difference test says it is needed.
Effort M. Test: finite-difference check first; then gradient identical to the old path (when FR+SR is kept) to 1e-10.

### E6. Symmetric tau-term GEMMs in the INT8 vmat (exact, ~2 s)

`numint._vmat_block_crt` (MGGA) forms one INT8 GEMM with K = 4 ng: components 0 (ao_0^T B_0 with B_0 = sum_x w_x ao_x, needs the full product) and 1-3 (ao_x^T (w_tau ao_x),
**symmetric**). Compute only upper-triangle output tiles for the three symmetric components: with b x b output tiles the work is (1 + 3 b(b+1)/(2b^2))/4 of the current: b=4 -> 0.72
(-28 % of the vmat GEMM), b=2 -> 0.81. Turing needs m,n,k % 16 == 0 and TN INT8; tile size 128 keeps the T4 INT8 GEMM efficient at nao_sub ~ 500.
Arithmetic: vmat GEMM+recon ~1.6 s of the 2.3 s per full eval x 0.28 = 0.45 s x 6 = 2.7 s [G]; quantisation unchanged => ~2-2.5 s. The same reformulation is why integrating by parts to merge the three
tau products into one (-grad.(w_tau grad phi)) is **not** possible: w_tau is only known at quadrature points, so its gradient would introduce quadrature error.
Sketch: `lib/crt_gemm.py` add `gemm_nt_syrk(a, b, moduli, tiles)`; call from `_vmat_block_crt` for the x=1..3 part. Effort M.

### E7. Increment-in-W assembly with moduli sized by |dW| (exact, ~2-3 s, overlaps E4)

Not error cancellation (that was E2): V = ao^T W ao is *exactly linear* in the pointwise weights W (= wv). Keep V0 = V(D0) from a full-precision evaluation and
add ao^T (W(D) - W(D0)) ao; dW is exact pointwise (FP64 libxc). The INT8 quantisation error is 2^-q |dW|_max, so the number of moduli can be reduced by
~log2(|W|/|dW|)/4 bits-per-modulus (4 bits per modulus [research_report.md sect. 0]): |dW|/|W| ~ 1e-2..1e-3 in late cycles => 10 bits => ~2.5 of 11 moduli; the accumulated error
sum_k 2^-q_k |dW_k| stays at the target if q_k is chosen from |dW_k|. GEMM time ~ n => ~-25 % of the vmat GEMM in the last ~4 cycles: 0.55 s x 4 = 2.2 s. For K it does not pay: a
factorised dD has rank up to 2 nocc (doubles the FLOPs) [G]. After E4 (V C_o replaces vmat) the gain shrinks to ~0.5 s. Effort S-M. Risk: dW at tail points spans decades; use max-norm scaling per row as now.

### E8. One AO evaluation per block and cycle (exact, ~1.5 s)

`nr_rks` runs two block loops (rho; then Vxc after one global libxc call) and `block_loop` evaluates the AOs in each [V, RESULTS_launch_overhead.md s.1]; gdft AO kernels are 4,060 of 5,989 launches
and ~0.3-0.4 s per `nr_rks` [M]. libxc is pointwise, so rho -> libxc -> B-build -> vmat can run in one pass per block (cost: libxc called on 16 k-point blocks, launch overhead the previous
branch tried to remove; use the batched call every 4-8 blocks). Saves one AO evaluation: ~0.2 s x 12 cycles ~ 2 s minus libxc launch cost ~0.5 s. Effort S. (Largest where AO evaluation is the
floor, i.e. after E4.)

---------------------------------------------------------------------------------------------------------

## 3. APPROXIMATE / OPT-IN items

### A1. cuEST-style single-tensor range-separated DF-K (opt-in; SCF -4.7 s, gradient -3.5 s)

What [V, NVIDIA cuEST "Density Fitting" page]: the fitting coefficients are always the full-range Coulomb fit; the range-separated operator is built in the *auxiliary space*:
s_FR (A|B) + s_LR (A|erf(wr)/r|B), "because the fit uses the full-range operator, the range-separation identity holds exactly, so short-range exchange can be set as s_SR = s_FR - s_LR". In our notation
with cderi B = L^-1 (P|mu nu) from the *full-range* DF build: K_LR ~ B^T G B with G = L^-1 L_erf L^-T (naux x naux, PSD, eigenvalues in [0,1]). For wB97M-V
(alpha = 1, hyb = 0.15) K = 0.15 K_FR + 0.85 K_LR = B^T M B with M = 0.15 I + 0.85 G, eigenvalues in [0.15, 1] => M = Z Z^T. Replace B by Z^T B (one naux x naux x nao_pair product,
2 x 1900^2 x 194 k = 1.4e12 FLOP once per geometry, ~0.2-0.3 s on INT8) and run **one** K build per cycle with the existing occupied-RI-K code. No LR 3c2e integrals, no second DF tensor.
Arithmetic: K compute 7.7 s -> 3.9 s; DF build 2.3 s -> 1.3 s [G] (one fill + one metric transform + Z product) => -4.7 s SCF; DF memory halves (2.98 GB [M] -> ~1.5 GB [G]) **only if J also comes from the
transformed tensor**: J = B'^T M^-1 (B' . D) is then an aux-space solve (essentially free; or keep the existing J-engine `df/j_engine_3c2e.py`, which needs no tensor at all). If the FR tensor is kept for J, you still hold B and Z^T B, and the 1.3 s / 1.5 GB figures do not hold.
Gradient: one `_jk_energy_per_atom` pass (full-range Rys ip1 with Gamma built from M-weighted dm_oo) + aux-aux terms with J^-1, L_erf derivatives (cheap, `int2c2e_ip1_per_atom` exists):
saves the second Rys fill+ip1 pass: ~3.5 s of the 8 s.
Relation to the existing knob: `range_separated_mode='mix_inside_kernel'` already halves K at dE 6.8e-7 Eh (BPA) [M]; A1 uses a different, 'fit-then-attenuate' model whose error I cannot predict:
K_LR from full-range fit coefficients is not "robust" (first-order in the fit residual delta through 2(delta|erf|rho~)), but erf/r is a low-pass kernel that kills most of delta's high-frequency content, and cuEST states DF-HF-like accuracy [V].
Opt-in until measured. Measurement: BPA fixed density: dE of (current mix_outside, mix_inside, A1) against a non-DF reference and against each other. Effort M (cderi transform, `get_jk` K path, gradient aux-aux terms).

### A2. SG-3 grid (opt-in; SCF ~-10 s, gradient ~-2.6 s)

PySCF's default `nwchem_prune` on (99,590) gives 1.06 M points for BPA (32 k/atom; unpruned would be 58 k/atom). **Check before relying on the ratio:** the SG-3/SG-2 totals below are raw atomic-grid counts, so compare them with
`len(mf.grids.weights)` *before* any weight/density screening (I could not run PySCF here; if the 1.06 M is already screened the ratio 0.54 is an underestimate of the benefit) -- one call settles it. Dasgupta & Herbert [V, JCC 38, 869 (2017), full text read] define SG-3 (99,590, DE2 radial
quadrature, per-element Lebedev partitions, Table 1: H 16,710; C 17,674; O 18,946 points) and SG-2 (75,302; H 7,094; C 7,790; O 8,574). BPA C15H16O2:
SG-3 = 15 x 17,674 + 16 x 16,710 + 2 x 18,946 = 570,362 points (0.54 x of 1.06 M); SG-2 = 15 x 7,790 + 16 x 7,094 + 2 x 8,574 = 247,502 (0.23 x). Reported: pruning speedups "approaching a factor
of two for hybrids at ~100 atoms", pruning errors MUD 0.01-0.04 kcal/mol (SG-2) and ~0.01-0.03 (SG-3) incl. wB97M-V/aTZ [V text]; they recommend SG-2 as the minimum for modern meta-GGAs, SG-3 for high quality. ORCA's default grid error
is 0.01 +- 0.03 kcal/mol [V ORCA 6.0 manual]. 0.01-0.03 kcal/mol = 1.6e-5 - 5e-5 Eh: **fails the 1e-8 gate against the PySCF-default-grid reference** -> opt-in, and the user must re-define the reference.
Arithmetic (all grid-linear parts scale with points): full XC eval 3.5 -> 1.9 s (-1.6 x 6 = -9.6 s; coarse stage unchanged); XC gradient 5.6 -> 3.0 s. Also NLC grid: nothing changes (separate grid).
Sketch: `dft/gen_grid.py` accept `radi_method=de2`, `prune=sg3_prune` with the Table-1 partitions (copy from the paper); note ORCA's adaptive, basis-set-driven pruning (position-operator expectation values) [V manual] as the alternative design. Effort S-M.
Stacks with E3 (E3 reduces the number of fine evaluations, A2 shrinks each one).

### A3. Dynamic-precision FP32 Rys for 3c2e (opt-in, controllable; ~-3.3 s)

Source: Luehr, Ufimtsev, Martinez, JCTC 7, 949 (2011) [S, abstract only]: choose single vs double per integral by size, lower the cutoff during the SCF; TeraChem's current default [V, arXiv 2410.22278 text]: integrals with a
density-weighted Schwarz bound above 1e-5 in FP64, the rest in FP32, "roughly half the cost of pure double precision" on an A100 (FP32:FP64 = 2:1 there). On T4 FP32:FP64 = 32:1, so if a fraction f of the work is eligible
time ~ (1 - f) + f/20 (20 = realistic Rys efficiency ratio, not 32): f = 0.7 -> 0.34x, f = 0.9 -> 0.15x. The Rys kernels `lib/gvhf-rys/ejk_int3c2e_ip1.cu` (3.3 s) and the 3c2e fill (2 x 0.7-1 s in the DF build, same in the gradient):
-> 3.3 -> ~0.8, fills 3 -> ~0.8: ~ -4.2 s of 65 s on T4 (zero on A100/H100). Accuracy: FP32 relative 6e-8 on terms < 1e-5 => random-walk error ~ 6e-13 x sqrt(1e8) = 6e-9 -- well inside 1e-6 Eh/bohr; **but** the DF *tensor* enters K with all pairs, so use the
bound on |(mu nu|P)| x (typical |c|) rather than a density weight in the SCF fill and test dE at 1e-8.
Why INT8/FP16/CRT does not apply to integrals (question 6): the Rys/MD kernels are element-wise recurrences + root/weight tables + Boys-function evaluation with wide dynamic range; the only GEMM-shaped step is the Hermite-to-Cartesian
transform of McMurchie-Davidson, which gpu4pyscf's Rys path does not use, and integrals are ~5 % (SCF) to ~20 % (gradient) of the time. Huang et al. state that no prior GPU ERI work used matrix engines [S snippet of arXiv 2601.08077]. No FLOP-saving case at 1e-12 relative accuracy; skip.
Effort M-L (kernel templating, threshold plumbing).

---------------------------------------------------------------------------------------------------------

## 4. Combined estimate and ordering (T4, BPA, assumptions of section 0)

| step | SCF (s) | gradient (s) | total |
|---|---|---|---|
| baseline | 46.7 (48.7 measured) | 16.8 | 63.5 (65.5) |
| + E1 (NLC grad) + E5 (DF grad) | | 16.8 - 2.5 - 2.4 = 11.9 | |
| + E2 (coarse NLC) | -3.4 -> 43.3 | | |
| + E6 (SYRK tau), E8 (one AO pass) | -2.2 - 1.5 -> 39.6 | | 51.5 |
| + E3 (dual grid, incl. NLC defect) | -8 -> ~31.5 | | 43.4 |
| + E4 (F C_o, ACE-style K reuse) on top of E3 | -5 .. -9 -> 23-27 (stand-alone E4: -7..-13.5) | | 35-39 |
Exact set: **~35-42 s (1.6-1.8x)**; E7 only if E4 is not built. Opt-in on top: A1 (-4.7 SCF -3.5 grad), A2 (-6 SCF with E3 in place -2.6 grad), A3 (-3 grad/DF): ~ -15 s => ~22-27 s [G, rough].
The estimates are not additive in general (E3/E4/E7 overlap, E8 shrinks after E4) and rest on the assumed 6+6 cycle split (+-30 % on E3/E4).
Priority order by gain/effort: E1 -> E2 -> E5 -> E3 (after the 30-min validation) -> E6/E8 -> E4 -> A1 (after a 1-hour accuracy test) -> A2 -> A3.

Experiments that decide everything (all fixed-density, T4 only):
1. Delta(D1) vs Delta(D2) for XC and NLC (E3 go/no-go, ratio < 0.05).
2. Spectrum of the LR metric j2c_LR (eigenvalues vs 1e-9 `LINEAR_DEP_THR`); if a large fraction lies below ~1e-6 and the K_LR error from dropping them is < 1e-9, an adaptive naux truncation of the LR tensor saves ~20-40 % of K_LR [G; error argument: dropped modes obey (v|rho)^2/w <= (rho|rho), so it is NOT guaranteed small -- measure].
3. dE of A1 against mix_outside/mix_inside/non-DF.
4. |dW|_max/|W|_max per SCF cycle (E7).

---------------------------------------------------------------------------------------------------------

## 4b. Other GPUs and 50-100 atom molecules (what changes in the ranking)

| item | T4 | A100 / H100 (FP64 TC ~ FP32 SIMT, INT8 not needed) | note |
|---|---|---|---|
| E1 NLC-grad contract-first | yes | yes (FP64 FLOPs x0.11 as well) | exact on every GPU |
| E2 coarse NLC grid | yes | **more**: NLC is 28-39 % of the A100 SCF [M] | |
| E3 dual grid | yes | yes (fine XC evaluations are 45 % of the A100 SCF) | architecture independent |
| E4 F C_o | yes (INT8 GEMM FLOPs) | yes (FP64 FLOPs x0.2 on the XC GEMM share; but A100 XC is launch/AO-bound, gain smaller) | |
| E5 DF gradient | yes | smaller (A100 gradient is 2.4 s total, not GEMM-bound [M]) | |
| E6 SYRK tau | yes | only if INT8/FP64 GEMM-bound; ~0 | |
| A1 single-tensor RS K | yes | yes: halves K FLOPs, DF memory and the second Rys pass on any GPU | |
| A2 SG-3 | yes | yes (all grid-linear) | |
| A3 FP32 Rys | **T4-class only** (FP32:FP64 = 32:1); ~2x at best on A100 | | TeraChem: ~half cost on A100 [V] |

Scaling to 50-100 atoms (nao ~ 950-1900, N_xc ~ 1.6-3.2 M, N_nlc ~ 0.33-0.65 M for 50-100 atoms; arithmetic scales linearly from BPA's 33 atoms / 217 k):
* NLC pair work ~ N^2: 100 atoms => (0.65 M / 0.217 M)^2 = 9x the BPA pair sum (FP32 sym ~ 0.35 s -> 3.2 s/call on T4, ~40 s over 12 cycles), while XC grows only ~3x => **NLC becomes a leading cost on T4**: E2, the NLC part of E3 and the lagged-NLC idea of report 1
  move up; far-field tile skipping (RESULTS_vv10: 100-atom clusters 37-74 bohr across keep only 55-90 % of the pairs at Rc = 25-40 bohr) starts to pay but is capped at ~2x.
* K ~ nao^2 nocc naux ~ N^4 with sparsity-free dense GEMMs vs XC ~ N^2 (AO-block sparsity) => K overtakes XC at roughly 100+ atoms: A1 (halves K), E4's K C_o (0.6x) and ACE-style reuse (E3/E4 surrogate cycles) move up; memory for the DF tensor (7.4 GB at 57 atoms, tamoxifen [M]) is the limiting factor, so A1's tensor halving matters.
* AO sparsity improves, nao_sub/nao falls to ~0.3-0.4 at 100 atoms (see section 5), so E4's ratio nao_sub : nocc shrinks from 9:1 to ~3.6:1; E4 still removes ~(1+3.6) -> (1+1) = 2.3x of the XC GEMM FLOPs but the AO evaluation (O(N) with fixed cost per point) becomes the floor: E8 and an FP32/float-float AO evaluation (cuEST uses float-float for VV10 [V]; the gdft AO kernels are FP64 on a 1/32-rate GPU) move up.

---------------------------------------------------------------------------------------------------------

## 5. Checked and rejected / not applicable (with reasons)

| idea | verdict | evidence |
|---|---|---|
| Auxiliary density expansion (ADE / DF-JX: rho, XC matrix from the fitted aux density; cost nao^2 naux GEMV instead of nao^2 ng) | **Not applicable to wB97M-V.** tau cannot be formed from the fitted density; Bienvenu & Knizia need the Laplacian (upsilon-form) and say accurate tau-form mGGAs would need reparametrising -- i.e. a different functional. Errors are kcal/mol-class even for PBE-type tests; 4-5x speedup on CPU. | [V] arXiv:1710.10049 (text read). Arithmetic: ADE rho costs ng x naux_local ~ 1e9 MAC vs ng x nao x nocc = 3.6e10 (30x), but unusable |
| XC potential projected onto the aux basis (Koster AFD / Dunlap robust fit) to avoid the nao^2 ng GEMM | Fit error enters the Fock matrix at first order unless the fully robust Dunlap-Palenik variational fit is used (then second order); gradients need the matching derivative scheme. Needs naux-scale 3c overlap tensor (3 GB). Research project; E4 gives the same 5x without any new approximation. | [V] arXiv:1604.00218 (Dunlap & Palenik, text read: robust energy has no first-order fitting error); Koster et al. 2004 JCP [S] |
| ISDF/THC for XC (interpolation vectors zeta_P(g)) | zeta is ng x N_ISDF (1.06 M x ~7-12 k = 7-13 G elements, only usable if made sparse), setup ~ 2 ng nao N_ISDF = 8e12 FLOP (= one full XC evaluation), approximation + no variational gradient. No Gaussian-basis XC-ISDF paper found. | [S] search found ISDF only for exchange/ERIs (1911.00470 [V in prev. report], 2510.20826 [S]) |
| THC-K / COSX / sn-LinK / ORCA RIJCOSX | At nao 623: occ-RI-K = 2 x 2 naux nao^2 nocc = 1e11 FLOP/cycle already ~0.65 s on INT8. COSX is not tensor-core friendly (analytic potential integrals), ORCA's own error 0.05+-0.1 kcal/mol (8e-5 Eh). | [V] ORCA 6.1 RI manual; [V] 2508.19212 (2-4x vs direct, not vs DF) |
| ADMM (auxiliary density matrix, small-basis K + GGA-exchange correction) | K is only 20 % of the SCF; ADMM errors are 1e-4..1e-3 Eh class [G]; for omega-B97M-V it would need a long-range-attenuated GGA exchange correction that is uncalibrated for this functional [G]. | [S] Guidon, Hutter, VandeVondele JCTC 2010 abstract |
| Localized-orbital / PNO / sparse-map / Cholesky exchange, adaptive naux | No gain at 33-100 atoms: nocc/nao = 0.1 already, and sparsity formats do not map onto dense INT8 GEMMs. LR-tensor spectrum test (section 4, item 2) is the only cheap variant. | [G] |
| Incremental / difference-density K with low precision | K[dD] needs a factorisation of dD (rank up to 2 nocc) => 2x FLOPs; the occ-RI-K advantage is lost. Only the vmat (linear in W) admits increments (E7). | [G arithmetic]; Huang et al. do not use incremental builds [V, 2601.08077] |
| Second-order / Newton-CG / ARH / GDM / trust-region SCF | Inference [G]: no net saving of Fock-equivalent builds for closed-shell organics (each Newton macro-step costs several Fock-equivalents). Context: DIIS cycle counts in a GPU code are 9-13 (alkanes, graphene 9-10, water clusters 10-12, tamoxifen 11, sphingomyelin 13) for **6-311G(d,p) DF HF/DFT** (Table 4 does not test Newton methods). Convergence speed is dominated by the extrapolation scheme, not the guess. | [V] Huang et al. Table 4 (arXiv:2601.08077 text; counts only); [V] Lehtola, arXiv:1810.11659 (accelerator dominates convergence; SAP best on average) |
| Better initial guess: SAP/Hueckel/basis-set projection | Already measured negative: SVP-seed + projection saves 2 coarse + 1 full cycle but the whole pipeline is slower (50.7 s vs 46.4 s). SAP gain is at most ~1 cycle (Lehtola reports guess *quality*, not iteration counts). | [M] RESULTS "Negative: small-basis seed"; [V] Lehtola |
| Dual-basis SCF (Liang & Head-Gordon) | Perturbative one-step correction in the large basis; it replaces rather than accelerates the final SCF, so it cannot meet a 1e-8 Eh gate (Q-Chem manual quotes errors "an order of magnitude below the small basis alone", not 1e-8) | [S] abstract + Q-Chem manual |
| Density-based grid-point screening / grid pruning | Already in the code: PySCF drops zero-weight points and `Grids.get_non0ao_idx` / `block_loop` mask AOs per 4096-point block (nao_sub ~ 500 of 623 for BPA => vmat FLOPs x0.64). Further exact gain only from smaller, spatially compact blocks (Morton order): nao_sub falls with block size, GEMM efficiency drops below ~1k points. Radial/angular pruning is the opt-in A2. At 100 atoms (nao ~ 1900) nao_sub/nao falls to ~0.3-0.4 (AO radial extent ~8-10 bohr vs 40 bohr molecule), so the vmat share shrinks ~3-5x relative to the dense estimate and E4's 9:1 ratio becomes nao_sub:nocc ~ 650:180 = 3.6:1. | [G]; code read (`numint.nr_rks`, `gen_grid.get_non0ao_idx`) |
| On-the-fly sparse AO-pair evaluation | The block-sparse AO tables of `block_loop` are the dense-GEMM compatible form of this; element-level sparse pair lists (rho from D with a pair list) lose the tensor cores and cost nao_sub^2 ng anyway. Not recommended; ADE (above) is the only route that changes the scaling and it fails for tau. | [G] |
| Hierarchical / FMM / H-matrix VV10, rVV10 interpolation | Already analysed in RESULTS_vv10.md (far-field low rank exists but cap ~2x at N = 0.3-0.6 M; 42 % of pairs within 12 bohr in a 37-bohr cluster; BPA is ~12 bohr across so ~0 % far field). rVV10/Roman-Perez-Soler needs a uniform grid; Laikov is a different functional. | [M] RESULTS_vv10.md; [S] Sabatini et al. PRB 87, 041108 abstract; [V] Laikov 2511.04169 (previous report) |
| Richardson extrapolation of the NLC / XC grid | Unsound (quadrature error is not smooth in n); use the defect correction (E3) | [G] |
| Z-vector-free gradient | A closed-shell SCF gradient has no Z-vector; the only exact levers are E1/E5/FR+LR consistency. |

---------------------------------------------------------------------------------------------------------

## 6. What other GPU hybrid-DFT groups do (2025-26) -- verified status

* **NVIDIA cuEST** [V, docs.nvidia.com/cuda/cuest, emulation + density-fitting + index pages read]: DF J/K from the AO density, K via Ozaki-I/II INT8 emulation by default ("early iterations lower slices/moduli, later higher; 5 slices / 8 moduli reached |g| = 1e-7 on a 110-atom HF/def2-TZVP/def2-universal-jkfit example"; fewer than 6 moduli cannot support a tight SCF);
  VV10 potential/gradient via a float-float type ("~47 bits", only if FP32:FP64 >= 32 -- i.e. the T4); LibXC 7.0.0 for functionals; grids user-supplied (pruned or product); **range-separated DF-K from one full-range fit** (A1). No SCF solver, no XC tensor-core path, no incremental builds. No performance numbers in the docs.
* **Huang, Shao, Hammond, arXiv:2601.08077** [V]: only K emulated (INT8, adaptive by relative dE), J and XC stay FP64; SCF cycles 9-13; speedups 3.0x (RTX 4090), 4.6x (RTX 6000 Ada), 1.37x (H100); they explicitly note XC becomes the next bottleneck -- our project already covers that.
* **TeraChem** [V, 2410.22278 + prior-report read of 2508.19212]: mixed FP32/FP64 ERIs with a 1e-5 density-weighted Schwarz threshold (A3); GPU THC-HF/DFT with a density-difference ansatz and a conventional double-precision start until dE < 0.1 Eh (K 2-4x faster than integral-direct, not than DF).
* **ORCA 6.x** [V, manual]: RIJCOSX default for hybrids (error 0.05+-0.1 kcal/mol), RI-JK slower for big molecules, three progressively finer COSX grids in the SCF with the third used for the final energy/gradients, adaptive basis-set-driven grid pruning (+10-20 % points for better accuracy), default XC grid error 0.01+-0.03 kcal/mol. No GPU statement on those pages.
* **QUICK 25.03** [S, GitHub README snippet]: SG-1 grid only, no meta-GGA, no range-separated hybrids -- not a comparison point for wB97M-V. **Q-Chem/BrianQC** [S, vendor page]: GPU J/K/XC and "mixed precision with double-precision accuracy", up to 6x SCF; no algorithmic detail. **GauXC** [S]: library for XC and exact exchange integration with GPU back-ends (Williams-Young et al., arXiv:2007.03143); no low-precision XC reported.
* Nothing found (negative result) in any of these for: coarse/fine defect-corrected grids, occupied-projected XC (V C_o) with DIIS, ML-compressed XC grids (searched 2023-26), VV10 beyond brute force / Laikov.

---------------------------------------------------------------------------------------------------------

## 7. Source list

[V] read in this session (PDF/HTML/doc text): Lin, "Adaptively Compressed Exchange Operator", arXiv:1601.07159; Dasgupta & Herbert, JCC 38, 869 (2017) (https://www.asc.ohio-state.edu/herbert.44/reprints/JCC_38_869.pdf);
Bienvenu & Knizia, arXiv:1710.10049; Dunlap & Palenik, arXiv:1604.00218 (abstract/intro/XC passages); Wang, Hait, Unzueta, Zhang, Martinez, arXiv:2410.22278 (precision section); Huang, Shao, Hammond, arXiv:2601.08077 (https://arxiv.org/html/2601.08077v3);
Lehtola, arXiv:1810.11659 (abstract, discussion of convergence); NVIDIA cuEST docs (index, emulation, density_fitting); ORCA 6.0 numint page and ORCA 6.1 RI page; repo files: `grad/rks.py`, `grad/rks.py::get_nlc_exc`, `df/grad/rhf.py`, `df/df_jk.py`, `dft/numint.py`, `scf/mixed_precision.py`, `dft/gen_grid.py`, `benchmarks/dft/*.md`.
[S] snippet/abstract only: Luehr-Ufimtsev-Martinez JCTC 7, 949 (2011); Hillers-Bendtsen/Martinez arXiv:2508.19212 (abstract page; text read in report 1); Liang & Head-Gordon dual-basis (Q-Chem manual); Guidon-Hutter-VandeVondele ADMM; Koster et al. AFD (JCP 2004); Laqua-Kussmann-Ochsenfeld JCP 149, 204111 (2018) (grid partitioning, not used for numbers);
QUICK README; BrianQC page; GauXC arXiv:2007.03143; Sabatini et al. rVV10; arXiv:2510.20826.
[G] everything marked as estimate; the cycle split (6+6) and per-cycle costs of section 0 are assumptions to be replaced by `mf.cycles_coarse` / timer output from one instrumented run.
