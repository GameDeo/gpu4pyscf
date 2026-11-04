# Implicit solvation (PCM family, SMD) in GPU4PySCF: audit, cost model, optimizations

Branch `worktree-agent-ae1020a66ad96d860` (based on `claude/serene-meitner-kinyv4` @ 1ef4f03).
**Nothing here was run on a GPU** (no cupy on the writing machine, Colab reserved): everything marked "model" is a static
estimate, everything marked "CPU" was measured with PySCF/NumPy, everything marked "untested on GPU" must be run by the lead
(`gpu4pyscf/solvent/tests/test_pcm_cache.py`, `test_pcm_mixed_precision.py`, `benchmarks/dft/bench_solvent.py`).

## 1. What the code does

| piece | file | notes |
|---|---|---|
| hook into SCF | `solvent/_attach_solvent.py` | `SCFWithSolvent` mixin placed in front of the (DF-)RKS/UKS class. `get_veff` = base veff, then (unless `frozen`) `with_solvent.kernel(dm)`; the result is attached to the veff as `e_solvent`, `v_solvent`. `get_fock` adds `v_solvent` **before** DIIS; `energy_elec` adds `e_solvent` (+ SMD `e_cds`). `density_fit()` after `.PCM()` raises (the DF class would shadow `get_veff`), so the order DF -> PCM is enforced. |
| surface | `pcm.gen_surface` | Lebedev sphere per atom (302 pts at `lebedev_order=29`), SWIG/ISWIG switching; only 0.29-0.37 of the sphere points survive (nsurf ~ 100-110 per atom for PCM, ~170-195 for SMD with 590). |
| dense matrices | `PCM.build` | CUDA `pcm_d_s` fills S, D ([n,n], FP64); IEF-PCM / SS(V)PE: `K = S - f/(2 pi) (D*A) S` (one n^3 DGEMM), `lu_factor` (2/3 n^3). Kept: `K_LU`, `D`, `S`, `A` (IEF) or `K_LU` (C-PCM/COSMO). `lowmem_intermediate_storage=True`: nothing O(n^2) stored, matvec kernels + MINRES/GMRES. Also the nuclear potential at the surface (CPU `int2c2e` cross, once). SMD has its own dense `build` (IEF-PCM, 590 pts/atom, **no lowmem path**) plus the CDS term (Fortran, CPU). |
| per SCF cycle | `PCM._get_vind` | (1) `v_g = v_nuc - int1e_grids(dm)`: `get_int3c1e_density_contracted` = CPU-side Hermite-density preprocessing (`GINTinit_J_density_rys_preprocess`, D2H + H2D, dense O(N^3) cart2sph GEMMs) + FP64 Rys kernel over (primitive pair x surface point); (2) `q = K^-1 R v` and `q^T`-solve (two `lu_solve`, memory-bound), `q_sym = (q+qt)/2`; (3) `V = -int1e_grids(q)`: `get_int3c1e_charge_contracted`, FP64 kernel over the same (pair, point) space; (4) `E = 1/2 v.q_sym`. Cost is `nsurf x prim-pairs` per pass, i.e. **O(natm^2)** FP64 work, no screening on the point side. |
| gradient | `solvent/grad/pcm.py` | `grad_qv` (`int1e_grids_ip1/ip2`, derivative point-charge integrals, own `VHFOpt(aosym=False)`), `grad_solver` (dense `dD, dS` = [3,n,n], dF/dA per atom, ~20 memory-bound passes over [3,n,n]), `grad_nuc` (CPU `int2c2e_ip1`). |
| Hessian / TDDFT | `solvent/hessian/pcm.py`, `solvent/tdscf/pcm.py`, `tdscf/rhf.py` | same building blocks; TDDFT non-equilibrium rebuilds the solver with `eps_optical`. Not performance-critical for this project. |

## 2. FLOP / time model (CPU-evaluated counting, GPU numbers are a model)

`benchmarks/dft/solvent_cost_model.py` replicates `gen_surface`'s switching (NumPy) and counts segmented primitive shell pairs
(|S_ij| > 1e-13 as proxy for the 1e-14 q_cond screening) of the real `benchmarks/molecules/organic` structures, def2-TZVP.
Per (primitive pair, surface point) work = 40 + NR (40 + 9(L+1)) + n_herm (2 NR + 2) FP64 instructions (NR = L//2+1 Rys roots;
charge-contracted pass: + AO-component formation), kernels assumed at 35 % of the non-tensor-core FP64 peak (T4 0.25 TFLOP/s,
A100 9.7, H100 34) and cuBLAS/cuSOLVER at 75 %.  **Uncertainty ~2x absolute; the scaling is what matters. The lead's first
`bench_solvent.py` run calibrates `KERNEL_EFF`.**

| molecule (def2-TZVP, IEF-PCM, 302/atom) | natm | nao | nsurf | prim pairs | GFLOP/cycle (dm + q pass) | T4 s/cycle | A100 s/cycle | build (DA S GEMM + LU) T4 | resident / peak GB |
|---|---|---|---|---|---|---|---|---|---|
| Bisphenol A | 33 | 623 | 3 695 | 58 k | 37 + 50 | **1.0** | 0.03 | 0.7 s | 0.44 / 0.66 |
| Tamoxifen | 57 | 1 042 | 6 164 | 122 k | 129 + 172 | **3.4** | 0.09 | 3.3 s | 1.2 / 1.8 |
| Taxol | 113 | 2 228 | 10 963 | 427 k | 806 + 1 077 | **21.6** | 0.56 | 18.7 s | 3.9 / 5.8 |

SMD (590 pts/atom, same IEF machinery): BPA nsurf 6 462 (1.8 s/cycle T4, 2.0 GB peak), Tamoxifen 10 596 (5.9 s, build 17 s T4, 5.4 GB peak),
Taxol 19 057 (37.5 s, build 98 s T4, **17.4 GB peak, does not fit a 16 GB T4**, resident 11.6 GB).

Amdahl: the crt pipeline made everything else ~5x faster while these kernels stay FP64.  Against the measured crt SCF on
T4 (48.7 s for BPA, roughly 8-9 Fock builds, ~5.5 s/cycle) the modelled exact solvent adds **~1.0 s/cycle = +18 %** at 33 atoms; the ratio
grows ~linearly with natm (solvent ~ natm^2, rest ~ natm^1.x-2): at Tamoxifen size ~3.4 s on a ~10 s cycle, at Taxol size
the solvent is comparable to the whole rest of the cycle.  On A100/H100 the solvent is 1-3 % of a cycle (there the Python/launch
overhead in `int1e_grids`, the D2H/H2D of the Hermite density and the dense cart2sph GEMMs are comparable to the kernels).
Dense linear algebra is never the problem for PCM: LU/GEMM are one-time per geometry (<= 20 s on T4 at 113 atoms), the two
solves per cycle read 2 x n^2 x 8 B (0.1-1 GB, ms).  For SMD the build time and the O(n^2) memory are the limits.

Gradient (IEF-PCM): the three consumers each called `_get_vind` again (cache never hit, see 4.1) = 3 x (dm pass + q pass + 2 solves)
= ~3 s (BPA), ~10 s (Tamoxifen), ~65 s (Taxol) on T4 (model).  After the fix the saving is the full amount when the gradient density equals the last
Fock density (plain FP64 / classic SCF); when the `_accept_converged_init` shortcut of the crt scheme fires, the gradient still pays one `_get_vind(with_vmat=False)`
(dm pass + solves, no Fock pass), i.e. the saving is ~2.6 s / ~8.6 s / ~54 s (about 1 - 0.43 of the removed cost).  What remains: `int1e_grids_ip1/ip2` (derivative
integrals, ~2-3x an energy pass each) and ~0.5 s of memory-bound dense passes in `grad_solver`.

## 3. Ranked optimization list

| # | item | exact? | expected gain (model) | risk | status |
|---|---|---|---|---|---|
| 1 | Write `_intermediates['dm']`; gradient/Hessian reuse the SCF response (`_ensure_vind`, `_get_vind(with_vmat=False)`) | exact | gradient -3 .. -65 s on T4 (-2.6 .. -54 s under the crt shortcut); A100 -0.1 .. -2 s | low | **implemented**, untested on GPU |
| 2 | C-PCM/COSMO single solve; `left_multiply_R` without `D*A`; lowmem C-PCM double build | exact (1e-14) | ms .. 10 ms/cycle, saves an n^2 temporary twice/cycle | low | **implemented** (algebra CPU-checked), untested on GPU |
| 3 | SMD CDS element bug for ECP atoms; SMD gradient ignored `dm=` | bug fix | correctness (iodine/def2-ECP gets Mn CDS parameters) | low | **implemented** |
| 4 | DF-projected solvent in the coarse stage (+ one exact defect-correction pass): `v_g = Wt^T (B D) + dv`, `V = -B^T (Wt q) + dV`, with `B` the DF tensor of the Fock build and `Wt = L^-1 (P|g)` static (naux x nsurf, 90 MB BPA) | final result exact (final stage exact) | removes ~7 of ~9 exact solvent evaluations: T4 -7 s (BPA, 14 %), -24 s (Tamoxifen), -150 s (Taxol), model, before a possible +1 final cycle (CPU glycine: exact \|g\| at hand-over 6.5e-6 vs final threshold 3e-6); the J-back-contraction rides the existing cderi pass, so the per-cycle solvent cost drops to O(naux x nsurf) GEMVs | medium-high: needs the aux ordering/`aux_coef`/`cd_low` conventions of `df.py::_decompose_j2c`, GPU iteration | **designed + validated on CPU, not implemented** (section 6) |
| 5 | Coarse solvent surface in the coarse stage (`mixed_precision['coarse_solvent_lebedev']`, 17 -> 110 pts, 23 -> 194) | final result exact | solvent cost x0.36 (x0.64) in the ~7 coarse cycles: -4.5 s BPA, -15 s Tamoxifen, -100 s Taxol on T4 (model), **but the final stage needs +1 cycle (CPU, measured: water 4 -> 5, glycine 5 -> 6 cycles to \|g\|<3e-6)** that costs a full cycle (~5 s BPA): net ~0 at BPA size, a gain only from ~60 atoms on T4 | low | **implemented, opt-in (default off)**, untested on GPU |
| 6 | Far-field shortcut in the FP64 kernels (erf(x)=1 for T > 36, asymptotic Boys), skip dead pairs per surface point | exact to 1e-15 | unknown (diffuse pairs have small T); only matters on T4 | medium (CUDA) | not done |
| 7 | Cache `intopt.cart2sph` and replace the dense O(N^3) `C @ D @ C^T` by per-block small GEMMs; GPU Hermite preprocessing instead of CPU+D2H | exact | 20-200 ms/cycle (matters on A100, N=1000-2200) | low | not done |
| 8 | Do not keep `S` resident for IEF/SS(V)PE (only the dense gradient needs it; recompute with `get_D_S(with_D=False)`) | exact | -25 % solvent memory (1.2 -> 0.9 GB Tamoxifen; 11.6 -> 8.7 GB SMD Taxol) | low-medium (touches grad/hessian/tdscf) | not done |
| 9 | SMD `lowmem` path / coarser default SASA surface | - | SMD currently OOMs above ~100 atoms on 16 GB | medium | not done |
| 10 | CRT for the build GEMM `D*A*S` | ~1e-9 | one-time 2-17 s on T4 (Tamoxifen/Taxol)... only if S is split into O(1) off-diagonal part + the 1e-10-scale diagonal (per-column power-of-two scaling is destroyed otherwise) | medium | not recommended |
| 11 | Lag/freeze `q` in the coarse stage | no | subsumed by 4/5; inconsistent Fock/DIIS | - | rejected |
| 12 | Cache the 3-index tensor `(mu nu|g)` | exact | nsurf x nao^2/2 x 8 B = 11.5 GB (BPA), 53 GB (Tamoxifen) | - | infeasible; item 4 is its low-rank form |

Why not FP32 / INT8 for the solvent: the kernels are Rys/Boys evaluations (transcendental, not GEMMs) and the dense solves are
already memory-bound; the only GEMM-like part is the DF contraction of item 4, which is a GEMV (memory-bound, FP64 is fine).

## 4. Compatibility audit

### 4.1 Bugs found and fixed
1. **Response cache never hit** (`grad/pcm.py`, `hessian/pcm.py`): all 3 gradient (4 Hessian) consumers test
   `pcmobj._intermediates.get('dm')`, but the GPU port never stored it (PySCF CPU does, `pyscf/solvent/pcm.py:495`).
   Every gradient therefore repeated the full `_get_vind` three times, including the Fock-matrix integral pass.  Now `_get_vind` stores the spin-summed
   density (a plain copy; not stored when `frozen_dm0_for_finite_difference_without_response` is set, because then `q` does not belong to `dm`)
   and `PCM._ensure_vind(dm)` re-solves only on a miss (norm > 1e-10, other shape, frozen right density).  Consumers that pass other densities (TDDFT gradients call
   `grad_qv/grad_nuc/grad_solver` with several different `dm`) simply miss and recompute exactly as before.
2. **SMD CDS with ECP atoms**: `get_cds_legacy` passed `mol.atom_charges()` (effective charge Z - ncore: 25 for an iodine with a
   def2 ECP) to the Fortran CDS code as atomic numbers.  Now `_atomic_numbers(mol)` (from the element symbols).  The nuclear-potential uses of
   `atom_charges()` (PCM/SMD `build`, `grad_nuc`, Hessian) are correct as they are (effective core charges).  `smd_experiment.py` still has the same pattern (not used).
3. **SMD gradient ignored `dm=`** (`dm = kwargs.pop('dm')` after `dm` was already a named parameter).
4. Lowmem C-PCM built the surface twice on the first call (`any(cupy.ndarray in _intermediates)` is False until `q` exists); replaced by `_is_built()`.

### 4.2 Interplay with `mf.mixed_precision` (read-through, no run possible)
* Class hierarchy: `SCFWithSolvent` (mixin) -> `_DFHF` -> KS class.  `scf()`/`_kernel` call `mf.get_veff(mol, dm, dm_last, vhf)`; the mixin accepts and forwards these.
  `mixed_precision` is read through `getattr(mf, ...)`; it is copied by `SCFWithSolvent.__init__` (`__dict__.update`) if set before `.PCM()` and is a plain attribute otherwise.
* The solvent is **independent of the DFT grids and of the XC/K/CDERI precision switches** (it only uses `int1e_grids` + FP64 LU), so the coarse-grid swap in `_coarse_grids`
  (`mf.grids`/`mf.nlcgrids` replaced and restored in `finally`) cannot corrupt it, and the same solvent energy function is used in both stages.
* State carried between stages: `with_solvent.e/v/_intermediates` of the coarse stage are stale but the final stage's first `get_veff` (the `init` Fock build in `_kernel`) always recomputes
  them (`frozen=False`), so the `_accept_converged_init` shortcut certifies the handed-over density with a full exact solvent evaluation.  After the shortcut `mf.make_rdm1()` (from the freshly diagonalised
  Fock) differs from the density the solvent saw by O(|g|); the gradient therefore misses the cache once (one `_get_vind(with_vmat=False)`), which is the correct behaviour.
* Cost structure: each stage adds one solvent evaluation for the `init` Fock build; with ~7 coarse + 1-3 final cycles that is ~10-12 exact evaluations per SCF.
* `grad_stage` (INT8 XC-gradient GEMMs) is entered by the DF-RKS gradient through `ks_grad.base`; the solvent gradient wrapper sets `vac_grad.base = <solvent mf>`, so `mixed_precision` is found.  The solvent terms themselves stay FP64.
* UKS: `_get_vgrids` sums the two spin densities, `v_solvent` (nao,nao) broadcasts onto the (2,nao,nao) veff; `energy_elec` adds `e_solvent` to the vacuum energy computed from the unsolvated veff, DIIS gets `veff + v_solvent`.  OK.
* Charged/open-shell: only the total density enters; nuclear charges are the effective ones, electrons are the valence electrons, consistent for ECP.
* Remaining risks (not changed): (i) lowmem solves use `conv_tol=1e-7` (MINRES/GMRES, relative to |b|): fine for the 3e-6 final threshold, but a user who tightens `conv_tol_grad` below ~1e-6 should set `with_solvent.conv_tol`;
  (ii) `PCM.copy()` (TDDFT) shares `_intermediates` between the objects while equilibrium solvation is on; all keys are rewritten together by `_get_vind`, and the cache is keyed on the density, so this stays consistent;
  (iii) SMD has no lowmem mode and holds ~6 n^2 at build time; (iv) all solvent kernels use the current device only (multi-GPU runs the solvent on one GPU).

### 4.3 Scope notes (what is and is not in this tree)
* There is **no GPU ddCOSMO / ddPCM** in `gpu4pyscf/solvent` (grep finds only mentions in `_solvent_data.py` / docs), hence no FMM/Lebedev-spherical-harmonic solver to optimize; the whole family is boundary-element (surface point) PCM + SMD.  The "FMM-like" scaling remedy would apply to the matrix-free `S v`, `D v` matvecs (`pcm_left_multiply_*`), which only run in `lowmem` mode.
* Dense vs iterative: dense = ~2.7 n^3 once per geometry (GEMM + LU, 0.1-14 TFLOP, <= 100 s on a T4 even for SMD/Taxol) + 2 x n^2 x 8 B reads per cycle; matrix-free = k (10-30) matvecs x O(n^2) on-the-fly erf/exp evaluations x 2-4 solves per cycle.
  Dense wins everywhere it fits; the iterative path only matters for memory (SMD above ~100 atoms on 16 GB, IEF-PCM above ~150 atoms), where `lowmem_intermediate_storage` (PCM only; SMD has none) already exists.
* `gpu4pyscf/lib/cuest_wrapper.py` (CuEST PCM) is a separate class with its own `kernel/grad/_intermediates['q']`; it does not use `grad_qv/grad_solver/grad_nuc/_ensure_vind`, so the cache change cannot affect it.  `_coarse_solvent` skips anything that is not a `pcm.PCM`/`smd.SMD` (CuEST locks its surface settings).
* `gpu4pyscf/qmmm` (external-field / QM-MM `itrf`) was not audited.

## 5. What was implemented (commits on the branch)

* `7a4f645` `benchmarks/dft/solvent_cost_model.py` (CPU counting model).
* `c36be52` solvent changes: `solvent/pcm.py` (`_total_dm`, `_is_built`, `_solve_q_sym`, `_get_vind(with_vmat)`, `_ensure_vind`, DA-free `left_multiply_R`), `solvent/grad/pcm.py` + `solvent/hessian/pcm.py` (use `_ensure_vind`),
  `solvent/smd.py` (borrowed methods, `_atomic_numbers`), `solvent/grad/smd.py` (`dm=`), `scf/mixed_precision.py` (`coarse_solvent_lebedev`, `_coarse_solvent`), tests `solvent/tests/test_pcm_cache.py` and `test_pcm_mixed_precision.py`,
  CPU scripts `solvent_algebra_cpu_check.py` (passes: C-PCM/COSMO q vs qt 1.5e-15 / 9e-16 relative, IEF 1.3e-2 and SS(V)PE 2.3e-1 so their second solve stays; DA-free product 9e-15) and `solvent_df_projection_cpu.py`.
* `23965f5` `benchmarks/dft/bench_solvent.py`.
* `8a14c74` `RESULTS_solvent.md`, `_coarse_solvent` guard for non-PCM/SMD solvents, test tolerance.

Everything in this list is exact up to rounding except (a) the opt-in `coarse_solvent_lebedev` (final answer unchanged to the SCF tolerance, but the path differs) and (b) the SMD ECP fix, which intentionally changes SMD energies/gradients of ECP-atom molecules (bug fix).

## 6. CPU evidence for the two coarse-stage ideas (PySCF CPU PCM, IEF-PCM, def2-TZVP, def2-universal-jkfit)

**DF-projected solvent** (`solvent_df_projection_cpu.py`): at the converged density

| | max \|dv_g\| | dE_solv | \|dV\|_F | SCF E error with DF-solvent |
|---|---|---|---|---|
| water (43 AO, 556 pts) | 7.6e-4 | -1.6e-4 Eh | 2.9e-3 | -1.63e-4 |
| glycine (185 AO, 1408 pts) | 1.0e-3 | -1.7e-4 Eh | 2.9e-3 | -1.74e-4 |

i.e. the plain DF projection is a **1e-4 Eh** approximation (first order in the fitting residual; size-independent), which is ~1.3 % of E_solv and about 10x the error of the coarse-grid stage.  It is therefore only usable with
a **defect correction**: one exact pass pair at a density D0 gives `dv = v_ex(D0) - v_DF(D0)` and `dV = V_ex(q0) - V_DF(q0)` (smooth, nearly D-independent), after which `v = v_DF(D) + dv`, `V = V_DF(q) + dV`.
Water: E error 8e-7, and the exact-solvent \|g\| at the DF-solvent density is 4.2e-4 (stop at 1e-3) / 2.8e-6 (stop at 1e-5, essentially the final-stage threshold 3e-6).  The final exact stage reproduces the exact energy to 1e-13 in all runs.
Glycine (185 AO), defect refresh at \|g\|<1e-2, then DF-projected to \|g\|<1e-5 (11 DF cycles): E error 6.0e-7, exact-solvent \|g\| at the hand-over 6.5e-6, 5 exact cycles to the tight CPU thresholds (|g|<1e-7, dE<1e-10) vs 16 from scratch, final E identical to 4e-12.
Implementation sketch for GPU (not done): per DF block `cderi_tril` ([nL, npair], `df.loop(unpack=False)`): `aux = cderi_tril @ dm_sparse`, `v_g += aux @ Wt[blk]`, then after the solve `vj_sparse += (Wt[blk] @ q_sym) @ cderi_tril`
(the second piece merges into the J back-contraction of `get_jk`).  `Wt = aux_coef^T (P|g)` must use the **same aux order and `aux_coef`** as the cderi (`_decompose_j2c` returns `aux_coef` already permuted by `aux_sorting`; 'cd' vs 'ed' (eigen) branches);
`(P|g)` can come from `int3c2e_bdiv.int2c2e` of the concatenated mol `auxmol + fakemol_for_charges(grids, expnt=xi^2)` (off-diagonal block), one-time naux x nsurf.  GPU test: `Wt^T (B D)` against `int1e_grids(dm)` to ~1e-3, and exact == staged final energy.

**Coarse solvent surface** (opt-in 5): coarse (order 17 / 23) converged to |g| 1e-4, then the fine (302) surface: water dE_surface 7.7e-5 / 2.1e-5 Eh, |g| at hand-over 3.8e-4 / 1.0e-4; glycine 1.9e-4 / 6.1e-5 Eh, |g| 7.6e-4 / 2.3e-4.

## 7. How to validate on a GPU (all untested)

```
pytest gpu4pyscf/solvent/tests/test_pcm_cache.py gpu4pyscf/solvent/tests/test_pcm_mixed_precision.py   # + existing test_pcm*.py, test_smd*.py
python benchmarks/dft/bench_solvent.py --warmup --grad                                   # FP64 + IEF-PCM, per-component table, gradient _get_vind count must be 0
python benchmarks/dft/bench_solvent.py --warmup --grad --scheme crt
python benchmarks/dft/bench_solvent.py --warmup --grad --scheme crt --coarse_solvent_lebedev 17 --ref <previous --out json>
python benchmarks/dft/bench_solvent.py --warmup --solvent none --scheme crt              # vacuum baseline: solvent overhead = difference
python benchmarks/dft/solvent_cost_model.py --mol 057_Tamoxifen                          # compare with the measured table
```
Calibration: `_get_v` + `_get_vmat` times per call (BPA, T4) against 0.42 s + 0.57 s; `PCM.build` against 0.7 s; the first gradient `grad_qv`/`grad_solver` split.
If the measured integral time is far below the model, item 4 loses its value and only A100-class overheads (item 7) remain.
