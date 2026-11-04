# HANDOFF — GPU4PySCF wB97M-V + DF optimization work (INT8/CRT emulated FP64, mixed precision, hardware-aware paths)

Read this first, then `RESULTS_mixed_precision.md` (all measured tables), `NOTES_next_steps.md` (short running notes),
`research/research_report1_literature.md` and `research/research_report2_flop_reduction.md` (two literature/FLOP-model surveys by research agents),
`RESULTS_vv10.md`, `RESULTS_launch_overhead.md`, `RESULTS_solvent.md` (the three implementation agents' reports).

---------------------------------------------------------------------------------------------------------------------------------

## 0. TL;DR

**Task as given by the user:** autonomously optimize the DFT code in GPU4PySCF (v1.9.0 base), target workload wB97M-V (MGGA + VV10, range-separated hybrid)
with density fitting (def2-TZVP, def2-universal-jkfit), 50-100 atom molecules. Rules: profile before changing; correctness first (a regression check per
change); report failures faithfully (negative results are recorded below); small commits; **no PR unless asked**; **work in local git only** until the user
sets up GitHub access (the push to `gamedeo/gpu4pyscf` returned 403 earlier). Hardware direction: be less bound by FP64, run well on GPUs without native
FP64 (T4 etc.), *and* speed up A100/H100; aim at the raw FLOPs of the best dtype; try to rewrite hard DFT pieces as dense tensor ops.
User standing instructions: **do not abandon an optimization just because it does not work on T4 — keep Ampere/Ada-only ideas as architecture-dispatched
options**; an L4 Colab runtime on the account belongs to a different project of the user (do not use it unless told); be frugal with Colab credits and always
`colab stop`; Colab allows **max 2 simultaneous allocations**.

**Headline results (Bisphenol A, 33 atoms, nao 623, naux ~1900, 1.06 M XC grid points (99,590), 217 k NLC points (50,194); reference energy −731.6494045101, file `reference/bpa_tight.json`):**

| | SCF | gradient | dE vs tight FP64 ref | max grad err |
|---|---|---|---|---|
| T4, original code (FP64) | 249.7 s | 47.1 s | – | – |
| T4, best earlier scheme (FP32 stages + FP64 polish, round 2) | 87.0 s | 17.3 s | 3.7e-10 | 1.2e-7 |
| **T4, new `crt` scheme (this work)** | **39.6 s** | **~14 s steady, 19 s first call (JIT)** | 1.8e-10 | 1.08e-7 |
| A100, original (round 2 measurement) | 10.95 s | 2.5 s | – | – |
| **A100, plain FP64 with merged VV10 kernel + fused XC loops** | **5.5 s** | **2.0 s** | −5.3e-10 | 3.4e-7 |
| A100, scheme `crt` (for reference) | 17.5 s | 1.6 s | −4.5e-10 | 1.9e-7 |
| T4 IEF-PCM water (3695 surface points), `crt` vs plain FP64 | 61.4 s vs 215.3 s (3.5x) | 23.4 s | 2e-10 | – |

So the T4 is ~5.0x faster than before end-to-end (296.8 s → ~54–59 s) but is still ~6–8x slower than the A100 (FP64 tensor cores); dtype tricks alone
cannot close that gap (FP64 peak 0.25 vs 19.5 TFLOP/s; INT8 130 vs 624 TOPS). On the A100 the XC GEMMs already run at ~92 % of the FP64 tensor-core peak;
further A100 gains need FLOP cuts, not better utilization (details §6).

---------------------------------------------------------------------------------------------------------------------------------

## 1. Repository state

* Repo: `/home/user/gpu4pyscf` (fork `gamedeo/gpu4pyscf`), **branch `claude/serene-meitner-kinyv4`**, ~120 local commits, **nothing pushed**. Working tree clean at
  the time of writing. (Cloud-session instructions mention a designated branch `claude/festive-galileo-28ip2z`; the user later said "just work within local git",
  so the work stays on `serene-meitner`. Ask the user how/where to push once GitHub access works; do not create a PR unless asked.)
* Agent worktree branches `worktree-agent-a6376478a84ce0aaf` (VV10), `worktree-agent-a4a064688593fc6e5` (launch overhead), `worktree-agent-ae1020a66ad96d860`
  (solvent) are **already merged** into the branch above. Their worktree directories live under `.claude/worktrees/` (excluded via `.git/info/exclude`; never
  `git add -A` them — an embedded-repo commit happened once and was amended away).
* Committing: end commit messages with the attribution lines required by the session harness, if any; no model identifiers in commits.

### Files added / changed (by area)
| Area | Files |
|---|---|
| INT8/CRT GEMM library | `gpu4pyscf/lib/crt_gemm.py` (new, ~430 lines) |
| Hooks | `gpu4pyscf/dft/numint.py` (`XC_PRECISION='crt'`, `_vmat_block_crt`, `_eval_rho2_crt`, NLC-grid branch, `XC_CRT_MODULI`, `XC_CRT_RHO`, fused-loop code from the launch agent), `gpu4pyscf/df/df_jk.py` (`K_PRECISION='crt'`, `K_CRT_MODULI`, `K_CRT_BUDGET`), `gpu4pyscf/df/df.py` (`CDERI_PRECISION`, `CDERI_CRT_MODULI`, `_j3c_dot`), `gpu4pyscf/grad/rks.py` (`XC_GRAD_CRT_MODULI`, contract-first `_apply_dm` / `_exc1_block`, NLC gradient contract-first), `gpu4pyscf/df/grad/rks.py` (`grad_stage`) |
| SCF driver | `gpu4pyscf/scf/mixed_precision.py` (schemes `crt` / `classic`, coarse stage, `final_stage`, `grad_stage`), `gpu4pyscf/scf/hf.py` (`scf()` wraps the final stage; init-|g| shortcut `_accept_converged_init`) |
| VV10 | `gpu4pyscf/dft/vv10_kernels.py` (agent: symmetric tiled cupy RawKernels, `VV10_KERNEL='sym'|'libgdft'`, `VV10_PRECISION='fp64'|'fp32'`) |
| Fused XC loops | `gpu4pyscf/dft/xc_fused.py` (agent: NVRTC `rho_from_c`, `build_B`, `row_dot3`; `numint.XC_FUSED`, `XC_MERGE_VMAT_GEMM`), `gpu4pyscf/lib/cutensor.py` (plan cache + host-path trim), `gpu4pyscf/dft/gen_grid.py` (view cache) |
| Solvent | `gpu4pyscf/solvent/*` (agent: gradient cache fix, SMD ECP charge fix, C-PCM/COSMO single solve, `coarse_solvent_lebedev` opt-in) |
| Tests (all pass on T4; most on A100) | `dft/tests/test_xc_crt.py`, `test_xc_fused.py`, `test_xc_fp32.py`, `test_vv10_kernels.py`, `grad/tests/test_exc_contract_first.py`, `lib/tests/test_cutensor_plan_cache.py`, `solvent/tests/test_pcm_cache.py`, `test_pcm_mixed_precision.py`. Last full regression on T4: **113 passed** |
| Benchmarks / experiments | see §10 |

---------------------------------------------------------------------------------------------------------------------------------

## 2. What was built — design and mechanics

### 2.1 `crt_gemm.py`: FP64-class GEMM from INT8 tensor-core products (Ozaki-scheme-II / CRT)
Own implementation (GEMMul8 could not be cloned: the permission classifier refused; never retried — ask the user first).

* Operand rows are scaled by a power of two (`2^e`, from the row max), quantized to q-bit integers (`q = floor((8n − log2 K − 1)/2)`, cap `QMAX=52`; n = number of
  moduli), reduced modulo n pairwise-coprime moduli ≤ 256 (`ALL_MODULI = 256,255,253,251,247,241,239,233,229,227,223,211,199,197,193,191,181,179,173,167`) to
  **balanced int8 residues**; each residue product is an exact int8×int8→int32 cuBLAS GEMM (`cublasGemmEx`, TN, compute type 32I = 72; `K·128² < 2³¹ ⇒ MAX_K=120000`);
  Garner's balanced mixed-radix CRT (generated NVRTC code with the moduli and inverses as literals, so `%` compiles to mul-hi) recovers the integer, evaluated by
  Horner in double and scaled by `2^(ea+eb−2q)`. Precision knob = number of moduli (≈8 bits each). Residues of q>30-bit values use 3 int32 limbs
  (`x = xh·2³⁶ + xm·2¹⁸ + xl`) because 64-bit `%` is slow on GPUs (valid |x|<2⁵⁴).
* API: `Operand(src[ncomp, rows, ng], nmod, q)` (rows = non-reduced index, reduction axis = (comp, ng) concatenated → one GEMM over comp×grid);
  `PlanesT(src3(d0,d1,d2) with stride-1 axis 1, nmod, q)` (**transposing quantize kernel** through shared-memory tiles: feeds the TN GEMM from [k][g]-layout AO
  values or the DF tensor [i, j, L] with L contiguous, no transposed copies); `matmul_operands(oa, ob, out)` (one **strided-batched call over the moduli** —
  small m,n underfill the T4 otherwise: 6.4 → 4.0 ms per XC block), `gemm_nt(a,b,nmod)` (vmat, K2), `gemm_planes(src3, bt, nmod)` (rho, K1, XC gradient),
  `gemm_dense(a,b,nmod)` (DF aux-metric transform j3c @ L⁻ᵀ, row-chunked under a memory budget), `rho_from_cT(c, ng, xctype)` (warp-per-grid-point rho/∇rho/τ assembly).
* Costs (T4): INT8 GEMMs run at ~37–40 TOPS (130 peak, power-capped T4). CRT only pays for **long reductions** (K ≳ 500): the Garner reconstruction costs
  ~n² integer ops per output element, more than a short-K GEMM itself (measured: XC-gradient second product with K = nocc = 61 got slower → left in FP64).
* Constraints found: legacy `cublasGemmEx`/`GemmStridedBatchedEx` for INT8 supports only TN layouts in the batched form (**NN/NT batched return
  CUBLAS_STATUS_NOT_SUPPORTED on Turing *and* Ampere**; non-batched NN works but is slower); Turing needs m,n,k multiples of 16 (all operands are padded).
  cublasLt (with special data orders) was never tried.

### 2.2 Hooks that make the SCF use it
* **XC (`numint.nr_rks`, `nr_nlc_vxc`)**: `XC_PRECISION='crt'` → rho via `_eval_rho2_crt` (INT8 half-transform, C = Cᵀ·ao with grid points as rows, `rho_from_cT`),
  vmat via `_vmat_block_crt` (MGGA: one GEMM of length 4·ng with A=[ao₀..ao₃], B=[Σ w ao, w_τ ao₁..₃] from the fused `xc_fused.build_B`). The unfused per-block loop is
  used (fused loops are disabled when crt is active, `crt_*` take precedence). Wv convention for crt = the fused-loop convention (row 0 ×½, row 4 ×¼).
  NLC-grid rho/vmat have the same branches. UKS (`nr_uks`) is **not** implemented for crt (falls back to FP64/fused XC; K still uses INT8).
* **K (`df_jk.get_jk`)**: `K_PRECISION='crt'`: K1 = `gemm_planes(cderi[i,L,j] view, Cᵀ)` → rhok rows (i,L), cols k; K2 = `gemm_nt(rhok, rhok)`; sub-blocks of L chosen
  so the int8 planes stay under `K_CRT_BUDGET=1.5e9` bytes. Only single-factor (real, hermitian) density; otherwise FP64.
* **DF build (`df.py`)**: `CDERI_PRECISION='crt'` replaces the big FP64 GEMM `j3c·L⁻ᵀ`: 4.70 s → 1.38 s per build. **14 moduli (q=44 cap) was NOT enough**:
  the large entries of L⁻ᵀ amplify the 2⁻⁴⁸ operand truncation (CRT n=14: 5e-12 vs long double, FP64 GEMM itself: 1.4e-14); this produced a +5.2e-9 Eh energy shift.
  Fixed with q cap 52 / 3-limb residues / **15 moduli** (3.6e-13, dE back to 1.4e-10).
* **Gradient**: `grad.rks.XC_GRAD_CRT_MODULI` (only the long-K product C(Cᵀx)); XC grad contract-first (no [3,nao,nao] matrices); **NLC gradient contract-first**
  (3.3 → 1.07 s); `mixed_precision.grad_stage(mf)` context in `df/grad/rks.py::energy_ee`.
* **VV10**: merged agent's symmetric-pair tiled kernel (each unordered pair once, tile-local FP32 coordinates, shuffle reduce-scatter). T4, N=303k: FP64
  20.1 → 12.4 s (exact, 1.6x), FP32 1.0 → 0.69 s (1.45x vs the older FP32 libgdft kernel); A100: FP64 1.49x, FP32 3.6x vs libgdft.
  FP32 VV10 error ≈1e-9 in E, no visible gradient impact. Results are not bit-reproducible (atomicAdd order).
* **Launch overhead (agent)**: fused NVRTC kernels + merged K=4·blk vmat GEMM; cuTENSOR host-path trim; Python calls per nr_rks 106k → 33k. T4 FP64 nr_rks 8.47 → 7.71 s,
  FP32 1.53 → 1.04 s; A100 0.390 → 0.194 s (fused vs unfused loops). Fused vs old loops differ by 3e-15 relative.

### 2.3 The `crt` SCF scheme (`mixed_precision.py`)
Enable: `mf.mixed_precision = True` (or a dict). `scheme='auto'` → `'crt'` on compute capability 75/86/87/89, `'classic'` otherwise.
Sequence for `crt`:
1. **Coarse stage**: `mf.grids` → copy with `atom_grid=(50,194)`, `mf.nlcgrids` → `(30,110)`; K/XC with `crt_moduli_coarse=8` moduli, DF tensors built with 15 moduli
   (static), VV10 FP32; converge to `|g| < coarse_conv_tol_grad=1e-3`, `conv_tol=1e-2` (≈8 cycles, 3.6 → 1.7 s each).
2. **Final stage** (`final_stage(mf)` context around the normal `_kernel`): full grids, XC/K at `crt_moduli_final=11` moduli, VV10 FP32 (`final_vv10`), gradient threshold
   = usual × `final_grad_factor=0.1` (needed: factor 1.0 gives gradient errors 8e-7 instead of 1e-7; costs ~2 cycles). ≈7 cycles.
3. Gradient runs with `grad_stage` (XC-grad CRT product).
`classic`: the round-2 scheme (FP32 stage 1, FP64-K/FP32-XC stage 1b with plateau detection, FP64 final), coarse stage **off by default** (neutral on A100: 13.1 vs 12.5 s Tamoxifen).
Options table is `mixed_precision.DEFAULTS`. Switch benchmarks with `bench_wb97mv.py --mixed auto --scheme crt|classic|auto`.

---------------------------------------------------------------------------------------------------------------------------------

## 3. Measured results (see `RESULTS_mixed_precision.md` for full tables)

### 3.1 Accuracy of the CRT GEMM (624×624, K=61440, AO-like synthetic operands; relative to max element)
moduli 8 / 10 / 12 / 14: 9e-7 / 9e-9 / 2.6e-11 / 4.7e-13 (older q cap 44); total time incl. both quantizations 22.5 / 25.6 / 35.8 / 38.5 ms vs 238 ms native FP64 on T4 (10.6x … 6.2x).
Real XC operands: n=12 → max |ΔV| 2.5e-11 (wB97M-V). SCF-level: n=11 XC/K reaches ΔE 3e-10 (gradient errors dominated by SCF convergence, not the GEMM).

### 3.2 T4 time progression for BPA SCF (plain/non-staged unless noted)
FP64 249.7 → staged FP32 (round 2) 87.0 → CRT 11 moduli first version 81.7 → batched GEMM + transposing quantize + INT8 rho/K1 63.8 → NLC grid on INT8 57.9 → coarse-grid first stage
46.4 → INT8 DF build 48.7 (incl. 15-moduli accuracy fix; DF.build 7.0 → 2.3 s) → symmetric VV10 + coarse NLC grid + contract-first NLC grad 45.7 → fused build_B + rho_from_cT 39.6.

### 3.3 Current T4 profile (BPA, scheme crt, last run; per-call numbers)
SCF 39.6 s = XC (`nr_rks`, 17 calls: 8 coarse ≈0.6 s + ~9 full ≈2.2–2.5 s) 21.9 s · NLC (`nr_nlc_vxc`; VV10 kernel 7.7 s total over 18 calls) 5.9 s ·
`get_jk` 10.0 s (34 calls ≈0.3 s each + DF build 2.2 s) · misc 1.3 s · grids 0.4 s.
Inside one *full-grid* crt `nr_rks` (before the last fusion step): quantize kernel 0.68 s, INT8 GEMMs 0.63 s (≈ 37 TOPS), `quant3t` 0.19, row-max 0.12+0.06, recon 0.12, AO eval 0.46 (both loops), glue.
Gradient (steady 14.3 s): DF J/K gradient ≈7 s (FP64 Rys `sum_ejk_int3c2e_ip1` 3.3 s, fill_int3c2e 0.5 s, cuTENSOR/cuBLAS DGEMMs ≈3 s), XC grad 5.1 s, NLC grad 1.07 s, h1e 0.3 s.

### 3.4 A100 (sm_80 libs were compiled on the T4 host; BPA unless stated)
`nr_rks` = 0.196 s wall = 0.195 s GPU busy: cutlass FP64 tensor-core GEMM 0.131 s (≈18 TFLOP/s ≈ 92 % of 19.5), AO 0.040, build_B 0.020 → **GEMM-roofline-bound**.
Other numbers: Tamoxifen (57 atoms, 1.84 M points): FP64 12.5 + 4.15 s grad; Vitamin C def2-QZVPP: 5.5 + 1.7 s. `XC_MERGE_VMAT_GEMM` (merged K) vs four GEMMs: 0.194 vs 0.206 s (keep merged).
`cublasDsyrkx` (triangle-only symmetric product): 6x slower than dgemm (SIMT) — no use. INT8 CRT on A100: 3x slower than FP64 (XC 0.705 vs 0.218 s/call) because the
quantize passes are memory/launch bound and FP64 TC is strong; `auto` therefore picks `classic` there.

### 3.5 Scaling with basis size (Vitamin C, 20 atoms, wB97M-V+DF, T4, grids (75,302)/(50,194); `scale_basis.py`)
| basis | nao | crt | FP64 | speedup | dE |
|---|---|---|---|---|---|
| def2-SVP | 208 | 27.9 s | 38.7 s | 1.4x | −3.3e-10 |
| def2-TZVP | 420 | 28.7 s | 60.7 s | 2.1x | −2.4e-10 |
| def2-TZVPD | 534 | 39.2 s | 82.8 s | 2.1x | −1.7e-10 |
| def2-QZVPP | 924 | 58.7 s | 142.3 s | 2.4x | −4.2e-10 |
DF tensor (2.9 GB at QZVPP) stayed on the GPU. **Not yet tested:** BPA-size (30+ atoms) at QZVPP or larger systems where the DF tensor must live on the host
(T4 Colab host: 2 CPU cores, 12.7 GB RAM!), multi-GPU, nao > ~1500, `MAX_K` limit for naux > 120k.

### 3.6 Compatibility matrix (`compat_test.py --grad`, T4; crt scheme vs plain FP64)
dE ≤ 1e-10 for LDA, PBE, B3LYP, TPSSh, r2SCAN, ωB97X-V, B97M-V (def2-TZVP), Cartesian AOs (+3.2e-10), UKS wB97M-V (CH₃) and UKS PBE0 (NO₂), PtCl₄²⁻ ECP (6.5e-10), acetate anion
def2-TZVPPD, water def2-QZVPP/QZVPPD, acetic acid def2-QZVPPD (390 AOs). Largest: **CH₃I (iodine ECP) dE −5.3e-9** (large AO dynamic range per row → use more `crt_moduli_final` for Z>36).
AgCl: neither FP64 nor crt converges (SCF problem, not precision). Gradient differences 1e-7–1e-5 are mostly the FP64 reference's own convergence (default |g|≈3e-5), not CRT.
Finite-difference check of the analytic DF gradient (grid response on; PBE0 and wB97M-V; mix_outside/inside; crt): mismatch 1.15e-6 (h=0.002) → 2.9e-7 (h=0.001) ∝ h² = FD truncation;
the analytic gradient is consistent (the research agent's worry about FR+SR vs FR+LR is not supported).

### 3.7 Solvation (IEF-PCM water on BPA, T4)
crt vs FP64: E −731.6685517723 vs −731.6685517725, SCF 61.4 s vs 215.3 s. Solvent costs ≈1.06 s per SCF cycle (`_get_vind`, two FP64 point-charge integral passes over 3695 surface
points; 15–17 s of the SCF) and ≈4.5 s of the gradient. Agent model: solvent scales O(natm²); Taxol-size ≈21 s/cycle on T4; SMD (590 pts/atom) at 113 atoms needs 17 GB. See `RESULTS_solvent.md`.

---------------------------------------------------------------------------------------------------------------------------------

## 4. Negative / neutral results (do not repeat)

* cuBLAS 13.1 FP64 emulation: slower than native on A100, silently native on T4. cuTENSOR "3xTF32" compute descriptor = plain FP32 SIMT. TF32 paths gave nothing on A100.
* Mixed precision FP32 (round 2 scheme) is useless on A100 (FP64 TC ≈ FP32 SIMT) → `_default_gemm_precision` returns fp64 for sm_80+.
* FP32 XC cannot go below |g|≈1e-5 (Vxc-matrix FP32 accumulation, ∝ chunk length; chunking costs 10x); K FP32 floor 1e-4 → always 2–3 exact Fock builds at the end.
* Init-|g| shortcut (accept converged init) never fires (FP64 |g| after FP32-XC solution ≈4e-5).
* Modulus ladder inside the full-grid stage (e.g. 9 moduli until |g|<1e-4): no gain (fixed per-cycle cost); stopping at 8 moduli never converges.
* def2-SVP seed density: saves cycles, loses time (50.7 vs 46.4 s).
* CRT for the short-K gradient product (K=nocc): slower; CRT in general is not worth it for K ≲ 300.
* Dual-grid defect-corrected SCF: **defect is smooth** (XC defect |ΔV|=1.8e-3 vs |V|=21; changes 1.2 % between cycles 3→9, 0.1 % after cycle 5) but modelled gain only ~4 s on T4/BPA
  because the coarse cycle still costs ~1.7 s and two defect evaluations are fine-grid evaluations; revisit when coarse cycles get cheaper.
* VV10: distance cutoff useless at needed accuracy; Gaussian/exponential-sum forms not low-rank; far-field low rank caps at ~2x for 0.3–0.6 M points (wins only >1–2 M);
  tensor-core TF32 split only helps ~6 of 24 instructions per pair.
* `cublasDsyrkx` on A100: 6x slower than dgemm. Batched non-TN INT8 gemmEx unsupported on Turing and Ampere.
* DF-JX / aux-density expansion fails for wB97M-V (τ not formable from fitted density). ISDF/THC for XC, COSX, ADMM, localized exchange, incremental K from ΔD, Newton-type SCF,
  better initial guesses, hierarchical VV10, Richardson extrapolation of the NLC grid: judged not applicable at this size (reasons in `research/research_report2…`).
* INT4 CRT has the same bits/s as INT8; FP8 Ozaki-II is not available on T4/A100.

---------------------------------------------------------------------------------------------------------------------------------

## 5. Hardware facts used (dense, vendor specs; achieved in parentheses)

| | FP64 | FP32 | TF32 TC | FP16/BF16 TC | INT8 TC | mem BW |
|---|---|---|---|---|---|---|
| T4 (70 W) | 0.25 TFLOP/s (≈0.20 GEMM) | 8.1 | – | 65 | 130 TOPS (≈37–40 achieved by cuBLAS int8 GEMM) | 320 GB/s |
| A100 40GB | 9.7 (19.5 TC; ≈18 achieved on the XC GEMMs) | 19.5 | 156 | 312 | 624 | 1.55 TB/s |
Other GPUs (not measured): L4/L40S/A10/consumer (FP64 1/64, strong INT8) → `crt` scheme should help (auto picks it for cc 8.6/8.9); H100: FP64 TC 67; B200/Blackwell consumer parts have
weak FP64 and strong INT8/FP8 → the `crt` scheme and Ozaki-II FP8 are *more* relevant there; **`auto` selects by compute capability only — a measured-throughput-based selector is an open item.**
Colab host (T4 runtime): 2 vCPU, 12.7 GB RAM → background compilation or Python launch-bound work slows GPU timings (a background `make` made an XC call 40 % slower; **don't time while building**).

---------------------------------------------------------------------------------------------------------------------------------

## 6. How to run / reproduce (infrastructure)

* **Compute**: Colab CLI `colab --auth adc ...` with Application Default Credentials. Credentials come from env `GOOGLE_ADC_B64` → decoded to a mode-600 file outside the repo
  (`/root/.secrets/adc.json`, `export GOOGLE_APPLICATION_CREDENTIALS=...`). **Never print, copy, commit or place credentials in the repo or in prompts.** `colab ls`, `new -s NAME --gpu T4|A100`, `upload`, `download`, `exec -f`, `stop -s NAME`.
  Max 2 simultaneous sessions (the user told us; a subagent starved on this once). Start on T4; A100 only for final numbers; always `colab stop`.
* Helpers in `colab/`: `cx.sh SESSION TIMEOUT [-b] 'cmd'` (run a shell command; `-b` = background with log), `env.sh` (runtime env), `build_t4.sh` (full build, ~25 min on a T4 host),
  `build_sm80_on_t4_host.sh` (build sm_80 libraries on the cheap T4 host, tar the `.so` files, upload to an A100 session: avoids burning A100 credits on compilation),
  `setup_a100.sh`, `example_batch_bench.sh`. Pattern: write a `batchNN.sh`, upload, run with `cx.sh … -b`, poll for a `done` flag in a background shell loop.
* **Setup recipe**: `pip install pyscf==2.14.0 gpu4pyscf-libxc-cuda12x==0.9.0 cutensor-cu13 basis-set-exchange pyscf-dispersion geometric`; cmake in `/content/build` with
  `-DCUDA_ARCHITECTURES=75-real` (or `80-real`) `-DBUILD_LIBXC=OFF` on `gpu4pyscf/lib`; targets `gdft cupy_helper gvhf gecp gvhf_rys gvhf_md gint solvent sem pbc mgrid_v2 mgrid_v3`.
  Runtime env: `PYTHONPATH=/content/src/gpu4pyscf CUPY_ACCELERATORS=cub,cutensor LD_LIBRARY_PATH=/usr/local/lib/python3.13/dist-packages/cutensor/lib:$LD_LIBRARY_PATH`
  (**keep the existing `LD_LIBRARY_PATH`** — dropping it gave `cudaErrorInsufficientDriver`). Source tarball: `tar czf … --exclude=.git --exclude='*.so' --exclude=build --exclude=.claude gpu4pyscf`.
* Python-only changes need no rebuild: upload the changed files into `/content/src/gpu4pyscf/…`. NVRTC kernels compile on first use per modulus count ⇒ always use `bench_wb97mv.py --warmup`;
  the *first* gradient call includes ~5 s of JIT (bench "gradient time" is a first-call number).
* Reference data: `reference/bpa_tight.json` (tight FP64 energy −731.6494045101365 and gradient, used via `bench_wb97mv.py --ref`). The default-tolerance FP64 run itself has gradient noise ~1e-6.
* Verify before claiming: correctness gate per change = energy ≤ 1e-8 Eh (we hold ~2e-10), gradient ≤ 1e-6 Eh/bohr (we hold 1e-7), regression tests listed in §1.

---------------------------------------------------------------------------------------------------------------------------------

## 7. OPEN WORK, ranked and annotated (start here)

Estimates are for T4/BPA unless noted ([M] measured, [G] guess; see the research reports for the arithmetic). "exact" = same fixed point to the gates above; "opt-in" = changes the answer.

### 7.1 Fundamental / algorithmic (reduce FLOPs, fewer expensive evaluations)
1. **Occupied-projected Fock** (research report 2, E4; exact): never build the nao×nao XC/NLC/K matrices in early cycles; work with F·C_occ (nao_sub:nocc ≈ 9:1 in XC GEMM work, K·C_occ ≈ 0.6x K).
   DIIS error vectors need only F·C_o plus a lagged F_vv (the first report's "DIIS impossible" claim was wrong). Estimated 7–13 s of the SCF. High effort; needs a direct-minimization/orbital-rotation or modified DIIS path in `scf/hf.py` / `scf/diis.py`.
2. **Defect-corrected dual-grid SCF** (E3; exact; go/no-go PASSED, `defect_test.py`): coarse-grid cycles with frozen Δ=(fine−coarse) for XC+NLC, only ~3 fine evaluations. Gain small now (~4 s) unless coarse cycles get cheaper (see 7.2 K reuse).
3. **Cheaper coarse cycles**: lag K / ACE-style K reuse (K changes little late in the coarse stage), lag NLC potential (smooth), reuse of the Rys/DF tensor; then 7.1.2 becomes worth ~10 s.
4. **DF-gradient** (~7 s of the 14 s gradient): build `j3c_oo` from the cached SCF tensor instead of refilling it (needs AO/aux layout mapping between SCF and gradient evaluators; ~2.4 s); FP32/threshold-controlled Rys for the derivative integrals (T4 only; Rys has no GEMM-shaped step so INT8/CRT is not applicable);
   the contraction DGEMMs in `df/grad/rhf.py` (cuTENSOR einsums) can use `crt_gemm.gemm_planes` where K is long.
5. **Single-tensor range-separated DF-K** (cuEST scheme; opt-in; ~4.7 s SCF, 3.5 s grad): one DF tensor K = Bᵀ(0.15 I + 0.85 G)B with G = L⁻¹L_erf L⁻ᵀ. Accuracy vs `mix_outside_kernel` unmeasured (`mix_inside_kernel` already exists as a knob: halves K, dE ≈ 1e-6 Eh).
6. **Smaller grids as an opt-in** (SG-3: ~0.54x points; tens of µEh).
7. **THC/ISDF-style exchange with INT8 GEMMs** for much larger systems (≥150 atoms) where the DF tensor no longer fits and occupied-RI-K becomes memory-bound; at 50–100 atoms research report 1 says FLOPs ≈ equal. Revisit with real numbers at scale.
8. **Solvent**: DF-projected solvent response in the coarse stage (agent spec in `RESULTS_solvent.md`; CPU evidence: plain projection off by 1.7e-4 Eh, with one exact defect pass 6e-7 Eh; modelled ~7 s at BPA, ~150 s at Taxol size) and a cached/FP32 surface-potential tensor; FP32 or CRT for `int1e_grids`.

### 7.2 Hardware-aware utilization (T4 / A100 / H100 / Blackwell / L-series)
1. **XC on T4 (21.9 s)**: fuse the quantize passes (row-max + quantize + B-operand build into one or two kernels per block; one AO pass per block feeding both rho and vmat — E8, ~1.5 s), cache A-operand
   row exponents across SCF cycles, use the narrow int32 path with n=10 (q=30) where the accuracy gate allows (~9 % less GEMM/quantize traffic); symmetric tau-term tiles (E6: tau = 3 of 4 vmat GEMM units are symmetric; ~2–2.5 s; **needs a custom tiled INT8 symmetric product, `syrkx` is useless**).
2. **K on T4**: static residue cache of the DF tensor (quantize once per SCF: ×1.4 memory of FP64, ~8 GB for BPA/TZVP, so only for small systems or with reduced moduli), K2 symmetric (SYRK-like) product, tril-packed unpack-free path (fuse `fill_symmetric` into the quantize kernel).
3. **INT8 via cublasLt** (IMMA with `CUBLASLT_ORDER_*` layouts) to get NN/NT/batched layouts and better efficiency than the legacy `gemmEx` TN (≈30 % of INT8 peak on T4). Untried.
4. **A100/H100**: XC already at FP64-TC roofline; only FLOP cuts help (7.1). H100 (FP64 TC 67 TFLOP/s) will be even more AO/quantize/memory-bound. For INT8 CRT to win on Ampere/Hopper it needs fused quantize + large K; the cuBLAS 13.4 Ozaki-II auto-emulation and GEMMul8 are worth an A/B on H100/B200 (research report 1 §7).
5. **Architecture-dispatched options to KEEP (user request)**: TF32 3-split with outside-TC accumulation (A100/H100 early cycles), FP8 Ozaki-II (Hopper/Ada/Blackwell), cublasLt INT8 layouts, Blackwell weak-FP64 handling. Make `scheme='auto'` a measured micro-benchmark selector (time one FP64 GEMM and one CRT GEMM at startup) instead of a compute-capability table.
6. **Multi-GPU** (`multi_gpu.run` paths exist in numint/df): untested with crt (per-device grids/DF slices, planes budget per device).
7. Python/host overheads remain visible on fast GPUs (A100 SCF: 'unaccounted' 0.56 s, DIIS/diagonalization, grids.build): CUDA graphs per block shape were judged unsafe for determinism by the launch agent, multi-stream overlap of VV10 with K/J (NLC depends only on dm) is described but not built.

### 7.3 Memory efficiency and scaling
* The DF tensor dominates memory: 2.98 GB (BPA/TZVP), 2.9 GB (Vitamin C/QZVPP, naux 1056). Packed-triangle FP64 now; ideas: store as FP32 (+correction) or INT8 residues for K, half-triangle K contraction, on-the-fly recomputation, host-resident with prefetch (code path exists: `use_gpu_memory=False`, only lightly exercised with crt).
* crt memory: int8 planes `n × rows × K_pad` per operand + `n × m × n × 4 B` int32 products; budgets `K_CRT_BUDGET` (1.5 GB), `gemm_dense(budget=1e9)`; XC blocks use block-size-dependent buffers — check peak memory at nao ≥ 1500 (QZVPP on 30+ atoms) and for the 4·ng reduction length vs `MAX_K=120000` (chunked automatically).
* Scaling tests still to do: 100+ atoms (Taxol 113, Valinomycin 168 in `benchmarks/molecules/organic`), QZVPP/QZVPPD on BPA-size systems (DF tensor ≥ 15 GB ⇒ host-resident), aux basis with l=5 (check GPU integral support), multiple GPUs, `max_memory` interactions.
* Solvent/SMD memory (17 GB at 113 atoms) and dense K/R matrices: see agent report; `lowmem` PCM path exists.

### 7.4 Novel paradigms (theory discussed with the user; none built)
* The CRT GEMM is already a *ring-native* kernel (exact integer residues + one CRT per output). Extending the ring-native idea: accumulate residues across grid blocks with one global fixed-point scale and reconstruct once per Vxc (removes per-block Garner cost); redundant extra modulus as an
  **overflow/error detector** (consistency check of the CRT result gives free error control → automatic modulus selection per call).
* TPU/MXU port (theory only, `NOTES_next_steps.md`): INT8 residue GEMMs map to INT8-capable MXUs (bf16 MXU is FP32-accumulated ⇒ exact modular GEMM only to ~2²⁴); libxc/VV10 nonlinear parts stay on the vector unit; Rys integrals do not map; full JAX/XLA rewrite needed. Tensor-train/THC/NTT stack: THC/ISDF for K as GEMMs, block-scaled quantization → RNS; NTT only helps true convolutions (grid Poisson); Amdahl: VV10 and AO evaluation become the critical path on matrix-heavy chips.
* VV10 as dense GEMMs: no exact low-rank form found (agent algebra in `RESULTS_vv10.md`); the pair sum stays O(N²) SIMT (FP32 symmetric kernel at ≈87–94 % FP pipe utilization). Laikov's v2 DF-VV10 is separable (7 bilinear terms) but changes the functional (≤18 % well-depth errors).
* Skala/ML functionals, second-order SCF solvers, ML initial guesses were reviewed and judged non-applicable for now.

---------------------------------------------------------------------------------------------------------------------------------

## 8. Risks, caveats, things that will bite

* Accuracy is set by moduli: coarse 8 (~1e-6), final 11 (~2e-10), DF build 15. Heavy elements (Z>36, ECP) need more final moduli (CH₃I 5e-9). Add a self-check (§7.4) before relying on 11 everywhere.
* `scheme='auto'` is a compute-capability table; on unusual GPUs measure first. UKS XC is FP64 (K is INT8). Open-shell/ROKS/ECP-gradient with crt only lightly tested (compat_test passed).
* `crt` disables the fused XC loops and uses the unfused block loop + fused `build_B`; the fused FP64/FP32 loops are used otherwise.
* The agent-written fused loops and sym-VV10 were validated on T4 (and parts on A100); multi-GPU untested. VV10 results are nondeterministic at the 1e-13 level.
* `coarse_grid` swaps `mf.grids`/`mf.nlcgrids` with copies; code relying on `mf.grids` object identity (e.g. grid-response gradients, solvent) was audited only by reasoning (solvent agent: compatible).
* The init-|g| shortcut `_accept_converged_init` is in `hf._kernel`; it is harmless but never triggers at final_grad_factor 0.1.
* Timing hygiene: no background compilation, `--warmup`, same grids/NLC grids when comparing; the coarse-stage default grids differ from the FP64 reference's only in the first stage.
* Do not run Colab sessions beyond two; do not leave sessions running (all sessions were stopped at the end of this session).
* Never push/PR without the user's go-ahead; do not retry the GEMMul8 clone without asking.

---------------------------------------------------------------------------------------------------------------------------------

## 9. Standing preferences from the user (carry these forward)
* Be autonomous, deep-profile, test every idea, report negatives; try parallel subagents for research/implementation (they have no GPU unless you give them one; give each a unique Colab session name and tell them about the 2-slot limit); merge their worktree branches yourself and resolve conflicts (the crt branches must take precedence over fused loops).
* Prefer exactness; opt-in approximations must be switchable and documented. Keep architecture-specific paths as options. Optimize for both non-FP64 GPUs (be at least as fast as the old A100 pipeline if possible) *and* A100/H100.
* Look into scaling with larger bases (QZVPP-D), compatibility (ECP, UKS, functionals, solvation) — first passes done (§3.5–3.7), deeper work open.

---------------------------------------------------------------------------------------------------------------------------------

## 10. Index of scripts (in `benchmarks/dft/`)
`bench_wb97mv.py` (main end-to-end bench; `--mixed auto --scheme crt|classic --grad --ref reference/bpa_tight.json --warmup --vv10 fp32 --crt_n N`) · `compat_test.py` (functional/spin/ECP/basis matrix) ·
`scale_basis.py` (basis scaling) · `ladder_test.py` (coarse-stage/modulus experiments, `--basis0` seed) · `defect_test.py` (dual-grid smoothness) · `fd_gradient_check.py` · `profile_xc_kernels.py` / `profile_k.py` / `profile_grad.py`
(torch-CUPTI kernel tables) · `noise_floor.py`, `xc_error_split.py` (FP32 noise-floor studies) · `crt_int8.py` + `crt_int8_test.py` (stand-alone CRT micro-benchmark with synthetic operands) · `ozaki_int8.py`, `gemm_precision_bench.py` (Ozaki-I, round 2) ·
`ab_merge.py` (fused/merged-GEMM A/B) · `bench_vv10.py`, `analyze_vv10_structure.py`, `tc_emul_vv10.py` · `bench_launch_overhead.py` · `bench_solvent.py`, `solvent_*_cpu*.py` · `colab/` helpers · `research/` reports · `reference/bpa_tight.json`.
