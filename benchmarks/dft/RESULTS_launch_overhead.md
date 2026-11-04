# Launch-overhead reduction in the XC grid loops

Branch base: `claude/serene-meitner-kinyv4` @ d8efa52 (the worktree was created from master, which lacks the
benchmarks/mixed-precision files referenced by the task, so the worktree branch was reset onto d8efa52 before any
change).  All numbers below are marked MEASURED (T4, own session) or UNVERIFIED (expectation, not run).

## 1. Catalogue of Python-level launches per 4096-point grid block (before)

`numint.nr_rks`, RKS, MO-based density (the SCF path), FP64, block touching all AOs.  One `nr_rks` makes two
block loops (rho, then Vxc); each loop iteration also evaluates the AOs (`block_loop`).

| stage | op | launches / block (MGGA) | avoidable |
|---|---|---|---|
| block iterator (x2 loops) | `cupy.asarray(coords, order='F')` copy | 1 | yes, cached once per grids |
| | GDFTeval_gto (AO values) | 1 | no (C++ kernel) |
| rho | `cupy.take(mo_coeff, idx)` | 1 | yes when the block touches all AOs (idx = identity) |
| | `cupy.dot(cpos^T, ao_x)` x4 | 4 | one batched contraction |
| | `GDFTcontract_rho` x7 + 3 `rho[i]=` copies + 3 `rho[tau]+=` adds + fill + 2 scalings + final `rho_tot[:,p0:p1]=` | 17 | one fused kernel writing into `rho_tot` |
| Vxc | `.5*wv` (tau weight) | 1 | folded into the global weight scaling |
| | 3 x (`cupy.multiply(ao_x, w_tau)` + cuTENSOR GEMM) | 6 | one build kernel + one GEMM of length 4*blk |
| | `cupy.asarray(wv[:4,p0:p1], order='C')` (non-contiguous slice copy) + GDFTscale_ao | 2 | the build kernel reads wv with a stride |
| | cuTENSOR GEMM (GGA part) | 1 | merged |
| | `add_sparse(vmat, vtmp, idx)` | 1 | GEMM accumulates (beta=1) into vmat when idx is the identity |
| **total** | | **37** (2 of them are the AO kernel) | |

GGA: 2 (AO) + rho 1 + 3x`dot`+... see the measured counts below; LDA is a subset.  Host-side cost per launch
is ~15-50 us (cupy ufunc/`dot` dispatch is 20-60 Python calls, the cuTENSOR wrapper had ~130 us before the plan
cache), which is where ~131k Python calls per `nr_rks` (2 x 260 blocks x ~250) come from.

Other per-block loops with the same pattern (same helpers now used where noted):

* `nr_uks`: rho and Vxc loops, per spin and per dm set: fused path added (spins share the AO evaluation).
* `nr_nlc_vxc` (NLC grid, GGA rho): fused path added.
* `grad.rks._get_exc_task` (XC gradient): libxc (`eval_xc_eff`, ~30 launches + Python) was called **inside** the
  block loop; `_exc1_block` issued ~45 launches per block for MGGA (multiply+sum chains, fancy-index gathers, 3
  tau applications of D).  Now: rho pass -> one libxc call -> block loop of ~12 launches.
* `nr_rks_fxc`, `nr_uks_fxc`, `cache_xc_kernel`, `get_rho`, hessian XC: same style, **not changed** (not in the
  SCF/gradient hot path of this task).
* `df_jk.get_jk` / `df.loop`: NOT launch-bound.  Per aux block (blksize chosen from 40 % of free memory, i.e.
  tens of blocks for a 600-AO molecule) it issues 2 (J) + 2-3 (K) cuTENSOR contractions of large GEMMs plus the
  asynchronous host->device copy of the next cderi block; there is no per-point or per-small-block loop.  No change
  except that every `contract` call now has less host overhead.  The K path is GEMM-bound (see RESULTS_mixed_precision.md).

## 2. What was implemented

(see git log; every item is an exact refactor, results agree to rounding)

1. `dft/xc_fused.py` (NVRTC kernels via `cupy.RawModule`, no extension rebuild):
   * `rho_from_c`: rho, nabla rho, tau from `c = C^T [ao_0..ao_3]` in one kernel (replaces 17 launches).
   * `build_B`: weighted AO operands.  For MGGA: `B_0 = sum_x w_x ao_x`, `B_x = w_tau ao_x`.
   * `row_dot3`: `out[i,n] += sum_g A[comp(n),i,g] Y[i,g]` for the gradient (replaces multiply + sum chains).
2. `numint._eval_rho_blocks_mo` / `_vmat_blocks_fused`: one batched contraction `ki,nkg->nig` (4 components, shared
   C) and one kernel per block for rho; one build kernel and one contraction `xig,xjg->ij` (K = 4*blk) per block for
   Vxc, accumulated directly into `vmat` by the GEMM (beta=1) when the block touches all AOs.  Work buffers are
   allocated once per call (zero-initialised: cuTENSOR evaluates `alpha*AB + beta*C` also for beta=0).
3. `_block_loop`: F-ordered coordinates for all full blocks produced by one transposing copy and cached on the grids
   (per device); `Grids.get_non0ao_idx` caches its per-block views (was ~800 cupy view creations per call).
4. `lib/cutensor.py`: pattern parsing, alpha/beta scalars and the empty workspace cached; no dict/array building on a
   plan-cache hit.
5. `grad/rks.py`: rho for all grids -> one libxc call -> block loop with `build_B`/`row_dot3` (see above).
6. Switches: `numint.XC_FUSED` (default True; False runs the original loops), `numint.XC_MERGE_VMAT_GEMM`.
   `XC_VMAT_CHUNK > 0` with FP32 uses the original loop.

Not done (and why):

* CUDA graphs: the GEMM shapes change with `nao_sub` from block to block, so a graph would have to be captured per
  block anyway; no gain.
* Multi-stream/multi-thread overlap of blocks: the blocks accumulate into one matrix; overlapping would need one
  accumulator per stream and a fixed-order reduction to stay deterministic.  After the fusion the loop is
  GPU-bound on the A100 estimate below, so it is not needed.  (UNVERIFIED)
* Larger grid blocks (`min_grid_blksize`): the sparse-AO machinery is built per block; not changed.

## 3. Results (MEASURED on a Tesla T4, own session; Bisphenol A, wB97M-V, def2-TZVP, (99,590) grid = 1.06 M points, 623 AO)

`bench_launch_overhead.py`, fixed density from a 2-cycle SCF, best of 2 after warm-up.  "old" = the pristine tree
(d8efa52, same built libraries); "new" = this branch with `numint.XC_FUSED=True`.  Python calls = cProfile
`total_calls`; kernels / GPU busy = torch profiler (all CUDA kernels, including those of cuBLAS/cuTENSOR/gdft).

| call | wall old -> new | Python calls old -> new | CUDA kernels old -> new | GPU busy old -> new | max rel. diff new vs legacy loops |
|---|---|---|---|---|---|
| `nr_rks`, FP64 | 8.47 -> 7.71 s (1.10x) | 106,607 -> 33,364 | 15,306 -> 5,989 | 8.57 -> 7.66 s | 2.8e-15 |
| `nr_rks`, FP32 contractions | 1.53 -> 1.04 s (1.47x) | 139,635 -> 34,938 | 19,472 -> 6,845 | 1.25 -> 1.03 s | 1.5e-6 (FP32 rounding, different accumulation) |
| `nr_nlc_vxc`, FP64 | 8.62 -> 8.55 s | 15,727 -> 7,381 | 3,140 -> 1,325 | 8.57 -> 8.55 s | 2.0e-15 |
| `grad.rks.get_exc` | 6.04 -> 4.93 s (1.23x) | 169,548 -> 77,677 | 29,799 -> 9,349 | 5.70 -> 4.99 s | 9.9e-15 |

Same molecule, def2-SVP (launch-bound regime, 1.06 M points): `nr_rks` FP64 3.31 -> 2.78 s (1.19x),
FP32 0.90 -> 0.52 s (1.74x); Python calls 106,596 -> 33,353.

Reading the table:

* **FP32 path (the T4 idle-gap problem from RESULTS_mixed_precision.md): idle time 0.28 s -> 0.005 s.** Wall time now equals
  GPU busy time (1.038 vs 1.034 s).  What is left is GPU work: cuTENSOR/cutlass SGEMM 0.35 s, AO evaluation ~0.30 s, the
  FP64->FP32 AO cast 0.15 s (2 per block, one in each loop; fusing it into the AO kernel needs a gdft change, not done),
  `build_B` 0.05 s.
* **FP64 on T4 is GEMM-bound (GPU busy ~ wall in both versions), so the launch savings do not show; the 10 % gain comes from the
  merged GEMM**: `xig,xjg->ij` with K = 4*4096 runs as ONE cuTENSOR/cutlass kernel (260 launches, 6.13 s) instead of four
  cuBLAS `volta_dgemm_64x64_tn` per block (1040 launches, 6.99 s).  With `XC_MERGE_VMAT_GEMM=False` the fused loop takes
  8.40 s (= old).  Longer K helps here; whether it helps on an A100 is UNVERIFIED (flag provided).
* 4,060 of the 5,989 remaining kernels are the gdft AO-evaluation kernels (one launch per contraction pattern and
  angular momentum, inside `GDFTeval_gto`); the Python-visible launches are now ~7 per block and loop pair
  (2 AO, 1 rho GEMM, 1 rho kernel, 1 build kernel, 1 Vxc GEMM, +take/+add_sparse for blocks that screen AOs) versus 37.
* Host profile of the new FP64 `nr_rks` (cProfile, def2-SVP, GPU-bound on the T4, so launch-queue waits are included in the
  callers' time): `eval_ao` 1.02 s tottime over 520 calls (the AO launches block when the queue is full),
  cuTENSOR `contract` 0.27 s / 520 calls, `add_sparse` 0.23 s / 260, `build_B` 0.08 s / 260, libxc 0.03 s;
  nothing else above 0.03 s.  33k Python calls in total = ~64 per block and loop (was ~205).
* The XC gradient launched 115 kernels per block (29,799 / 260), now 36 (9,349 / 260), of which ~16 are AO-evaluation kernels.

Correctness (T4, all pass): `dft/tests/test_xc_fused.py` (18 tests: fused vs original loops for LDA/GGA/MGGA, MO and
density-matrix rho paths, a two-fragment molecule whose blocks screen AOs, UKS, NLC grid, FP32, gradient LDA/GGA/MGGA incl. the
dense-dm branch and sparse blocks, determinism, kernel-vs-numpy), `test_numint.py` (41), `test_numint2c.py`, `test_xc_fp32.py`,
`test_vv10_fp32.py`, `test_cutensor_plan_cache.py`, `test_exc_contract_first.py`, `test_grids.py`, `test_cupy_helper.py`,
`grad/tests/test_rks_grad.py` (17), `test_uks_grad.py` (12), `dft/tests/test_rks.py` (17), `test_uks.py` (10), `test_rks_lowmem.py` (4), `df/tests/test_df_k_fp32.py`, `test_df_jk.py`, `grad/tests/test_vv10_grid.py`, `test_grid_response.py` (7), `test_lowmem_grad.py` (4).  Multi-GPU was not tested (single T4).

## 4. Expectations NOT measured (A100/H100) -- UNVERIFIED

* A100 FP64 `nr_rks` (0.489 s wall, 0.245 s GPU busy before; 0.43 s after the cuTENSOR plan cache): the T4 FP32 case
  (same launch-bound structure: wall 1.45 s -> 1.04 s = GPU busy, idle gaps removed) suggests the wall time drops to roughly
  the GPU busy time, i.e. ~0.25-0.30 s (1.5-1.9x), and a full SCF gets ~10-15 % faster (XC main is ~45 % of the SCF).  Not run.
* The merged K = 4*blk GEMM may or may not beat 4 separate cuBLAS FP64 tensor-core GEMMs on an A100; A/B with
  `numint.XC_MERGE_VMAT_GEMM` before relying on it.
* XC gradient on A100 (2.4 s total for BPA): fewer launches per block (115 -> 36, measured above); the wall gain depends on
  the idle share there, not measured.

## 5. How to reproduce

```
python benchmarks/dft/bench_launch_overhead.py --basis def2-tzvp --repeat 2     # old/new rows (XC_FUSED False/True)
python -m pytest gpu4pyscf/dft/tests/test_xc_fused.py
```
On a tree without the fused loops the script measures the original loops only (use it for the "old" column).
Note that `XC_FUSED=False` in the new tree is not the pristine baseline: it still has the cuTENSOR host-path trimming and
the cached grid-block views (Python calls 106.6k -> 97.6k for FP64 `nr_rks`).
