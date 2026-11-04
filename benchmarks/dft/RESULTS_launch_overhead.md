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

RESULTS_PLACEHOLDER
