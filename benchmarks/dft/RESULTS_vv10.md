# VV10 non-local correlation: pair-kernel speed-up

Code: `gpu4pyscf/dft/vv10_kernels.py` (cupy RawKernels), switch `numint.VV10_KERNEL`
(`'sym'` default, `'libgdft'` = original CUDA kernel) and `numint.VV10_PRECISION`
(`'fp64'` default, `'fp32'`).  Benchmark: `benchmarks/dft/bench_vv10.py`
(standalone, only needs cupy).  Test: `gpu4pyscf/dft/tests/test_vv10_kernels.py`.

Note on the baseline: `VV10_PRECISION` / `vxc_vv10nlc_fock_eval_UWE_fp32` are *not* on this
branch's parent commit (they live on the `claude/serene-meitner-kinyv4` family of branches).
"sib32" below is a verbatim copy of that FP32 kernel (hi/lo coordinates, 4 targets per thread,
chunk-32 FP32 sums flushed to FP64), so both the FP64 original and the existing FP32 kernel are
measured side by side.  The `VV10_PRECISION` name was reused so that the two lines merge cleanly.

## Workload

Real-molecule NLC grids, wB97M-V (b = 6.0, C = 0.01), `atom_grid=(50,194)` (pruned as in
`Grids` default), MINAO density, density threshold 1e-10, 6-31G:

| set | system | atoms | N (grid points kept) | pairs N^2 |
|-----|--------|-------|----------------------|-----------|
| w17 | 17 H2O cluster  | 51  | 303,046 | 9.2e10 |
| a24 | C24H50 zig-zag (74 bohr long) | 74 | 428,568 | 1.8e11 |
| w33 | 33 H2O cluster  | 99  | 576,944 | 3.3e11 |

## Operation count per pair

Original kernel (per ordered pair): 3 sub + 3 mul/fma (R^2), 2 fma (g_ij, g_ji), 1 add, 2 mul,
1 divide (FP64: ~10 DFMA-class instr.), then E: 1, U: 4, W: 1, 3 accumulations = about 20 + divide.
FP32 existing kernel (hi/lo coordinates): about 29 FP32 instr. + 1 MUFU rcp.

New kernel, algebra checked in `check_alg`-style test (1e-16): with a = g_ij, b = g_ji, s = a + b,
P = 1/(a b s), q = P^2:

* E_i += rw_j P,  U_i += rw_j q (b s + a b),  W_i += U_ij R^2;
* the *same* P, q, R^2, a, b, s give the column side:  E_j += rw_i P,  U_j += rw_i q (a s + a b)
  (Phi_ij = Phi_ji; only the per-point factors differ).

Shared work per unordered pair: 6 (R^2) + 2 + 1 + 1 + 1 + rcp + 1 (q) = 12 + rcp;
per side: t2 = fma(b,s,ab), rw*q, m, 3 accumulations = 6.  Total 24 instr. + 1 rcp per
unordered pair = 12.5 per ordered pair (vs ~20 + divide, or ~30 for the FP32 original).
The column sums are reduced with a register tile (8 columns x 3 quantities per thread),
a shuffle reduce-scatter over the 32 lanes (27 shuffles / 8 columns), shared-memory atomics
over the 4 warps and one FP64 atomicAdd per column and tile pair.  Pair-tile scheduling is
"circulant" (tile I with tiles I..I+n/2 mod n), so every block does the same work.

FP32 accuracy measures: coordinates are stored relative to the Morton-tile centre (tile of 512
points), so differences are accurate to 1e-7 of the *tile* size; MUFU rcp (no Newton step);
FP32 partial sums over <= 8 pairs, then <= 512 pairs, then FP64.  Max |a/b-1| of any single
point is 2e-6 (points with tiny weights); rho*w-weighted relative errors are 1e-7.

## Measured: A100-SXM4-40GB (Colab, CUDA 12.9 runtime, cupy 14.2), kernel time only

Times are best of 3 (ref64: 1 run) of the O(N^2) U/W/E evaluation; pairs/s = N^2/t (ordered pairs,
also for the symmetric kernel).  Variants: `ref64` = verbatim `vv10_fock_eval_UWE_kernel`;
`sib32` = existing FP32 kernel; `v1_*` = new tiled one-sided kernel; `sym_*` = new symmetric
kernel.  Errors are against `ref64`; "w-rel" are rho*w weighted relative L1 errors, E_nl is
the NLC energy.

C24H50 (a24, N = 428,568), all rows from one run with the final kernel:

| variant | time (s) | pairs/s | speed-up vs ref64 | rel. err E_nl | w-rel err v0 / v1 | max pointwise rel. E |
|---|---|---|---|---|---|---|
| ref64 (libgdft FP64) | 1.006 | 1.8e11 | 1.0 | - | - | - |
| sib32 (existing FP32) | 0.523 | 3.5e11 | 1.9 | 5e-10 | 2.0e-8 / 3.7e-8 | 1.8e-7 |
| **sym_64** (new, exact) | 0.674 | 2.7e11 | **1.49** | 3e-14 | 7e-14 / 8e-14 | 5e-13 |
| **sym_32** (new FP32) | 0.277 | 6.6e11 | **3.6** (1.9 vs sib32) | 8e-10 | 2.2e-8 / 3.7e-8 | 3.8e-7 |

Other sizes (A100; ref64 and sib32 are the same code in all runs; the `v1`/`sym` rows of the first
run used the kernel *before* the final fma(b,s,ab) op-count reduction, the sym rows of the second
w17 run are after it):

| set | ref64 | sib32 | v1_64 | v1_32 | sym_64 | sym_32 |
|---|---|---|---|---|---|---|
| w17, N=303k (pre-reduction) | 0.503 | 0.280 | 0.553 | 0.215 | 0.353 | 0.151 |
| w17, N=303k (final)         | 0.503 | 0.280 | -     | 0.206 | 0.331 | 0.137 |
| w33, N=577k (pre-reduction) | 1.760 | 0.960 | 1.879 | 0.733 | 1.290 | 0.546 |

(w33 with the final kernel was not re-run.)  The one-sided FP64 tiled kernel (`v1_64`) is *slower*
than libgdft; all of the FP64 gain comes from evaluating each unordered pair once.

**Achieved rate vs peak.**  A100 peaks: FP64 9.7, FP32 19.5 TFLOPS = 4.85 / 9.75e12 FMA-pipe
instructions/s (FMA = 1 instruction).  Executed FP-pipe instructions per ordered pair, counted
from the source (the DP divide is taken as ~8 DFMA; MUFU, shuffles and LDS use other pipes):
ref64 ~28, sib32 ~29, sym_64 ~16.8, sym_32 ~12.8.  Resulting FMA-pipe utilisation:
sym_64 2.7e11 x 16.8 / 4.85e12 ~ 94 %, sym_32 6.6e11 x 12.8 / 9.75e12 ~ 87 %.  (The same estimate
gives >100 % for ref64/sib32, i.e. the instruction counts of the originals are only accurate to
~20 %; the useful statement is that all kernels are pipe-bound.)  Both new kernels are therefore
issue/pipe bound, not memory bound (tiles live in registers and shared memory; no N x N
traffic), and further speed-ups can only come from fewer instructions per pair.  The earlier
"nominal" counting that charges the symmetric kernel the one-sided op count would give 2x the
executed rate and is not used.  A `(NT, MI)` sweep over 32..256 threads x 2..8 rows per thread
changes the time by < 10 % (default 128 x 4 is within noise of the best); MI = 8 does not help.

**Real molecule, `nr_nlc_vxc`** (gpu4pyscf 1.9.0 wheel + this branch's numint.py/vv10_kernels.py;
12 waters, def2-SVP, wB97M-V, NLC grid (50,194), N = 237,312; converged density; warm calls):

| kernel | nr_nlc_vxc wall (s) | E_nl | E_nl - E_nl(libgdft) | max abs diff of V_nl (max abs V_nl = 4.7e-3) |
|---|---|---|---|---|
| libgdft FP64 | 0.349 | 0.500715553294 | - | - |
| sym FP64 | 0.253 | 0.500715553294 | -2.6e-15 | 1.6e-16 |
| sym FP32 | 0.154 | 0.500715552983 | -3.1e-10 | 1.9e-11 |

Total SCF energies (10 cycles each, conv_tol 1e-9): libgdft -915.8715121014, sym-FP64 -915.8715121014
(diff 2e-13), sym-FP32 -915.8715121017 (diff -3.1e-10 Eh, with no FP64 polish at all).  (The SCF
wall times printed by that script, 24 s vs 9.9 s, are dominated by JIT/warm-up of the first run and
are not a speed-up measurement.)  `nr_nlc_vxc` also contains the density evaluation, the Vxc
contraction and (new) the Morton sort/padding of the grid (~45 % of the FP32 call at this size),
which is why the end-to-end gain (0.349 -> 0.154 s) is smaller than the kernel-only gain.

The unit test (`dft/tests/test_vv10_kernels.py`, 4 tests: random inputs with ragged tile counts and
several (NT, MI), FP32, distance cutoff, `_vv10nlc` switch on a real grid) passes on the A100.

## T4

T4_SECTION

## What was analysed and did not pay off

**Distance-cutoff screening** (`rcut`, tile pairs beyond rcut skipped; implemented and measured).
Numpy study on the real grids (`analyze_vv10_structure.py`), error of dropping all pairs with R > Rc,
rho*w-weighted mean relative error of the point sums (E / U / W):

| Rc (bohr) | w33 (37 bohr across): E, U, W | a24 (74 bohr long chain): E, U, W | pairs kept, w33 / a24 |
|---|---|---|---|
| 15 | 1.1e-3, 7.7e-5, 3.3e-3 | 5.1e-4, 3.9e-5, 1.4e-3 | 63 % / 42 % |
| 20 | 5.0e-5, 2.3e-6, 1.5e-4 | 1.1e-4, 4.9e-6, 3.2e-4 | 91 % / 55 % |
| 25 | 7.8e-8, 2.7e-9, 1.7e-7 | 3.0e-5, 9.0e-7, 8.8e-5 | 99 % / 65 % |
| 30 | 5e-12, 1e-13, 6e-12 | 9.7e-6, 2.1e-7, 2.9e-5 | 100 % / 75 % |
| 40 | 0 | 1.2e-6, 1.6e-8, 3.7e-6 | 100 % / 89 % |

(The error is *relative to the point sum*; individual low-weight points are far worse: max errors
of 1e-3..1e-1 at Rc = 15-25.)  On the compact 99-atom cluster a 1e-7 cutoff removes 1 % of the
pairs, on the 74 bohr chain Rc = 60 bohr is needed for 1e-9.  On the A100 the real kernels agree:
a24 rcut=40: 0.261 s vs 0.277 s (-6 %), E_nl error 1.6e-7; rcut=25: -22 % time but E_nl error
3.9e-6 and V errors 1e-5; w33 rcut=25: no speed-up.  Cutoff screening is not useful at the required
accuracy (the option is kept, default off, `rcut=0`).

**Gaussian / exponential sums, frequency integral.**  The (2/pi) int du 1/((a^2+u^2)(b^2+u^2))
form (or 1/x = sum_k c_k exp(-a_k x)) separates the i- and j-parameters only if the point
parameters are binned, and R^2 remains in every term: each of the K ~ 10-30 (or K^2..K^3 for the
u, t, t' nodes) terms is a full N-body Gauss transform exp(-tau |r_i - r_j|^2), whose GEMM form
exp(2 tau r_i.r_j) is not low rank.  Several N^2 Gauss transforms are more expensive than the one
direct N^2 pass (about 12 instructions per pair), so this was analysed on paper only.

**Far-field low rank (SVD ranks measured, `analyze_vv10_structure.py`).**  Two well separated
clusters of box side L, centre distance D, <= 1500 points each on the real grid: numerical rank
(relative Frobenius error 1e-3 / 1e-6 / 1e-8) of the E-kernel block:

| w33 | rank (1e-3, 1e-6, 1e-8) |
|---|---|
| L=4, D=12 | 12, 59, 126 |
| L=4, D=20 | 6, 33, 67 |
| L=4, D=30 | 4, 17, 35 |

(a24: 12/56/121, 9/39/80, 6/30/60; the U and W blocks have the same ranks within 10 %.)  So the
far field *is* low rank (rank ~30-60 at 1e-6 versus block size 1500) in spite of the per-point
omega, kappa.  But (i) these are SVD-optimal ranks; ACA/Chebyshev/ID constructions need ~1.5-3x
more; (ii) only pairs with R >~ 3 L qualify: for the 37 bohr cluster 42 % of all pairs are within
12 bohr (28 % within 10 bohr), so even a perfect H-matrix is capped at roughly 1/(0.42 + 0.58/6)
~ 2x over the direct sum at N = 0.3-0.6 M, before the cost of building the skeleton factors
(sequential pivoted cross-approximation on an irregular kernel) and the precision loss of
FP16/TF32 factors; (iii) the direct symmetric FP32 kernel already runs at ~90 % of the FP32 pipe.
A hierarchical far-field / tensor-core algorithm would only win for N >~ 1-2 M grid points.
**No tensor-core path was built.**

**Tensor cores for the pair kernel itself.**  R^2 = |xi|^2 + |xj|^2 - 2 xi.xj, and also
a_ij = omega_i R^2 + kappa_i and b_ij, are bilinear forms with K = 5, i.e. they can be produced by
m16n8k8 TF32 (or FP16 k16) MMAs whose accumulator fragments feed the element-wise code directly.
`tc_emul_vv10.py` emulates the operand rounding on 512 x 512 tile pairs of w33 (tile-local
coordinates): a single TF32/FP16 MMA gives R^2 errors of 0.1-0.7 bohr^2 and 2e-4..6e-4 errors of
the tile sums (useless); the 3-term hi/lo split (3 MMAs) gives 4e-8..1e-7, i.e. FP32 quality.  But
R^2 is only 6 of the 24 instructions per unordered pair (a, b would add 4 more); the reciprocal, the
products and the E/U/W accumulations are element-wise and non-linear.  Best case a 20-30 %
instruction reduction, at the price of register pressure and an MMA fragment layout that conflicts
with the row-register tiling.  Not built.

**Other things that did not work / were not worth it.**
* One-sided tiled FP64 kernel (`v1_64`): slower than libgdft (0.553 vs 0.503 s on w17).
* FP64 reciprocal as MUFU.RCP(float) + 2 Newton steps instead of `1.0/x`: only 4 % faster on A100
  (0.317 vs 0.331 s on w17); removed again.
* `(NT, MI)` register-tile sweep: flat (see above).
* FP16x2 (HFMA2) element-wise math: 3 decimal digits and range 6e4; not safe for a*b*s.

## Overlapping VV10 with other work (streams): description only, nothing implemented

In `rks.get_veff` the sequence is `nr_rks` (XC on the main grids) -> `nr_nlc_vxc` (VV10 on the
NLC grids) -> `get_j` -> `get_k`.  The NLC branch depends only on `dm` (density on the NLC grid,
the pair sums, the Vxc contraction), not on J or K.  It can therefore be issued on a second
`cupy.cuda.Stream` (or from a Python thread with its own stream, as the multi-GPU code already
does) while the J/K builds run on the default stream; the only join is the final `vxc += vnlc`.
Since the pair kernel is compute-bound at ~90 % of the FP pipe, the gain is limited to hiding the
small latency-bound kernels around it (mask, gather, Morton sort, density and Vxc contractions,
launches: ~45 % of an FP32 `nr_nlc_vxc` at N ~ 2e5, see above) and the K-build phases that do not
fill the SMs; it does not make the O(N^2) phase itself faster.  Temporaries of the NLC path are
O(N) (an N x N FP32 matrix at N = 3e5 would be 360 GB, so any scheme that materialises the pair
matrix in HBM loses).

## Caveats

* `sym` results are not bitwise reproducible run to run (FP64 `atomicAdd` ordering): differences
  are ~1e-15 relative.
* The `sym` kernels need NVRTC (cupy) and `__shfl_xor_sync`; gpu4pyscf has no ROCm path, so no HIP
  guard was added.  `numint.VV10_KERNEL = 'libgdft'` restores the original path.
* The gradient/Hessian VV10 kernels (`grad/rks.py`, `hessian/rks.py`) still call libgdft directly.
* FP32 (`VV10_PRECISION='fp32'`): omega is clipped at 1e9 (the pre-existing FP32 kernel clips the
  g's at 1e12).  The A100 was used once for ~15 minutes (session `a100v`, stopped).
