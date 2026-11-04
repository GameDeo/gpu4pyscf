# Where we are / what to do next (parked while the theory discussion runs)

Branch: `claude/serene-meitner-kinyv4` (local only; push blocked until GitHub access is set up; do not open a PR unless asked).
Last commit at time of writing: "cuTENSOR plan cache, A100 results, plan status". See RESULTS_mixed_precision.md and
PLAN_stage2_algorithms.md for all numbers.

## Headline state
* Bisphenol A, wB97M-V/def2-TZVP, DF, default grids. T4: original 296.8 s (SCF 249.7 + grad 47.1) -> 104.3 s (SCF 87.0 + grad 17.3), dE 3.7e-10 Eh, max grad error 1.2e-7.
* A100 FP64: SCF 12.35 -> 10.95 s from the exact cuTENSOR plan cache. Mixed precision gives nothing on A100 (FP64 TC = FP32 SIMT); contract-first gradient gives nothing there either.
* T4 is ~7.8x slower than A100 FP64 now (was ~19.5x). Raw FP64 ratio is ~100x.

## Next steps, in order
1. **CRT (Ozaki-II) INT8 micro-benchmark on T4** (cheap, decides everything below).
   Own implementation (GEMMul8 clone was refused by the permission classifier; ask before trying again).
   8-12 coprime moduli <= 256, symmetric residues in [-p/2, p/2], integer-only fused quantize+mod kernel, INT8 GEMM
   via legacy `cublasGemmEx` TN (Turing needs m,n,k % 16 == 0), mod-reduce, Garner/Horner reconstruction.
   Operands: real BPA XC-vmat (K=61440) and K-build shapes (see gemm_precision_bench.py / ozaki_int8.py).
   Report accuracy vs moduli, GEMM time, quantize time, reconstruct time **together** (the Ozaki-I benchmark left slicing out:
   cupy slicer 334 ms vs 24 ms GEMM). Compare with Ozaki-I (28 products at 7 slices). Go/no-go: >= 5x native FP64 end to end at SCF-grade accuracy.
   Also add achieved-TFLOPS per kernel (FLOPs from shapes / profiler time) to profile_xc_kernels.py.
2. If go: integrate into FP64-stage XC (accumulate residues across grid blocks with one global fixed-point scale, reconstruct once per Vxc),
   then K (DF tensor is static: quantize + reduce once per SCF), then the gradient. Precision ladder = number of moduli (Huang et al. 2601.08077).
   Expected T4 gain: FP64 XC is ~22 s of the 87 s SCF; at 4-5x saves 16-18 s (SCF ~87 -> ~70 s). Not yet measured.
3. Launch-overhead reduction in the XC loops (helps A100/H100 most, exact): one batched GEMM for c = C^T [ao0..ao3], a fused MGGA rho kernel,
   fewer small kernels in the Vxc path (~100k python calls per XC evaluation, A100 GPU idle ~50 % of nr_rks).
4. VV10 symmetric-pair kernel (exact, ~1.5x fewer ops) and stream overlap of VV10 with XC/K.
5. Reduced-precision DF-tensor storage (2x memory), half-triangle K contraction.
6. TF32 3-split with outside-TC accumulation as a fused kernel (A100/H100 early cycles only).

## Findings to remember
* FP64 XC on T4 is 95 % GEMM; FP32-XC solution has FP64 |g| ~ 4e-5 (Vxc-matrix FP32 accumulation, error ~ linear in chunk length), so 2-3 FP64 Fock builds always remain.
* cuTENSOR "3xTF32" descriptor is plain FP32 SIMT. cuBLAS FP64 emulation: slower than native on A100, no-op on T4.
* final_grad_factor 0.1 is needed for gradient quality; 1.0 gives 1-2e-6 gradient error.
* Open knobs not defaulted: `range_separated_mode='mix_inside_kernel'` (halves K, dE ~1e-6 Eh), coarser NLC grid, Laikov DF-VV10 (changes the functional).

## Setup recipe (Colab CLI, `--auth adc`, credentials from env, never in repo)
`colab new -s t4 --gpu T4`; upload tgz of the repo; `pip install pyscf==2.14.0 gpu4pyscf-libxc-cuda12x==0.9.0 cutensor-cu13 basis-set-exchange pyscf-dispersion geometric`;
cmake `-DCUDA_ARCHITECTURES=75-real|80-real -DBUILD_LIBXC=OFF`; build targets gdft cupy_helper gvhf gecp gvhf_rys gvhf_md gint solvent sem pbc mgrid_v2 mgrid_v3 (~25 min on T4).
Runtime env: `PYTHONPATH=/content/src/gpu4pyscf CUPY_ACCELERATORS=cub,cutensor LD_LIBRARY_PATH=/usr/local/lib/python3.13/dist-packages/cutensor/lib:$LD_LIBRARY_PATH` (keep the existing LD_LIBRARY_PATH!).
Always `colab stop`. An L4 runtime on the account is the user's side project, not ours.

## Theory side-thread (no implementation planned)
* Ring-native / CRT view: GEMM chains, polynomial matrix functions (McWeeny purification, Newton-Schulz for inverse / S^-1/2) are ring-compatible; division, comparison,
  rescaling, libxc, VV10 exp/rsqrt, Rys integrals, and the O(N_grid^2) VV10 pair sum are not. On TPU the MXU takes bf16 (FP32 accumulate: exact modular GEMM only to ~2^24, short reductions)
  and, on recent generations, INT8 (INT32 accumulate, ~133k-term reductions): prefer INT8-capable generations. No FP64 anywhere on TPU.
* Tensor trains: QTT / tensor-structured Hartree-Fock (Khoromskij et al.) exists for small molecules; TT-cross handles nonlinear functions by sampling (pointwise evaluation of few points can be done at high precision on the vector unit);
  rank truncation needs SVD/QR (not ring-native). Pragmatic nearer-term relatives: THC / ISDF density fitting (K as pure GEMMs on grid points).
* Self-correcting iterations tolerate low precision up to a noise floor; one precise final evaluation fixes the energy (this is what the staged scheme already exploits).

### Revised TPU-oriented stack (theory only, nothing built or measured)
* Order is: THC/ISDF (K and Coulomb as GEMMs on grid points) -> global fixed-point scale -> residues mod INT8 moduli (RNS)
  -> INT8 GEMMs accumulated in the ring across blocks -> one CRT reconstruction per matrix. Nonlinear pieces (libxc, VV10 kernel,
  TT-cross sampling) are evaluated pointwise at higher precision on the vector unit.
* NTT is probably the wrong tool: it only accelerates true convolutions (e.g. grid Poisson), needs primes p = 1 mod 2n that do not fit INT8 moduli,
  and THC points are irregular. "Quantize then reduce mod p" (not NTT) is what maps reals to the ring.
* Block-scaled INT8 and RNS are one precision knob: ~8 bits per modulus, so the modulus count is the precision ladder (3-4 early cycles, 8-12 final).
* MXU fill: small TT cores underfill 128x128 tiles; THC ranks in the thousands are fine. TT is partly redundant with THC.
* Amdahl: on a matrix-unit-heavy chip VV10 (9 % of T4 SCF, 28 % of A100 SCF) and AO/libxc work become the critical path; recasting VV10 as
  GEMMs (low-rank / separable kernel) matters more than the ring arithmetic.
* First real tests are GPU-side: CRT micro-benchmark on T4, then THC-K with INT8 on T4.
