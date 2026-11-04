# Where we are / what to do next (parked while the theory discussion runs)

Branch: `claude/serene-meitner-kinyv4` (local only; push blocked until GitHub access is set up; do not open a PR unless asked).
Last commit at time of writing: "cuTENSOR plan cache, A100 results, plan status". See RESULTS_mixed_precision.md and
PLAN_stage2_algorithms.md for all numbers.

## Headline state (updated after round 3)
* Bisphenol A, wB97M-V/def2-TZVP, DF, default grids, T4: original 296.8 s (SCF 249.7 + grad 47.1) -> 70 s (SCF 48.7 + grad 21.6 first call, 16.8 steady),
  dE 1.4e-10 Eh vs a tight FP64 reference, max grad error 1.1e-7.  Scheme `mf.mixed_precision = {'scheme': 'crt'}` (auto on sm_75/86/87/89):
  coarse-grid stage (50,194) with 8 INT8 moduli to |g|<1e-3, then full grid, 11 moduli (XC, K), 15 moduli for the DF-tensor build,
  FP32 VV10 (symmetric tiled kernel), final_grad_factor 0.1.  Everything is in `gpu4pyscf/lib/crt_gemm.py` + hooks (see RESULTS round 3).
* A100 FP64 (old pipeline, before round 3): SCF 10.95 s + grad 2.5 s.  A100 has NOT been measured with: crt scheme, coarse-grid stage,
  symmetric VV10 kernel, fused XC loops (agent branches merged).  sm_80 libraries are being built on the T4 host (`libs80.tgz`).
* Merged agent work: VV10 symmetric tiled kernel (`dft/vv10_kernels.py`), fused XC loops (`dft/xc_fused.py`, `numint.XC_FUSED`; crt paths take
  precedence over the fused loops).  Research reports: scratchpad `research_report.md` (done), `research_report2.md` (running).

## Next steps, in order
1. A100 run of the new pipeline: classic (FP64) vs crt scheme, with/without the coarse-grid stage, fused loops, VV10 FP32 (decide 'auto' per GPU).
2. Research agent 2's report -> test the top exact / safe items (SCF solver / fewer Fock builds, grid reduction, K FLOP cuts, cheaper gradient).
3. Gradient (16.8 s steady on T4): DF J/K gradient ~8 s (FP64 Rys int3c2e_ip1 3.3 s + cuTENSOR DGEMMs), XC grad 5.6 s, VV10 grad 3.3 s.
4. XC (26 s of the SCF): quantize passes (fuse B build), recon, AO evaluation; symmetric tau-term GEMM halves; K static residue cache.
5. Arch-dispatched Ampere/Ada-only paths must be kept as options (user request): TF32 3-split, FP8, batched non-TN INT8 (Turing rejects it).

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
