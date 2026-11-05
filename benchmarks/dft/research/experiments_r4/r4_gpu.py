#!/usr/bin/env python
'''Round-4 GPU measurements (T4), BPA wB97M-V/def2-TZVP + DF, crt scheme.

R1  block statistics of the XC grids: sum nao_sub^2*ng (vmat work) and sum nao_sub*ng (V*C_occ work) per grid level
R2  unit costs at the converged density: fine/coarse XC, NLC, J/K (FR and LR), eigh
R3  prototype "occupied-only" XC: V*C_o = sum_k ao_k Y_k^T (Y from the half-transformed orbitals that rho already needs)
    vs the production per-block vmat (build_B + INT8 CRT GEMM); correctness vs FP64 V @ C
R4  error-bounded storage of the DF tensor: uniform absolute quantization step s -> dE_K, and FP32 storage
R5  bit statistics of the DF tensor (bits per element needed for an absolute error eps, per-element and block-16 exponents)
R6  int8 GEMM throughput: cuBLAS gemmEx (strided batched) vs torch._int_mm on XC / V*C_o / K shapes
Writes r4_gpu.json next to this script.
'''
import os, sys, time, json
import numpy as np
import cupy

OUT = {}
here = os.path.dirname(os.path.abspath(__file__))
molfile = os.path.join(here, '..', '..', '..', 'molecules', 'organic', '033_Bisphenol_A.xyz')


def sync():
    cupy.cuda.Stream.null.synchronize()


def tm(f, n=1):
    sync()
    t = time.perf_counter()
    for _ in range(n):
        r = f()
    sync()
    return (time.perf_counter() - t) / n, r


def save():
    with open(os.path.join(here, 'r4_gpu.json'), 'w') as fh:
        json.dump(OUT, fh, indent=1, default=float)


def log(*a):
    print(*a, flush=True)


from pyscf import gto
from gpu4pyscf import dft
from gpu4pyscf.dft import numint, gen_grid
from gpu4pyscf.scf import mixed_precision as mp
from gpu4pyscf.lib import crt_gemm
from gpu4pyscf.lib.cupy_helper import tag_array

# ---------------------------------------------------------------- warm-up (JIT)
wm = gto.M(atom='O 0 0 0; H 0 .76 .59; H 0 -.76 .59', basis='def2-tzvp', verbose=0)
w = dft.RKS(wm, xc='wb97m-v').density_fit()
w.grids.atom_grid = (20, 50); w.nlcgrids.atom_grid = (10, 26)
w.mixed_precision = {'scheme': 'crt'}
w.kernel()

mol = gto.M(atom=molfile, basis='def2-tzvp', verbose=0, max_memory=14000)
mf = dft.RKS(mol, xc='wb97m-v').density_fit()
mf.grids.atom_grid = (99, 590)
mf.nlcgrids.atom_grid = (50, 194)
mf.mixed_precision = {'scheme': 'crt'}
mf.conv_tol = 1e-9
t, _ = tm(mf.kernel)
OUT['scf'] = {'time': t, 'e': float(mf.e_tot), 'nao': mol.nao, 'nocc': int((mf.mo_occ > 0).sum())}
log('SCF', OUT['scf'])
save()

ni = mf._numint
dm = mf.make_rdm1()
mo = mf.mo_coeff
occ = mf.mo_occ
dm = tag_array(dm, mo_coeff=mo, mo_occ=occ)
nocc = int((occ > 0).sum())

# ---------------------------------------------------------------- R1 block statistics
def mkgrid(ag):
    g = gen_grid.Grids(mol)
    g.atom_grid = ag
    g.build(with_non0tab=False, sort_grids=True)
    return g

grid_objs = {}
R1 = {}
for ag in [(99, 590), (75, 302), (50, 194), (30, 110), (20, 50)]:
    g = mf.grids if ag == (99, 590) else mkgrid(ag)
    grid_objs[ag] = g
    if ni.gdftopt is None:
        ni.build(mol, g.coords)
    idx = g.get_non0ao_idx(ni.gdftopt)
    ns = np.array([len(x[1]) for x in idx], dtype=np.float64)
    nb = len(ns)
    ng = g.size
    blk = numint.MIN_BLK_SIZE
    R1[str(ag)] = dict(ngrids=int(ng), nblocks=nb, nao_sub_mean=float(ns.mean()),
                       nao_sub_p10=float(np.percentile(ns, 10)), nao_sub_p50=float(np.percentile(ns, 50)),
                       nao_sub_p90=float(np.percentile(ns, 90)),
                       sum_n2g=float((ns ** 2).sum() * blk), sum_ng=float(ns.sum() * blk),
                       vmat_TF_mgga=float(8 * (ns ** 2).sum() * blk / 1e12),
                       vco_TF_mgga=float(2 * 8 * ns.sum() * blk * nocc / 1e12))
    log('R1', ag, R1[str(ag)])
OUT['R1'] = R1
save()

# ---------------------------------------------------------------- R2 unit costs (final-stage precision)
R2 = {}
try:
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
except Exception:
    omega, alpha, hyb = 0.3, 1.0, 0.15
R2['rsh'] = [float(omega), float(alpha), float(hyb)]
with mp.final_stage(mf):
    for ag in [(99, 590), (50, 194), (20, 50)]:
        g = grid_objs[ag]
        t, r = tm(lambda: ni.nr_rks(mol, g, mf.xc, dm))
        t, r = tm(lambda: ni.nr_rks(mol, g, mf.xc, dm))
        R2['nr_rks_%d_%d' % ag] = t
        log('nr_rks', ag, t)
    for ag in [(50, 194), (30, 110), (10, 26)]:
        g = gen_grid.Grids(mol); g.atom_grid = ag; g.build(with_non0tab=False, sort_grids=True)
        t, _ = tm(lambda: ni.nr_nlc_vxc(mol, g, mf.xc, dm))
        t, _ = tm(lambda: ni.nr_nlc_vxc(mol, g, mf.xc, dm))
        R2['nlc_%d_%d' % ag] = t
        R2['nlc_ngrids_%d_%d' % ag] = int(g.size)
        log('nlc', ag, t)
    wdf = mf.with_df
    t, _ = tm(lambda: wdf.get_jk(dm, hermi=1, with_j=True, with_k=False)); R2['J'] = t
    t, _ = tm(lambda: wdf.get_jk(dm, hermi=1, with_j=False, with_k=True)); R2['K_FR'] = t
    t, _ = tm(lambda: wdf.get_jk(dm, hermi=1, with_j=False, with_k=True, omega=omega)); R2['K_LR'] = t
    t, _ = tm(lambda: mf.get_veff(mol, dm)); R2['get_veff_total'] = t
    h1 = mf.get_hcore(); s1 = mf.get_ovlp()
    fock = mf.get_fock(h1, s1, mf.get_veff(mol, dm), dm)
    t, _ = tm(lambda: mf.eig(fock, s1)); R2['eigh'] = t
    log('R2', R2)
OUT['R2'] = R2
save()

# ---------------------------------------------------------------- R3 occupied-only XC prototype
R3 = {'t_vmat_prod': 0., 't_rho_prod': 0., 't_vco_gemm': 0., 't_vco_Y': 0., 'maxerr_rel': 0.}
g = mf.grids
opt = ni.gdftopt
_sorted_mol = opt._sorted_mol
cpos = opt.sort_orbitals(mo[:, occ > 0] * np.sqrt(occ[occ > 0]), axis=[0])
nmod = 11
rng = cupy.random.default_rng(1)
p0 = 0
nblk_checked = 0
for ao, idx, weight, _ in ni.block_loop(_sorted_mol, g, mol.nao, 1, max_memory=None):
    ng = weight.shape[0]
    nsub = ao.shape[1]
    C = cupy.ascontiguousarray(cpos[idx])
    # production pieces
    t, c4 = tm(lambda: numint._eval_rho2_crt(ao, C, 'MGGA'))
    R3['t_rho_prod'] += t
    wv = rng.standard_normal((5, ng)) * weight
    out = cupy.empty((nsub, nsub))
    t, _ = tm(lambda: numint._vmat_block_crt(ao, wv, 0, 'MGGA', out))
    R3['t_vmat_prod'] += t
    # half-transformed orbitals psi_k (4, ng, nocc) in FP64 (in a real kernel they come out of the rho GEMM)
    psi = cupy.einsum('kig,ia->kga', ao[:4], C)

    def buildY():
        Y = cupy.empty((4, C.shape[1], ng))
        Y[0] = (wv[0][:, None] * psi[0] + wv[1][:, None] * psi[1] + wv[2][:, None] * psi[2]
                + wv[3][:, None] * psi[3]).T
        for x in range(1, 4):
            Y[x] = (wv[x][:, None] * psi[0] + .5 * wv[4][:, None] * psi[x]).T
        return Y
    t, Y = tm(buildY)
    R3['t_vco_Y'] += t
    a4 = cupy.ascontiguousarray(ao[:4])
    t, VC = tm(lambda: crt_gemm.gemm_nt(a4, Y, nmod))
    R3['t_vco_gemm'] += t
    if nblk_checked < 6 and nsub > 0:
        V = ao[0] @ (wv[0] * ao[0]).T
        for x in range(1, 4):
            V += ao[x] @ (wv[x] * ao[0]).T + ao[0] @ (wv[x] * ao[x]).T + .5 * ao[x] @ (wv[4] * ao[x]).T
        ref = V @ C
        err = float(abs(VC - ref).max() / abs(ref).max())
        R3['maxerr_rel'] = max(R3['maxerr_rel'], err)
        nblk_checked += 1
    p0 += ng
log('R3', R3)
OUT['R3'] = R3
save()

# ---------------------------------------------------------------- R4/R5 DF tensor storage
wdf = mf.with_df
cd = wdf._cderi[0]
OUT['cderi_shape'] = list(cd.shape)
dm0 = cupy.asarray(dm)
vk0 = wdf.get_jk(dm, hermi=1, with_j=False, with_k=True)[1]
R4 = {}
backup = cupy.asnumpy(cd)          # host copy (3 GB) to restore after each test


def restore():
    cd.set(backup)


def ek_shift(vk):
    return float(-.25 * cupy.einsum('ij,ji->', dm0, vk - vk0))     # dE of 1.0*K (FR) at fixed D


for s in [1e-8, 1e-9, 1e-10, 1e-11]:
    step = 1 << 22
    for i0 in range(0, cd.shape[0]):
        row = cd[i0]
        row[:] = cupy.rint(row / s) * s
    vk = wdf.get_jk(dm, hermi=1, with_j=False, with_k=True)[1]
    R4['abs_step_%g' % s] = {'dE_K': ek_shift(vk), 'max_dK': float(abs(vk - vk0).max())}
    log('R4', s, R4['abs_step_%g' % s])
    restore()
for i0 in range(cd.shape[0]):
    cd[i0] = cd[i0].astype(np.float32).astype(np.float64)
vk = wdf.get_jk(dm, hermi=1, with_j=False, with_k=True)[1]
R4['fp32'] = {'dE_K': ek_shift(vk), 'max_dK': float(abs(vk - vk0).max())}
log('R4 fp32', R4['fp32'])
restore()
OUT['R4'] = R4
save()

R5 = {}
amax = float(abs(cd).max())
R5['absmax'] = amax
edges = np.arange(-40, 4)
hist = np.zeros(len(edges) - 1)
blkbits = {}
for eps in [1e-9, 1e-10, 1e-11, 1e-12]:
    blkbits[eps] = [0., 0.]
nel = 0
for i0 in range(0, cd.shape[0], 64):
    x = cd[i0:i0 + 64]
    a = abs(x)
    lg = cupy.log2(cupy.maximum(a, 1e-300))
    hist += cupy.asnumpy(cupy.histogram(lg, bins=cupy.asarray(edges, dtype=np.float64))[0])
    n = x.size
    nel += n
    # block of 16 consecutive pair indices shares an exponent
    m = x.shape[1] // 16 * 16
    bmax = a[:, :m].reshape(x.shape[0], -1, 16).max(axis=2)
    for eps in blkbits:
        per_el = cupy.maximum(0, cupy.ceil(cupy.log2(cupy.maximum(a, 1e-300) / eps)) + 1)
        blk = cupy.maximum(0, cupy.ceil(cupy.log2(cupy.maximum(bmax, 1e-300) / eps)) + 1)
        blkbits[eps][0] += float(per_el.sum())
        blkbits[eps][1] += float(blk.sum() * 16)
R5['log2_hist_edges'] = edges.tolist()
R5['log2_hist'] = hist.tolist()
R5['bits_per_el'] = {str(e): v[0] / nel for e, v in blkbits.items()}
R5['bits_block16'] = {str(e): v[1] / nel for e, v in blkbits.items()}
log('R5', R5['bits_per_el'], R5['bits_block16'])
OUT['R5'] = R5
save()
del backup

# ---------------------------------------------------------------- R6 int8 GEMM throughput
R6 = {}
shapes = {'xc_vmat': (512, 512, 16384), 'xc_vco': (512, 64, 16384), 'K2': (624, 624, 61 * 1904),
          'K1_slab': (64 * 1904, 64, 624)}
for name, (m, n, k) in shapes.items():
    k = min(k, crt_gemm.MAX_K) // 16 * 16
    a = cupy.random.randint(-127, 127, (nmod, m, k), dtype=np.int8)
    b = cupy.random.randint(-127, 127, (nmod, n, k), dtype=np.int8)
    c = cupy.empty((nmod, m, n), dtype=np.int32)
    t, _ = tm(lambda: crt_gemm.int8_mm_nt_batched(a, b, c), n=3)
    R6[name] = {'mnk': [m, n, k], 'gemmEx_s': t, 'gemmEx_TOPS': 2. * m * n * k * nmod / t / 1e12}
    try:
        import torch
        ta = torch.as_tensor(a, device='cuda')
        tb = torch.as_tensor(b, device='cuda')

        def run():
            for i in range(nmod):
                torch._int_mm(ta[i], tb[i].t())
        run()
        t2, _ = tm(lambda: (run(), torch.cuda.synchronize()), n=3)
        R6[name].update(torch_int_mm_s=t2, torch_TOPS=2. * m * n * k * nmod / t2 / 1e12)
    except Exception as e:
        R6[name]['torch_err'] = repr(e)[:200]
    log('R6', name, R6[name])
    del a, b, c
OUT['R6'] = R6
save()
log('done')
