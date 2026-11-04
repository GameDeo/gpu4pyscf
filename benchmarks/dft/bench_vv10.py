#!/usr/bin/env python
# Copyright 2021-2025 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
Standalone benchmark / accuracy check of the VV10 pair kernels.  Needs only cupy
(no gpu4pyscf build): ``gpu4pyscf/dft/vv10_kernels.py`` is loaded by file path and
the reference FP64 kernel of lib/gdft/vv10.cu is embedded below.

    python bench_vv10.py --make w17 --out w17.npz      # (CPU, pyscf) build a dataset
    python bench_vv10.py --data w17.npz --variants ref64,v1_64,v1_32,sym_64,sym_32

Datasets are (coords, weights, rho_drho) of a real molecule on the wB97M-V NLC grid
(atom_grid=(50,194)); rho comes from the MINAO guess density.
'''

import argparse
import importlib.util
import os
import sys
import time
import numpy as np

NLC_PARS = (6.0, 0.01)           # wB97M-V
RHO_THRESHOLD = 1e-10

# --------------------------------------------------------------------------- datasets


def make_dataset(name, out, basis='6-31g'):
    from pyscf import gto, scf
    from pyscf.dft import numint, gen_grid
    rng = np.random.default_rng(1)

    def rot():
        q = rng.normal(size=4); q /= np.linalg.norm(q); a, b, c, d = q
        return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                         [2*(b*c+a*d), a*a-b*b+c*c-d*d, 2*(c*d-a*b)],
                         [2*(b*d-a*c), 2*(c*d+a*b), a*a-b*b-c*c+d*d]])
    water = np.array([[0, 0, 0], [0.7586, 0, 0.5043], [-0.7586, 0, 0.5043]])
    n = int(name[1:])
    atoms = []
    if name[0] == 'w':      # compact water cluster
        R = 2.9 * n**(1/3) * 0.62
        cs = []
        while len(cs) < n:
            p = rng.uniform(-R, R, 3)
            if np.linalg.norm(p) <= R and all(np.linalg.norm(p-c) > 2.7 for c in cs):
                cs.append(p)
        for c in cs:
            w = water @ rot().T + c
            atoms += [('O', w[0]), ('H', w[1]), ('H', w[2])]
    else:                   # extended zig-zag alkane C_nH_(2n+2)
        for i in range(n):
            x = i*1.27; y = 0.4 if i % 2 else -0.4; s = 1 if i % 2 else -1
            atoms += [('C', [x, y, 0]), ('H', [x, y+s*0.65, 0.89]), ('H', [x, y+s*0.65, -0.89])]
        atoms += [('H', [-1.0, -0.8, 0]), ('H', [(n-1)*1.27+1.0, 0.8 if (n-1) % 2 else -0.8, 0])]
    mol = gto.M(atom=atoms, basis=basis, verbose=0)
    g = gen_grid.Grids(mol); g.atom_grid = (50, 194); g.build()
    dm = scf.hf.init_guess_by_minao(mol)
    rho = np.hstack([numint.eval_rho(mol, ao, dm, xctype='GGA')
                     for ao, mask, w, c in numint.NumInt().block_loop(mol, g, mol.nao, 1, 2000)])
    np.savez(out, coords=g.coords, weights=g.weights, rho=rho)
    print(name, mol.natm, 'atoms', g.coords.shape[0], 'grid points ->', out)


# --------------------------------------------------------------------------- physics


def vv10_inputs(coords, weights, rho_drho, nlc_pars=NLC_PARS):
    '''Same preprocessing as numint._vv10nlc (float64 cupy arrays).'''
    import cupy
    coords = cupy.asarray(coords); weights = cupy.asarray(weights); rho_drho = cupy.asarray(rho_drho)
    kappa_prefactor = nlc_pars[0] * 1.5 * np.pi * (9 * np.pi)**(-1.0/6.0)
    C = nlc_pars[1]
    mask = cupy.where((rho_drho[0] >= RHO_THRESHOLD) & (cupy.abs(weights) > 1e-14))[0]
    rho = rho_drho[0][mask]
    xyz = cupy.ascontiguousarray(coords[mask])
    w = weights[mask]
    gamma = (rho_drho[1:4][:, mask]**2).sum(axis=0)
    omega = cupy.sqrt(C * gamma**2 / rho**4 + 4.0/3.0*np.pi*rho)
    kappa = kappa_prefactor * rho**(1.0/6.0)
    return dict(xyz=xyz, w=w, rho=rho, gamma=gamma, omega=omega, kappa=kappa, rw=rho * w,
                kpre=kappa_prefactor, C=C)


def vv10_energy_vxc(d, U, W, E, nlc_pars=NLC_PARS):
    '''E_c^nl and the (rho*w)-weighted gradient/density potential from the UWE sums.'''
    import cupy
    beta = 0.03125 * (3.0 / nlc_pars[0]**2)**0.75
    rho, gamma, C = d['rho'], d['gamma'], d['C']
    omega2 = C * gamma**2 / rho**4 + 4.0/3.0*np.pi*rho
    domega_drho = 0.5 * (4.0/3.0*np.pi - 4*C*gamma**2/rho**5) / cupy.sqrt(omega2)
    domega_dgamma = C * gamma / rho**4 / cupy.sqrt(omega2)
    dkappa = d['kpre'] / 6.0 * rho**(-5.0/6.0)
    exc = beta + 0.5 * E
    v0 = beta + E + rho * (dkappa * U + domega_drho * W)
    v1 = rho * domega_dgamma * W
    enlc = float((rho * d['w'] * exc).sum())
    return enlc, v0, v1


def rel_errors(d, ref, test):
    '''Returns (dE_nl abs, relative errors of E_nl, and rho*w weighted rel. norms of E, U, W, v0, v1).'''
    import cupy
    e0, v00, v01 = vv10_energy_vxc(d, *ref)
    e1, v10, v11 = vv10_energy_vxc(d, *test)
    rw = d['rw']
    out = {'dEnlc': e1 - e0, 'relEnlc': (e1 - e0) / abs(e0)}
    for name, a, b in (('U', ref[0], test[0]), ('W', ref[1], test[1]), ('E', ref[2], test[2]),
                       ('v0', v00, v10), ('v1', v01, v11)):
        out['rel_' + name] = float((rw * cupy.abs(b - a)).sum() / (rw * cupy.abs(a)).sum())
    # worst point among those carrying weight
    out['max_rel_E'] = float((cupy.abs(test[2] - ref[2]) / cupy.abs(ref[2])).max())
    return out


# --------------------------------------------------------------------------- reference kernels

_REF_SRC = r'''
#define NG_PER_BLOCK 128
extern "C" __global__
void vv10_fock_eval_UWE_kernel(double* __restrict__ U, double* __restrict__ W, double* __restrict__ E,
                               const double* __restrict__ grid_coord, const double* __restrict__ rho_weight,
                               const double* __restrict__ omega, const double* __restrict__ kappa,
                               const int ngrids)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = i < ngrids;
    double omega_i = __longlong_as_double(0x7ff8000000000000LL);
    double kappa_i = __longlong_as_double(0x7ff8000000000000LL);
    double3 r_i = { 0.0, 0.0, 0.0 };
    if (active) {
        omega_i = omega[i];
        kappa_i = kappa[i];
        r_i.x = grid_coord[i * 3 + 0];
        r_i.y = grid_coord[i * 3 + 1];
        r_i.z = grid_coord[i * 3 + 2];
    }
    double U_i = 0;
    double W_i = 0;
    double E_i = 0;
    __shared__ double3 shared_omega_kappa_rhow_j[NG_PER_BLOCK];
    __shared__ double3 shared_r_j[NG_PER_BLOCK];
    for (int j_block_offset = 0; j_block_offset < ngrids; j_block_offset += NG_PER_BLOCK) {
        const int j = j_block_offset + threadIdx.x;
        if (j < ngrids) {
            shared_omega_kappa_rhow_j[threadIdx.x].x = omega[j];
            shared_omega_kappa_rhow_j[threadIdx.x].y = kappa[j];
            shared_omega_kappa_rhow_j[threadIdx.x].z = rho_weight[j];
            shared_r_j[threadIdx.x].x = grid_coord[j * 3 + 0];
            shared_r_j[threadIdx.x].y = grid_coord[j * 3 + 1];
            shared_r_j[threadIdx.x].z = grid_coord[j * 3 + 2];
        }
        __syncthreads();
        const int block_upper_bound = min(NG_PER_BLOCK, ngrids - j_block_offset);
        for (int j_in_block = 0; j_in_block < block_upper_bound; j_in_block++) {
            const double omega_j = shared_omega_kappa_rhow_j[j_in_block].x;
            const double kappa_j = shared_omega_kappa_rhow_j[j_in_block].y;
            const double3 r_j = shared_r_j[j_in_block];
            const double rho_weight_j = shared_omega_kappa_rhow_j[j_in_block].z;
            const double r_ij2 = (r_i.x - r_j.x) * (r_i.x - r_j.x) + (r_i.y - r_j.y) * (r_i.y - r_j.y) + (r_i.z - r_j.z) * (r_i.z - r_j.z);
            const double g_ij = omega_i * r_ij2 + kappa_i;
            const double g_ji = omega_j * r_ij2 + kappa_j;
            const double g_sum = g_ij + g_ji;
            const double g_ij_ji_sum_1 = 1 / (g_ij * g_ji * g_sum);
            const double Phi_ij = -g_ij_ji_sum_1;
            const double E_ij = rho_weight_j * Phi_ij;
            const double U_ij = E_ij * (g_sum + g_ij) * g_ji * g_ij_ji_sum_1;
            const double W_ij = U_ij * r_ij2;
            U_i += U_ij;
            W_i += W_ij;
            E_i += E_ij;
        }
        __syncthreads();
    }
    if (active) {
        U[i] = -U_i * 1.5;
        W[i] = -W_i * 1.5;
        E[i] = E_i * 1.5;
    }
}
'''

# the pre-existing FP32 kernel (hi/lo coordinates, 4 targets per thread, chunk-of-32 FP32 sums
# flushed to FP64); verbatim from the fp32 branch (vv10_fock_eval_UWE_fp32_kernel)
_SIB32_SRC = r'''
#define NG_PER_BLOCK 128
__device__ __forceinline__ static float fast_rcp(float x)
{
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}
#define VV10_CHUNK 32
#define VV10_NI 4
extern "C" __global__
void vv10_fock_eval_UWE_fp32_kernel(double* __restrict__ U, double* __restrict__ W, double* __restrict__ E,
                                    const double* __restrict__ grid_coord, const double* __restrict__ rho_weight,
                                    const double* __restrict__ omega, const double* __restrict__ kappa,
                                    const int ngrids)
{
    const int i_base = blockIdx.x * (NG_PER_BLOCK * VV10_NI) + threadIdx.x;
    float omega_i[VV10_NI], kappa_i[VV10_NI];
    float3 rh_i[VV10_NI], rl_i[VV10_NI];
    double U_i[VV10_NI], W_i[VV10_NI], E_i[VV10_NI];
#pragma unroll
    for (int k = 0; k < VV10_NI; k++) {
        const int i = i_base + k * NG_PER_BLOCK;
        omega_i[k] = 0; kappa_i[k] = 1;
        rh_i[k] = make_float3(0, 0, 0); rl_i[k] = make_float3(0, 0, 0);
        U_i[k] = 0; W_i[k] = 0; E_i[k] = 0;
        if (i < ngrids) {
            omega_i[k] = fminf((float)omega[i], 1e12f);
            kappa_i[k] = (float)kappa[i];
            const double x = grid_coord[i * 3 + 0], y = grid_coord[i * 3 + 1], z = grid_coord[i * 3 + 2];
            rh_i[k].x = (float)x; rl_i[k].x = (float)(x - (double)rh_i[k].x);
            rh_i[k].y = (float)y; rl_i[k].y = (float)(y - (double)rh_i[k].y);
            rh_i[k].z = (float)z; rl_i[k].z = (float)(z - (double)rh_i[k].z);
        }
    }
    __shared__ float4 s_a[NG_PER_BLOCK];
    __shared__ float4 s_b[NG_PER_BLOCK];
    __shared__ float  s_k[NG_PER_BLOCK];
    for (int j_block_offset = 0; j_block_offset < ngrids; j_block_offset += NG_PER_BLOCK) {
        const int j = j_block_offset + threadIdx.x;
        if (j < ngrids) {
            const double x = grid_coord[j * 3 + 0], y = grid_coord[j * 3 + 1], z = grid_coord[j * 3 + 2];
            float4 a, b;
            a.x = (float)x; b.x = (float)(x - (double)a.x);
            a.y = (float)y; b.y = (float)(y - (double)a.y);
            a.z = (float)z; b.z = (float)(z - (double)a.z);
            a.w = (float)rho_weight[j];
            b.w = fminf((float)omega[j], 1e12f);
            s_a[threadIdx.x] = a;
            s_b[threadIdx.x] = b;
            s_k[threadIdx.x] = (float)kappa[j];
        }
        __syncthreads();
        const int n = min(NG_PER_BLOCK, ngrids - j_block_offset);
        for (int j0 = 0; j0 < n; j0 += VV10_CHUNK) {
            float U_c[VV10_NI], W_c[VV10_NI], E_c[VV10_NI];
#pragma unroll
            for (int k = 0; k < VV10_NI; k++) { U_c[k] = 0; W_c[k] = 0; E_c[k] = 0; }
            const int j1 = min(j0 + VV10_CHUNK, n);
            for (int jj = j0; jj < j1; jj++) {
                const float4 a = s_a[jj];
                const float4 b = s_b[jj];
                const float kap_j = s_k[jj];
#pragma unroll
                for (int k = 0; k < VV10_NI; k++) {
                    const float dx = (rh_i[k].x - a.x) + (rl_i[k].x - b.x);
                    const float dy = (rh_i[k].y - a.y) + (rl_i[k].y - b.y);
                    const float dz = (rh_i[k].z - a.z) + (rl_i[k].z - b.z);
                    const float r2 = fmaf(dx, dx, fmaf(dy, dy, dz * dz));
                    const float g_ij = fminf(fmaf(omega_i[k], r2, kappa_i[k]), 1e12f);
                    const float g_ji = fminf(fmaf(b.w, r2, kap_j), 1e12f);
                    const float g_sum = g_ij + g_ji;
                    const float inv = fast_rcp(g_ij * g_ji * g_sum);
                    const float E_ij = -a.w * inv;
                    const float U_ij = E_ij * (g_sum + g_ij) * g_ji * inv;
                    E_c[k] += E_ij;
                    U_c[k] += U_ij;
                    W_c[k] = fmaf(U_ij, r2, W_c[k]);
                }
            }
#pragma unroll
            for (int k = 0; k < VV10_NI; k++) {
                E_i[k] += (double)E_c[k];
                U_i[k] += (double)U_c[k];
                W_i[k] += (double)W_c[k];
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int k = 0; k < VV10_NI; k++) {
        const int i = i_base + k * NG_PER_BLOCK;
        if (i < ngrids) {
            U[i] = -U_i[k] * 1.5;
            W[i] = -W_i[k] * 1.5;
            E[i] = E_i[k] * 1.5;
        }
    }
}
'''


class RefKernels:
    def __init__(self):
        import cupy
        self.cupy = cupy
        self.k64 = cupy.RawModule(code=_REF_SRC).get_function('vv10_fock_eval_UWE_kernel')
        self.k32 = cupy.RawModule(code=_SIB32_SRC).get_function('vv10_fock_eval_UWE_fp32_kernel')

    def run(self, which, d):
        cupy = self.cupy
        n = d['xyz'].shape[0]
        U = cupy.empty(n); W = cupy.empty(n); E = cupy.empty(n)
        if which == 'ref64':
            self.k64(((n + 127) // 128,), (128,), (U, W, E, d['xyz'], d['rw'], d['omega'], d['kappa'], np.int32(n)))
        else:
            self.k32(((n + 511) // 512,), (128,), (U, W, E, d['xyz'], d['rw'], d['omega'], d['kappa'], np.int32(n)))
        return U, W, E


def load_kernels_module():
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.path.join(here, 'vv10_kernels.py'),
                 os.path.join(here, '..', '..', 'gpu4pyscf', 'dft', 'vv10_kernels.py')):
        if os.path.exists(cand):
            spec = importlib.util.spec_from_file_location('vv10_kernels', cand)
            m = importlib.util.module_from_spec(spec)
            sys.modules['vv10_kernels'] = m
            spec.loader.exec_module(m)
            return m
    raise RuntimeError('vv10_kernels.py not found')


# --------------------------------------------------------------------------- driver

# flop accounting (instruction slots per ordered pair, MUFU rcp counted as 4 FP32 slots;
# FP64 reciprocal counted as 10 DP instructions).  Used for the "achieved" columns.
FLOPS_PER_PAIR = {  # (instr per ordered pair, nominal flops with FMA=2)
    'ref64': (20 + 10, 28), 'sib32': (29 + 4, 32),
    'v1_64': (19 + 10, 26), 'v1_32': (19 + 4, 26),
    'sym_64': (16 + 8 + 5, 26), 'sym_32': (16 + 4 + 3, 26), 
}


def run_variant(name, d, refk, vk, rcut=0.0, extra=None):
    import cupy
    if name in ('ref64', 'sib32'):
        fn = lambda: refk.run(name, d)
    else:
        kind, prec = name.split('_')
        # a trailing 'c' applies the --rcut distance cutoff to this variant only
        cut = rcut if prec.endswith('c') else 0.0
        prec = prec.rstrip('c')
        kw = {'precision': 'fp32' if prec == '32' else 'fp64', 'rcut': cut,
              'symmetric': kind == 'sym'}
        kw.update(extra or {})
        fn = lambda: vk.eval_UWE(d['xyz'], d['rw'], d['omega'], d['kappa'], **kw)
    return fn


def timeit(fn, nrep=2):
    import cupy
    fn(); cupy.cuda.Stream.null.synchronize()
    ts = []
    out = None
    for _ in range(nrep):
        s = cupy.cuda.Event(); e = cupy.cuda.Event()
        s.record(); out = fn(); e.record(); e.synchronize()
        ts.append(cupy.cuda.get_elapsed_time(s, e) * 1e-3)
    return min(ts), out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--make')
    p.add_argument('--out')
    p.add_argument('--data', nargs='*', default=[])
    p.add_argument('--variants', default='ref64,sib32,v1_64,v1_32')
    p.add_argument('--sub', type=int, default=0, help='random subset of this many points (timing of the slow kernels)')
    p.add_argument('--rcut', type=float, default=0.0)
    p.add_argument('--skip-ref', action='store_true', help='do not run ref64 (use v1_64 as the reference)')
    p.add_argument('--sweep', default='', help='nt:mi,nt:mi,... thread-block / register-tile sweep of the sym_/v1_ variants')
    p.add_argument('--peak', default='', help='name:fp64_tflops:fp32_tflops, for the efficiency columns')
    a = p.parse_args()
    if a.make:
        make_dataset(a.make, a.out)
        return
    import cupy
    dev = cupy.cuda.runtime.getDeviceProperties(0)
    print('device', dev['name'].decode(), flush=True)
    refk = RefKernels()
    vk = load_kernels_module()
    peaks = {}
    if a.peak:
        nm, p64, p32 = a.peak.split(':')
        peaks = {'64': float(p64), '32': float(p32)}
    for f in a.data:
        z = np.load(f)
        d = vv10_inputs(z['coords'], z['weights'], z['rho'])
        if a.sub:
            rs = np.random.default_rng(0)
            idx = cupy.asarray(np.sort(rs.choice(d['xyz'].shape[0], a.sub, replace=False)))
            for k in ('xyz', 'w', 'rho', 'gamma', 'omega', 'kappa', 'rw'):
                d[k] = cupy.ascontiguousarray(d[k][idx])
        n = d['xyz'].shape[0]
        print('=== %s: N = %d (pairs %.3e)' % (os.path.basename(f), n, float(n)**2), flush=True)
        ref = None
        if a.sweep:
            for name in a.variants.split(','):
                for cfg in a.sweep.split(','):
                    nt, mi = (int(v) for v in cfg.split(':'))
                    fn = run_variant(name, d, refk, vk, rcut=a.rcut, extra={'nthreads': nt, 'mi': mi})
                    try:
                        t, out = timeit(fn, nrep=2)
                    except Exception as e:
                        print('%-8s nt=%d mi=%d failed: %s' % (name, nt, mi, str(e)[:80]))
                        continue
                    print('%-8s nt=%3d mi=%d  %.3f s  %.3e pairs/s' % (name, nt, mi, t, float(n)**2 / t), flush=True)
            continue
        for name in a.variants.split(','):
            fn = run_variant(name, d, refk, vk, rcut=a.rcut)
            if name == 'ref64' and a.skip_ref:
                continue
            t, out = timeit(fn, nrep=1 if name == 'ref64' else 3)
            if ref is None:
                ref = out
            err = rel_errors(d, ref, out) if out is not ref else {}
            ins, fl = FLOPS_PER_PAIR.get(name, (0, 0))
            pps = float(n)**2 / t
            prec = '32' if name.rstrip('c').endswith('32') else '64'
            tflops = pps * fl * 1e-12
            line = '%-8s %8.3f s  %8.3e pairs/s  %6.2f TFLOPS(nominal)' % (name, t, pps, tflops)
            if peaks:
                line += ' (%.0f%% of FP%s peak)' % (100 * tflops / peaks[prec], prec)
            if err:
                line += '  dEnlc=%.2e relEnlc=%.1e relE=%.1e relU=%.1e relW=%.1e relv0=%.1e relv1=%.1e maxrelE=%.1e' % (
                    err['dEnlc'], err['relEnlc'], err['rel_E'], err['rel_U'], err['rel_W'],
                    err['rel_v0'], err['rel_v1'], err['max_rel_E'])
            print(line, flush=True)


if __name__ == '__main__':
    main()
