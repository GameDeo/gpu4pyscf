# Copyright 2025 The PySCF Developers. All Rights Reserved.
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
INT8-stored density-fitting tensor for surrogate operators (multi-fidelity SCF inner loop).

The fitted tensor B[L, p] = (L^-1 (P|mu nu))_L over the screened AO pairs p = (mu >= nu) is stored as
    B[L, p] ~= s[p] * Q[L, p],   Q int8, s[p] = max_L |B[L, p]| / 127
(1 byte per element: ~4 GB for Taxol/def2-TZVP with a ~3600-function auxiliary basis, against 32 GB in FP64).
The quantisation error (<= 0.4 % of each pair row) only enters the surrogate operator, whose value at the reference
density is replaced by the exact one through the defect correction; the inner loop needs its response, so this
precision is enough (it changes the contraction rate, not the converged result).

    J_s(D):  rho = Q (s * D_p),  c = rho,  J_p = s * (Q^T c)        two int8 GEMV passes, FP64 accumulation
    K_s:     per aux block, B_L unpacked to dense FP16 (scale applied) and FP16 tensor-core GEMMs with FP32 output.
'''
import numpy as np
import cupy as cp
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import get_avail_mem

_GEMV_ROWS = 8

_KERNELS = r'''
extern "C" __global__
void q8_rowdot(const signed char* __restrict__ Q, const double* __restrict__ u, double* __restrict__ out,
               long long npairs, int nrows)
{
    // out[r] = sum_p Q[r, p] * u[p] for ROWS consecutive rows per block (u loaded once per p for ROWS rows)
    const int ROWS = 8;
    int r0 = blockIdx.x * ROWS;
    double acc[ROWS];
    for (int k = 0; k < ROWS; k++) acc[k] = 0.;
    for (long long p = threadIdx.x; p < npairs; p += blockDim.x) {
        double up = u[p];
        for (int k = 0; k < ROWS; k++) {
            if (r0 + k < nrows) acc[k] += (double)Q[(long long)(r0 + k) * npairs + p] * up;
        }
    }
    __shared__ double red[ROWS][256];
    for (int k = 0; k < ROWS; k++) red[k][threadIdx.x] = acc[k];
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            for (int k = 0; k < ROWS; k++) red[k][threadIdx.x] += red[k][threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x < ROWS && r0 + threadIdx.x < nrows) out[r0 + threadIdx.x] = red[threadIdx.x][0];
}

extern "C" __global__
void q8_coldot(const signed char* __restrict__ Q, const double* __restrict__ c, const float* __restrict__ s,
               double* __restrict__ out, long long npairs, int nrows)
{
    // out[p] = s[p] * sum_r Q[r, p] * c[r]
    long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ double cs[];
    double acc = 0.;
    for (int r0 = 0; r0 < nrows; r0 += 1024) {
        int n = min(1024, nrows - r0);
        __syncthreads();
        for (int k = threadIdx.x; k < n; k += blockDim.x) cs[k] = c[r0 + k];
        __syncthreads();
        if (p < npairs) {
            for (int k = 0; k < n; k++) acc += (double)Q[(long long)(r0 + k) * npairs + p] * cs[k];
        }
    }
    if (p < npairs) out[p] = acc * (double)s[p];
}

extern "C" __global__
void q8_unpack16(const signed char* __restrict__ Q, const float* __restrict__ s, const int* __restrict__ rows,
                 const int* __restrict__ cols, __half* __restrict__ out, long long npairs, int L0, int nL,
                 int nao, int naop, float inv_scale)
{
    // out[L, i, j] = out[L, j, i] = s[p] * Q[L0 + L, p] * inv_scale  (FP16), (i, j) = (rows[p], cols[p])
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)nL * npairs) return;
    int L = idx / npairs;
    long long p = idx - (long long)L * npairs;
    float v = (float)Q[(long long)(L0 + L) * npairs + p] * s[p] * inv_scale;
    __half h = __float2half(v);
    long long base = (long long)L * nao * naop;
    int i = rows[p], j = cols[p];
    out[base + (long long)i * naop + j] = h;
    out[base + (long long)j * naop + i] = h;
}
'''
_mod = None


def _kern(name, bits=8, out32=False):
    '''q8_* kernels on int8 storage; bits=16 / 32: the same kernels on int16 / float32 storage;
    out32: the unpack kernel writes float32 instead of FP16'''
    global _mod
    if _mod is None:
        srcs = []
        for b, t in ((8, 'signed char'), (16, 'short'), (32, 'float')):
            src = _KERNELS.replace('signed char', t).replace('void q8_', 'void q%d_' % b)
            src32 = src[src.index('extern "C" __global__\nvoid q%d_unpack16' % b):]
            src32 = src32.replace('__half', 'float').replace('__float2half(v)', 'v').replace('_unpack16', '_unpack32')
            srcs += [src, src32]
        _mod = cp.RawModule(code='#include <cuda_fp16.h>\n' + '\n'.join(srcs), options=('-std=c++14',))
    name = name.replace('q8_', 'q%d_' % bits)
    if out32:
        name = name.replace('_unpack16', '_unpack32')
    return _mod.get_function(name)


def _sym_half_metric(auxmol, aux_sorting, omega, lr_factor, sr_factor):
    '''J^-1/2 = V w^-1/2 V^T over the eigenvalues above LINEAR_DEP_THR, rows in the sorted decontracted aux order.'''
    from gpu4pyscf.df import int3c2e_bdiv
    from gpu4pyscf.df.df import LINEAR_DEP_THR
    j2c = int3c2e_bdiv.int2c2e(auxmol, omega=omega, lr_factor=lr_factor, sr_factor=sr_factor)
    w, v = cp.linalg.eigh(j2c)
    idx = w > LINEAR_DEP_THR
    v = v[:, idx]
    c = (v / cp.sqrt(w[idx])).dot(v.T)
    c = auxmol.apply_C_dot(c, axis=0)
    if aux_sorting is not None:
        c, tmp = cp.empty_like(c), c
        c[aux_sorting] = tmp
    return c


class Int8DF:
    '''INT8-stored fitted DF tensor of the operator lr_factor * erf(omega r)/r + sr_factor * erfc(omega r)/r
    (omega None or 0: Coulomb).  build() streams the 3c integrals over AO-pair batches with all auxiliary
    functions, applies the half metric (INT8-CRT GEMM with `nmod` moduli) and quantises each pair row.'''

    def __init__(self, mol, auxbasis, omega=None, lr_factor=None, sr_factor=None, nmod=11, metric='cd',
                 bits=8, host=False):
        self.mol = mol
        # metric 'cd': the Cholesky half metric of the DF build; 'sym': J^-1/2 (same operator B^T B; measured
        # ~2x larger quantisation error in J on BPA, kept for experiments).
        # bits 8 | 16: per-pair-row scaled integers.  The fitted coefficients rho_L of a physical density are large
        # and coherent while J_p = sum_L B_Lp rho_L cancels, so J needs 16 bits (8 bits: 1-8 % error in J
        # increments on BPA); K tolerates 8 bits (0.3 %).  nmod >= 11: with 8 CRT moduli the build error is
        # systematic and the same cancellation amplifies it ~20x.
        # host: keep Q in pinned host memory and stream it per aux block (K only; frees GPU memory for large
        # molecules at ~npairs * naux / PCIe-bandwidth per K evaluation)
        self.metric = metric
        self.bits = bits
        self.host = host
        self.auxbasis = auxbasis
        self.omega, self.lr_factor, self.sr_factor = omega, lr_factor, sr_factor
        self.nmod = nmod
        self.Q = self.s = None
        self._dense = None
        self.host_block_bytes = 2.5e8     # device buffer per streamed block of the host-resident J

    def build(self, mem_frac=0.25):
        from pyscf.df import addons
        from gpu4pyscf.df import int3c2e_bdiv
        from gpu4pyscf.df import df as gdf
        from gpu4pyscf.gto.mole import SortedMole
        mol = self.mol
        log = logger.new_logger(mol)
        t0 = log.init_timer()
        auxmol = addons.make_auxmol(mol, self.auxbasis)
        intopt = int3c2e_bdiv.Int3c2eOpt(mol, auxmol)
        intopt.mol = SortedMole.from_mol(mol, decontract=True)
        intopt.build()
        sorted_mol = intopt.mol
        recontract_bas = cp.asnumpy(sorted_mol.recontract_bas)
        if any(recontract_bas[:, int3c2e_bdiv.NCTR_OF] != 1):
            raise NotImplementedError('Int8DF: basis sets with general contractions')
        naux_sorted = intopt.auxmol.nao
        mem = get_avail_mem(exclude_memory_pool=True)
        batch_size = max(256, int(mem * mem_frac / 8 / (2 * naux_sorted)))
        omega, lr, sr = self.omega, self.lr_factor, self.sr_factor
        eval_j3c, aux_sorting, ao_pair_offsets, _ = intopt.int3c2e_evaluator(
            ao_pair_batch_size=batch_size, reorder_aux=True, omega=omega, lr_factor=lr, sr_factor=sr)
        pair_addresses, diag_addrs = intopt.pair_and_diag_indices()
        npairs = len(pair_addresses)
        if self.metric == 'sym':
            aux_coef = _sym_half_metric(intopt.auxmol, aux_sorting, omega, lr, sr)
        else:
            aux_coef, _ = gdf._decompose_j2c(intopt.auxmol, aux_sorting, omega, lr, sr)
        naux = aux_coef.shape[1]
        self.naux, self.npairs, self.nao = naux, npairs, mol.nao
        qdt = {8: np.int8, 16: np.int16, 32: np.float32}[self.bits]
        qmax = {8: 127., 16: 32767., 32: 1.}[self.bits]
        if self.host == 'auto':
            self.host = naux * npairs * (self.bits // 8) > 0.5 * get_avail_mem(exclude_memory_pool=False)
        self.qdt = np.dtype(qdt)
        if self.host:
            # pinned host memory in row-block chunks of <= 2 GB (single multi-GB pinned allocations can fail)
            import cupyx
            R = max(1, int(2 ** 31 // (npairs * self.qdt.itemsize)))
            self.Qc = [(r0, cupyx.empty_pinned((min(R, naux - r0), npairs), dtype=qdt)) for r0 in range(0, naux, R)]
            self.Q = None
        else:
            self.Q = cp.empty((naux, npairs), dtype=qdt)
        self.s = cp.empty(npairs, dtype=np.float32)
        old = gdf.CDERI_CRT_MODULI
        gdf.CDERI_CRT_MODULI = self.nmod
        amax32 = 0.
        try:
            for batch_id in range(len(ao_pair_offsets) - 1):
                j3c = eval_j3c(batch_id)
                p0, p1 = int(ao_pair_offsets[batch_id]), int(ao_pair_offsets[batch_id + 1])
                blk = cp.empty((j3c.shape[0], naux))
                gdf._j3c_dot(j3c, aux_coef, blk)
                j3c = None
                if self.bits == 32:
                    self.s[p0:p1] = 1.
                    amax32 = max(amax32, float(abs(blk).max()))
                    q = blk.astype(np.float32).T
                else:
                    amax = abs(blk).max(axis=1)
                    sc = cp.where(amax > 0, amax / qmax, 1.)
                    self.s[p0:p1] = sc
                    q = cp.rint(blk / sc[:, None]).astype(qdt).T
                blk = None
                if self.host:
                    qh = q.get()
                    for r0, c in self.Qc:
                        c[:, p0:p1] = qh[r0:r0 + len(c)]
                    qh = None
                else:
                    self.Q[:, p0:p1] = q
                q = None
        finally:
            gdf.CDERI_CRT_MODULI = old
        pa = cp.asarray(pair_addresses)
        self.rows, self.cols = [x.astype(np.int32) for x in divmod(pa, mol.nao)]
        diag = cp.zeros(npairs, dtype=bool)
        diag[cp.asarray(diag_addrs)] = True
        self.pair_w = cp.where(diag, 1., 2.)          # D_p weights of the symmetric pair sum
        self.amax = amax32 if self.bits == 32 else float(self.s.max()) * qmax
        log.info('Int8DF build naux=%d npairs=%d bits=%d host=%s (%.2f GB)  %.1f s', naux, npairs, self.bits,
                 self.host, naux * npairs * self.bits / 8e9, logger.perf_counter() - t0[1])
        return self

    def _host_iter(self, nL):
        '''(L0, n, pinned host view of rows L0:L0+n) in blocks of <= nL rows that never straddle a chunk'''
        for r0, c in self.Qc:
            for a in range(0, len(c), nL):
                n = min(nL, len(c) - a)
                yield r0 + a, n, c[a:a + n]

    # -------------------------------------------------------------- J
    def j(self, dm):
        '''J_s(dm) (FP64 accumulation over the INT8 tensor).'''
        dm = cp.asarray(dm)
        u = dm[self.rows, self.cols] * self.pair_w * self.s
        rho = cp.empty(self.naux)
        npairs = np.int64(self.npairs)
        rowdot = _kern('q8_rowdot', self.bits)
        coldot = _kern('q8_coldot', self.bits)
        if not self.host:
            nb = (self.naux + _GEMV_ROWS - 1) // _GEMV_ROWS
            rowdot((nb,), (256,), (self.Q, u, rho, npairs, np.int32(self.naux)))
            jp = cp.empty(self.npairs)
            coldot(((self.npairs + 255) // 256,), (256,), (self.Q, rho, self.s, jp, npairs, np.int32(self.naux)),
                   shared_mem=1024 * 8)
        else:
            # pinned host storage, one streamed pass: the tensor is metric-orthonormalised (B = L^-1 (P|mn)), so
            # rho_L = B_L . D depends on row L only and J = sum_L rho_L B_L accumulates block by block.  Two device
            # buffers on a copy stream: the transfer of block k+1 overlaps the GEMVs of block k.
            nL = int(self.host_block_bytes / (self.npairs * self.qdt.itemsize)) // _GEMV_ROWS * _GEMV_ROWS
            nL = max(_GEMV_ROWS, nL)
            nbuf = min(nL, self.naux)
            qd = [cp.empty((nbuf, self.npairs), dtype=self.qdt) for _ in range(2)]
            jp = cp.zeros(self.npairs)
            part = cp.empty(self.npairs)
            cstream = cp.cuda.Stream(non_blocking=True)
            main = cp.cuda.get_current_stream()
            blocks = list(self._host_iter(nL))
            done = [None, None]

            def upload(k):
                b = k % 2
                if done[b] is not None:
                    cstream.wait_event(done[b])           # the GEMVs reading this buffer have finished
                _, n, hv = blocks[k]
                with cstream:
                    qd[b][:n].set(hv)
                ev = cp.cuda.Event()
                ev.record(cstream)
                return ev
            ready = upload(0) if blocks else None
            for k, (L0, n, hv) in enumerate(blocks):
                b = k % 2
                main.wait_event(ready)
                nxt = upload(k + 1) if k + 1 < len(blocks) else None
                rowdot(((n + _GEMV_ROWS - 1) // _GEMV_ROWS,), (256,), (qd[b], u, rho[L0:], npairs, np.int32(n)))
                coldot(((self.npairs + 255) // 256,), (256,), (qd[b], rho[L0:L0 + n].copy(), self.s, part, npairs,
                                                              np.int32(n)), shared_mem=1024 * 8)
                jp += part
                done[b] = cp.cuda.Event()
                done[b].record(main)
                ready = nxt
            main.synchronize()
            qd = part = None
        nao = self.nao
        J = cp.zeros((nao, nao))
        J[self.rows, self.cols] = jp
        J[self.cols, self.rows] = jp
        return J

    # -------------------------------------------------------------- K
    def _blocks(self, nL=None):
        '''yield (L0, L1, dense view (nL * nao, naop) of B[L0:L1] / self.amax): FP16 for 8-bit storage, FP32 for
        16/32-bit storage (FP16 would discard their extra precision)'''
        nao = self.nao
        naop = (nao + 7) // 8 * 8
        dt = np.float16 if self.bits == 8 else np.float32
        isz = np.dtype(dt).itemsize
        if nL is None:
            mem = get_avail_mem(exclude_memory_pool=True)
            nL = max(8, min(self.naux, int(mem * 0.2 / (isz * nao * naop * 3))))
        if self._dense is None or self._dense.shape[0] < nL or self._dense.dtype != dt:
            self._dense = cp.zeros((nL, nao, naop), dtype=dt)            # unscreened pairs stay zero
        buf = self._dense
        inv = 1. / self.amax
        qdev = cp.empty((nL, self.npairs), dtype=self.qdt) if self.host else None
        it = self._host_iter(nL) if self.host else ((L0, min(nL, self.naux - L0), None)
                                                    for L0 in range(0, self.naux, nL))
        for L0, n, hv in it:
            tot = n * self.npairs
            if self.host:
                qdev[:n].set(hv)
                src, l0 = qdev, 0
            else:
                src, l0 = self.Q, L0
            _kern('q8_unpack16', self.bits, self.bits != 8)(((tot + 255) // 256,), (256,),
                                 (src, self.s, self.rows, self.cols, buf, np.int64(self.npairs), np.int32(l0),
                                  np.int32(n), np.int32(nao), np.int32(naop), np.float32(inv)))
            yield L0, L0 + n, buf[:n].reshape(n * nao, naop), naop

    def free_work(self):
        self._dense = None

    def k_pairs(self, A, Bm=None, half_b=None):
        '''sum_L (B_L A)(B_L Bm')^T as float64 (nao, nao) with FP16 tensor cores, where Bm' = Bm + half_b * A
        (A, Bm: (nao, n) float64; Bm None: Bm' = A).  Used as K_s(C C^T) = k_pairs(C) and the increment
        K_s(2(C+dC)(C+dC)^T) - K_s(2CC^T) = 2 (P + P^T),  P = k_pairs(dC, C, 0.5) (P = sum Y (X + Y/2)^T).'''
        from gpu4pyscf.scf.multifidelity import gemm16_nt, _to16
        nao, n = A.shape
        npad = (n + 7) // 8 * 8
        m = 2 if Bm is not None else 1
        if self.bits != 8:
            return self._k_pairs32(A, Bm, half_b)
        P = cp.zeros((nao, nao), dtype=np.float32)
        # A and Bm scaled separately (an increment dC is orders of magnitude below C: one common FP16 scale would
        # push it into the subnormal range)
        sA = float(abs(A).max()) or 1.0
        sB = (float(abs(Bm).max()) or 1.0) if Bm is not None else sA
        Ct = cp.zeros((m * npad, (nao + 7) // 8 * 8), dtype=np.float16)
        Ct[:n, :nao] = (A.T * (1. / sA)).astype(np.float16)
        if Bm is not None:
            Ct[npad:npad + n, :nao] = (Bm.T * (1. / sB)).astype(np.float16)
        for L0, L1, B16, naop in self._blocks():
            nL = L1 - L0
            X = gemm16_nt(B16, Ct)                                       # (nL*nao, m*npad) fp32
            X = X.reshape(nL, nao, m, npad)
            Ya = cp.ascontiguousarray(X[:, :, 0].transpose(1, 0, 2)).reshape(nao, nL * npad)
            if Bm is None:
                Yb = Ya
            else:
                Yb = cp.ascontiguousarray(X[:, :, 1].transpose(1, 0, 2)).reshape(nao, nL * npad)
                Yb *= sB / sA                                            # both in units of sA * amax
                if half_b:
                    Yb += half_b * Ya
            X = None
            a16, sa = _to16(Ya)
            b16, sb = (a16, sa) if Bm is None else _to16(Yb)
            P += gemm16_nt(a16, b16) * (sa * sb)
        self._dense = None
        return P.astype(np.float64) * (sA * self.amax) ** 2

    def _k_pairs32(self, A, Bm=None, half_b=None):
        """k_pairs with FP32 dense blocks and FP32 GEMMs (16/32-bit storage)."""
        nao, n = A.shape
        m = 2 if Bm is not None else 1
        naop = (nao + 7) // 8 * 8
        Ct = cp.zeros((naop, m * n), dtype=np.float32)
        Ct[:nao, :n] = A
        if Bm is not None:
            Ct[:nao, n:] = Bm
        P = cp.zeros((nao, nao), dtype=np.float32)
        for L0, L1, B32, naop in self._blocks():
            nL = L1 - L0
            X = (B32 @ Ct).reshape(nL, nao, m, n)
            Ya = cp.ascontiguousarray(X[:, :, 0].transpose(1, 0, 2)).reshape(nao, nL * n)
            if Bm is None:
                Yb = Ya
            else:
                Yb = cp.ascontiguousarray(X[:, :, 1].transpose(1, 0, 2)).reshape(nao, nL * n)
                if half_b:
                    Yb += half_b * Ya
            X = None
            P += Ya @ Yb.T
        self._dense = None
        return P.astype(np.float64) * self.amax ** 2
