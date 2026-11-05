import numpy as np, time, sys
from pyscf import gto, dft
from pyscf.dft import gen_grid, numint
mol = gto.M(atom='/home/user/gpu4pyscf/benchmarks/molecules/organic/033_Bisphenol_A.xyz', basis='def2-tzvp', verbose=0)
print('nao', mol.nao)
g = gen_grid.Grids(mol); g.atom_grid=(99,590); g.prune=gen_grid.nwchem_prune; g.level=0
g.build(with_non0tab=False)
# spatial sort: Morton-like via coarse cell hashing
c = g.coords
cell = np.floor((c - c.min(0)) / 0.9).astype(np.int64)
order = np.lexsort((cell[:,2], cell[:,1], cell[:,0]))
# morton order
def part(x):
    x = x & 0x1fffff
    x = (x | x<<32) & 0x1f00000000ffff
    x = (x | x<<16) & 0x1f0000ff0000ff
    x = (x | x<<8) & 0x100f00f00f00f00f
    x = (x | x<<4) & 0x10c30c30c30c30c3
    x = (x | x<<2) & 0x1249249249249249
    return x
cellf = np.floor((c - c.min(0)) / 0.05).astype(np.int64)
m = part(cellf[:,0]) | part(cellf[:,1])<<1 | part(cellf[:,2])<<2
order = np.argsort(m)
print('ng', len(c))
rng = np.random.default_rng(0)
bs = 4096
nblk = len(c)//bs
sel = rng.choice(nblk, 12, replace=False)
for ib in sorted(sel):
    idx = order[ib*bs:(ib+1)*bs]
    pts = c[idx]
    ao = numint.eval_ao(mol, pts, deriv=1)   # [4, ng, nao]
    nrm = np.linalg.norm(ao[0],axis=0)
    keep = nrm > 1e-10   # crude screening (AO columns with non-negligible weight)
    # Stack 4 comps along rows: columns = AOs
    A = np.concatenate([ao[k][:,keep] for k in range(4)], axis=0)
    # weight by sqrt of grid weight so norm matches integration
    w = np.sqrt(np.abs(g.weights[idx]))
    A = np.concatenate([ao[k][:,keep]*w[:,None] for k in range(4)], axis=0)
    s = np.linalg.svd(A, compute_uv=False)
    nsub = A.shape[1]
    ranks = [int((s > s[0]*e).sum()) for e in (1e-6,1e-8,1e-10,1e-12)]
    ext = np.ptp(pts,axis=0)
    print(f'blk {ib:4d} extent {ext.round(2)} nao_sub {nsub:3d} rank@1e-6/8/10/12 {ranks}  ratio@1e-10 {ranks[2]/nsub:.2f}')
