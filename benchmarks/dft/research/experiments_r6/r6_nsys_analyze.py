'''Analyse an nsys sqlite export: GPU busy (union of kernel+memcpy intervals) vs wall, per kernel category,
per NVTX range (busy fraction inside each range).'''
import sqlite3, sys, re, collections
db = sqlite3.connect(sys.argv[1])
cur = db.cursor()
names = dict(cur.execute('select id, value from StringIds'))
k = cur.execute('select start, end, shortName, demangledName from CUPTI_ACTIVITY_KIND_KERNEL').fetchall()
try:
    m = cur.execute('select start, end, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY').fetchall()
except Exception:
    m = []
nv = []
try:
    nv = cur.execute('select start, end, text, textId from NVTX_EVENTS where end is not null').fetchall()
except Exception:
    pass

CATS = [('int8 GEMM (tensor core)', r'i8|imma|s8|int8'), ('FP16 GEMM', r'h1688|f16|hgemm|884|1688gemm'),
        ('DF/Rys integrals', r'int3c|rys|ip1|int2c|fill_int|GINT|_jk|ejk|int3c2e'),
        ('AO evaluation', r'GDFTeval|eval_gto|_ao'),
        ('CRT quantize/recon', r'^quant|recon|colmax|rowexp|quant3t'),
        ('FP64 GEMM', r'dgemm|d884|sm\d+_xmma_gemm_f64|cutlass.*f64|gemm.*double|dgemv'),
        ('FP32 GEMM', r'sgemm|s1688|f32'),
        ('cuTENSOR', r'tensor|cutt|elementwise.*Tensor'),
        ('eigensolver', r'syevd|sytrd|stedc|ormtr|potrf|trsm|syrk|larf|lasr|steqr|orgtr|sterf'),
        ('VV10 / NLC', r'vv10|nlc|VV10'),
        ('libxc', r'xc_|libxc|work_mgga|work_gga|work_lda'),
        ('cupy elementwise/reduction', r'cupy_|cub|reduce|Reduce|DeviceReduce|DeviceScan')]


def cat(n):
    for c, pat in CATS:
        if re.search(pat, n):
            return c
    return 'other'


ivs = sorted([(a, b) for a, b, *_ in k] + [(a, b) for a, b, _ in m])


def union(ivs, lo=None, hi=None):
    tot = 0; cs = ce = None
    for a, b in ivs:
        if lo is not None:
            a, b = max(a, lo), min(b, hi)
            if b <= a:
                continue
        if cs is None or a > ce:
            if cs is not None:
                tot += ce - cs
            cs, ce = a, b
        else:
            ce = max(ce, b)
    if cs is not None:
        tot += ce - cs
    return tot


t0 = min(a for a, b in ivs); t1 = max(b for a, b in ivs)
print('profiled GPU span %.2f s, GPU busy %.2f s (%.0f %%)' % ((t1 - t0) / 1e9, union(ivs) / 1e9, 100 * union(ivs) / (t1 - t0)))
bycat = collections.Counter(); byname = collections.Counter(); cnt = collections.Counter()
for a, b, sn, dn in k:
    n = names.get(dn, '') or names.get(sn, '')
    c = cat(n)
    bycat[c] += b - a
    byname[(c, names.get(sn, '')[:60])] += b - a
    cnt[c] += 1
memt = sum(b - a for a, b, _ in m)
print('memcpy total %.2f s' % (memt / 1e9))
print('kernel time by category:')
for c, t in bycat.most_common():
    print('  %-28s %7.2f s  %7d launches' % (c, t / 1e9, cnt[c]))
print('top kernels:')
for (c, n), t in byname.most_common(25):
    print('  %7.3f s  %-24s %s' % (t / 1e9, c, n))
print('NVTX ranges (wall, GPU busy inside, busy %):')
agg = collections.defaultdict(lambda: [0, 0, 0])
for a, b, txt, tid in nv:
    t = txt or names.get(tid, '?')
    v = agg[t]; v[0] += b - a; v[1] += union(ivs, a, b); v[2] += 1
for t, (w, bz, n) in sorted(agg.items(), key=lambda kv: -kv[1][0]):
    print('  %-20s %7.2f s  busy %7.2f s (%3.0f %%)  n=%d' % (t, w / 1e9, bz / 1e9, 100 * bz / max(w, 1), n))
