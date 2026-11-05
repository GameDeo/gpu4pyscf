import sqlite3, sys, re, collections, bisect
sys.path.insert(0, '/home/user/gpu4pyscf/benchmarks/dft/research/experiments_r6')
db = sqlite3.connect('r6prof.sqlite'); cur = db.cursor()
names = dict(cur.execute('select id, value from StringIds'))
k = cur.execute('select start, end, shortName, demangledName from CUPTI_ACTIVITY_KIND_KERNEL').fetchall()
nv = cur.execute('select start, end, text, textId from NVTX_EVENTS where end is not null').fetchall()
nv = [(a, b, t or names.get(tid, '?')) for a, b, t, tid in nv if (t or names.get(tid, '?')) not in ('SCF', 'GRAD') and not (t or '').startswith(('cub', 'thrust'))]
exec(open('/home/user/gpu4pyscf/benchmarks/dft/research/experiments_r6/r6_nsys_analyze.py').read().split("def cat")[0].split("CATS =")[1].join(["CATS =", ""]) if False else "")
from importlib import util
src = open('/home/user/gpu4pyscf/benchmarks/dft/research/experiments_r6/r6_nsys_analyze.py').read()
cats_src = src[src.index('CATS ='):src.index('ivs = sorted')]
exec(cats_src)
agg = collections.defaultdict(collections.Counter)
nv.sort()
for a, b, sn, dn in k:
    n = names.get(dn, '') or names.get(sn, '')
    best = None
    for s, e, t in nv:
        if s <= a <= e and (best is None or e - s < best[1] - best[0]):
            best = (s, e, t)
    key = best[2] if best else '(none)'
    agg[key][cat(n) + ' | ' + names.get(sn, '')[:28]] += b - a
for key, c in sorted(agg.items(), key=lambda kv: -sum(kv[1].values())):
    tot = sum(c.values())
    print('%-22s %6.2f s' % (key, tot / 1e9))
    for kk, t in c.most_common(6):
        print('      %6.3f  %s' % (t / 1e9, kk))
