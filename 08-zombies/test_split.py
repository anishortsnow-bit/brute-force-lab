"""Check: the stepped simulation (zombies move between cells at fixed steps) gives the same outbreaks as the fully exact
(Gillespie) simulation, on real patches of the grid. Compares, over the outbreaks still alive at the end, the humans left
and the number of cells reached.
Run: python test_split.py --dts 0.1    (rural patch at 15 and 30 min, suburban patch at the 6-minute step of run_all.py)
     python test_split.py --city       (lower Manhattan, 12 hours)"""
import os
import time
import numpy as np
import zsim
import sys as _s
DTS = tuple(float(v) for v in _s.argv[_s.argv.index("--dts") + 1].split(",")) if "--dts" in _s.argv else (0.25, 0.5)

g = np.load(os.path.join(os.environ.get('ZOMBIE_OUT', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')), 'grid.npz'))
pop_all, land_all = g['pop'], g['land']
H, W = pop_all.shape


def patch(lo, hi, maxcell, seed):
    rng = np.random.default_rng(seed)
    while True:
        y, x = rng.integers(0, H - 20), rng.integers(0, W - 20)
        p = pop_all[y:y+20, x:x+20]
        if land_all[y:y+20, x:x+20].all() and lo < p.sum() < hi and p.max() < maxcell:
            return y, x, p


def compare(name, p2d, T, N, dts):
    land = np.ones_like(p2d, bool)
    _, ys, xs, nbr = zsim.land_index(land)
    p = p2d[ys, xs].astype(np.int32)
    sc = np.array([int(np.argmax(p))])
    kap = zsim.ALPHA * zsim.BETA
    out = {}
    ex = []
    t = time.time()
    for r in range(N):
        S, alive = zsim.exact_grid(p, nbr, sc, zsim.BETA, kap, zsim.NU, zsim.MU, T, 10_000 + r)
        ex.append((alive, S.sum(), (S < p).sum()))
    out['exact'] = ex
    tex = time.time() - t
    one = np.ones(len(p)); blk = np.zeros(len(p), np.uint8)
    for dt in dts:
        hy = []
        for r in range(N):
            tot, sn, tr, th, alive = zsim.us_run(p, nbr, sc, zsim.BETA, kap, zsim.NU, zsim.MU, dt, T, np.array([T]), 50_000 + r, one, blk)
            hy.append((tot[-1, 1] + tot[-1, 2] > 0, sn[0].sum(), np.isfinite(tr).sum()))
        out[dt] = hy
    print(f'\n{name}: {p.sum()} people, biggest cell {p.max()}, {N} runs each, {T:.0f} h (exact took {tex:.0f}s)')
    e = np.array(out['exact'], float)
    for dt in dts:
        h = np.array(out[dt], float)
        ea, ha = e[e[:, 0] > 0], h[h[:, 0] > 0]
        pa, pb = e[:, 0].mean(), h[:, 0].mean()
        sp = np.sqrt(pa * (1 - pa) / N + pb * (1 - pb) / N)
        sh = np.sqrt(ea[:, 1].var() / len(ea) + ha[:, 1].var() / len(ha))
        print(f'  dt={dt:.2f} h | alive: exact {pa:.3f} steps {pb:.3f} ({(pb-pa)/sp:+.1f} se) | humans left if alive: '
              f'exact {ea[:,1].mean():.0f} steps {ha[:,1].mean():.0f} ({(ha[:,1].mean()-ea[:,1].mean())/sh:+.1f} se)')
    return out


import sys
if "--urban" not in sys.argv and "--city" not in sys.argv:
    y, x, p = patch(15000, 30000, 3000, 1)
    compare(f"rural patch at row {y}, col {x}", p, 120.0, 3000, (0.25, 0.5))
if "--city" not in sys.argv:
  y, x, p = patch(150000, 400000, 25000, 2)
  compare(f"suburban patch at row {y}, col {x}", p, 48.0, 1500, DTS)

if "--city" in sys.argv:
    # dense city patch (New York): most land, cells up to ~300k people; the tau-leaping is what runs here
    land_all2 = land_all
    yy, xx = np.unravel_index(np.argmax(pop_all), pop_all.shape)
    y, x = yy - 2, xx - 2
    p2d = pop_all[y:y+5, x:x+5].astype(np.int32)
    lnd = land_all2[y:y+5, x:x+5]
    _, ys2, xs2, nbr2 = zsim.land_index(lnd)
    p = p2d[ys2, xs2]
    sc = np.array([int(np.argmax(p))])
    kap = zsim.ALPHA * zsim.BETA
    N, T = 400, 48.0
    ex, hy = [], []
    t = time.time()
    for r in range(N):
        S, alive = zsim.exact_grid(p, nbr2, sc, zsim.BETA, kap, zsim.NU, zsim.MU, T, 70_000 + r)
        ex.append((alive, S.sum()))
    tex = time.time() - t
    one = np.ones(len(p)); blk = np.zeros(len(p), np.uint8)
    for r in range(N):
        tot, sn, tr, th, alive = zsim.us_run(p, nbr2, sc, zsim.BETA, kap, zsim.NU, zsim.MU, 0.1, T, np.array([T]), 90_000 + r, one, blk)
        hy.append((tot[-1, 1] + tot[-1, 2] > 0, sn[0].sum()))
    e, h = np.array(ex, float), np.array(hy, float)
    ea, ha = e[e[:, 0] > 0], h[h[:, 0] > 0]
    sh = np.sqrt(ea[:, 1].var() / len(ea) + ha[:, 1].var() / len(ha))
    print(f'\nNew York patch: {p.sum()} people on {len(p)} land cells, biggest {p.max()}, {N} runs, {T:.0f} h (exact {tex:.0f}s)')
    print(f'  alive: exact {e[:,0].mean():.3f} steps {h[:,0].mean():.3f} | humans left if alive: exact {ea[:,1].mean():.0f} '
          f'steps {ha[:,1].mean():.0f} ({(ha[:,1].mean()-ea[:,1].mean())/sh:+.1f} se)')
