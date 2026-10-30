"""The ring scenes of the video (b12 to b16), cell by cell: soldiers and zombies in every ring square over time.

The showcase maps of run2.py (seed 10000) keep people and zombies per cell on 14 days only, and no soldiers. This
script replays the SAME runs (same seed, same engine, deterministic) up to many instants and keeps, for each one, the
soldiers and zombies in every ring cell, plus people and zombies in the area around the quarantine zone.
Method of review r5 (reviews/r5.md, experiment 1): Z2.us_run2 is run from the start up to each instant, so nothing in
the engine changes.

  standing_x10   60,000 soldiers, 10x, ring on day 8, soldiers stay put (b13; breaks on day 9.0)
  standing_x100  the same at 100x (b15; breaks on day 14.0)
  moving_x100    100x, soldiers spread out again every 24 h (b16; holds)

Out: <ZOMBIE2_OUT>/vitrine_anel/<name>.npz (ZOMBIE2_OUT as in run2.py, default out/)
Run: python vitrine_anel.py [name ...]
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
_argv, sys.argv = sys.argv, [sys.argv[0]]       # run2 reads --quick/--resume from argv
import zsim2 as Z2                               # noqa: E402
import run2 as R                                 # noqa: E402  (setup and constants only)
sys.argv = _argv

OUT = os.path.join(R.OUT, 'vitrine_anel')
os.makedirs(OUT, exist_ok=True)
DAY = 24.0
P = R.P
NBR, NYC, WIDX, DT = R.NBR, R.NYC, R.WIDX, R.DT
YS, XS = P['ys'], P['xs']
SEED = 10000
RING_T = 8 * DAY
PAD = 12                                          # cells around the quarantine zone kept in the area arrays

CASES = {
    'standing_x10': dict(sx=10.0, rebal=0.0, end=10.5 * DAY),
    'standing_x100': dict(sx=100.0, rebal=0.0, end=15.0 * DAY),
    'moving_x100': dict(sx=100.0, rebal=24.0, end=30.0 * DAY),
}


def run_to(S0, b, kk, up, ks, t_end, rebal_h):
    st = Z2._seed(SEED)
    S = S0.copy()
    E = np.zeros(len(S0), np.int32)
    Z = np.zeros(len(S0), np.int32)
    Rr = np.zeros(len(S0), np.int32)
    tr = np.full(len(S0), np.inf, np.float32)
    Z2._infect_one(st, S, E, NYC)
    tot, _, _, _, nring, br, _, _ = Z2.us_run2(st, S, E, Z, Rr, tr, b, kk, up, Z2.NU, P['mu_flat'], NBR, DT, float(t_end),
                                               np.zeros(0), RING_T, R.RING_M, R.RING_W, 60000, ks, WIDX, 5, float(rebal_h))
    return S, E, Z, tr, nring, br


def depth_of(tr, t_build):
    """BFS depth from the cells reached before the ring (as Z2._build_ring)."""
    depth = np.full(len(tr), -1, np.int32)
    q = np.nonzero(tr < t_build)[0].tolist()
    depth[q] = 0
    dmax = R.RING_M + R.RING_W
    head = 0
    while head < len(q):
        c = q[head]
        head += 1
        if depth[c] >= dmax + 1:
            continue
        for d in range(4):
            j = NBR[c, d]
            if j >= 0 and depth[j] == -1:
                depth[j] = depth[c] + 1
                q.append(j)
    return depth


def times_for(breach, end):
    """Every 3 h from the ring to the end; every 0.5 h in the 12 h around the break."""
    ts = set(np.round(np.arange(RING_T + 0.2, end + 1e-6, 3.0), 1).tolist())
    if np.isfinite(breach):
        ts |= set(np.round(np.arange(breach - 12.0, min(end, breach + 12.0) + 1e-6, 0.5), 1).tolist())
        ts.add(round(breach - DT, 1))
        ts.add(round(breach, 1))
    return sorted(t for t in ts if RING_T < t <= end)


def one(name):
    cfg = CASES[name]
    S0, b, kk, up, ks, _ = R.setup('army', soldier_x=cfg['sx'])
    t0 = time.time()
    S, E, Z, tr, nring, br = run_to(S0, b, kk, up, ks, cfg['end'], cfg['rebal'])
    t_build = next((s + 1) * DT for s in range(1900, 1930) if (s + 1) * DT > RING_T)
    depth = depth_of(tr, t_build - 1e-6)
    ring = np.nonzero(depth == R.RING_M + 1)[0]
    zone = np.nonzero((depth >= 0) & (depth <= R.RING_M))[0]
    assert len(ring) == nring, (len(ring), nring)
    y0, y1 = YS[ring].min() - PAD, YS[ring].max() + PAD
    x0, x1 = XS[ring].min() - PAD, XS[ring].max() + PAD
    area = np.nonzero((YS >= y0) & (YS <= y1) & (XS >= x0) & (XS <= x1))[0]
    ts = times_for(br, cfg['end'])
    print(f'{name}: {nring} ring cells, break at {br / DAY:.3f} d, {len(ts)} instants, area {len(area)} cells '
          f'(first run {time.time() - t0:.0f}s)', flush=True)
    sold = np.zeros((len(ts), nring), np.int32)
    zr = np.zeros((len(ts), nring), np.int32)
    hum_a = np.zeros((len(ts), len(area)), np.int32)
    zom_a = np.zeros((len(ts), len(area)), np.int32)
    first_out = np.full(len(ts), -1, np.int64)
    civ = [k for k in range(S0.shape[1]) if k != ks]
    outside = (depth == -1) | (depth > R.RING_M + R.RING_W)
    for i, t in enumerate(ts):
        Si, Ei, Zi, tri, _, bri = run_to(S0, b, kk, up, ks, t, cfg['rebal'])
        assert bri == br or bri == np.inf, (bri, br)          # same run as the full one
        sold[i] = Si[ring, ks]
        zr[i] = Zi[ring]
        hum_a[i] = Si[area][:, civ].sum(1)
        zom_a[i] = Zi[area]
        out_hit = np.nonzero(outside & np.isfinite(tri))[0]
        if len(out_hit):
            first_out[i] = int(out_hit[np.argmin(tri[out_hit])])
        if i % 10 == 0:
            print(f'  {name} t={t / DAY:.2f} d: {sold[i].sum()} standing, {int((sold[i] == 0).sum())} empty ring cells, '
                  f'{zr[i].sum()} zombies in the ring ({time.time() - t0:.0f}s)', flush=True)
    np.savez_compressed(os.path.join(OUT, f'{name}.npz'), t=np.array(ts), ring=ring, zone=zone, area=area,
                        sold=sold, zr=zr, hum_a=hum_a, zom_a=zom_a, first_out=first_out, breach=br, seed=SEED,
                        soldier_x=cfg['sx'], rebal_h=cfg['rebal'], t_build=t_build)
    print(f'{name}: done in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    for nm in (sys.argv[1:] or list(CASES)):
        one(nm)
