"""Checks of the new engine (zsim2.py) before any number goes into the video.

T1. One room, 200 people, alpha 0.6 (the paper's first test): zombies win 40%.
T2. Same small pieces of the real grid as the first video's test_split.py: the new engine with one class of people gives
    the same results as the old engine (zsim.py) and as the fully exact simulation.
T3. Three classes with identical rates = one class (classes are only labels).
T4. Classes, learning, police and terrain: the split simulation against the fully exact one.
Run: python test2.py   (writes test2.json)
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '08-zombies'))
import zsim          # noqa: E402
import zsim2 as Z2   # noqa: E402

P = np.load(os.path.join(os.getenv('ZOMBIE2_OUT', os.path.join(HERE, 'out')), 'prep.npz'))
RES = {}


def patch(r0, c0, size):
    """Land cells of a size x size block of the grid, with neighbours inside the block only."""
    ys, xs = P['ys'], P['xs']
    sel = np.nonzero((ys >= r0) & (ys < r0 + size) & (xs >= c0) & (xs < c0 + size))[0]
    loc = -np.ones(len(ys), np.int64)
    loc[sel] = np.arange(len(sel))
    nb = P['nbr'][sel]
    nbl = np.where(nb >= 0, loc[np.maximum(nb, 0)], -1).astype(np.int32)
    return sel, nbl


def stats(humans, alive):
    h = np.asarray(humans, float)
    a = np.asarray(alive, bool)
    out = {'runs': len(h), 'alive': float(a.mean()), 'humans_mean': float(h.mean()),
           'humans_se': float(h.std(ddof=1) / np.sqrt(len(h)))}
    return out


def z(a, b):
    return round((a['humans_mean'] - b['humans_mean']) / np.hypot(a['humans_se'], b['humans_se']), 2)


def run_split(S0, nbl, seed_cell, b, kk, up, mu, t_end, nruns, seed0, dt=0.1):
    seeds = np.arange(seed0, seed0 + nruns, dtype=np.int64)
    tots, _, _, _, al, _ = Z2.many(S0, seed_cell, b, kk, up, Z2.NU, mu, nbl, dt, t_end, seeds,
                                   np.full(nruns, -1.0), 0, 0, np.zeros(nruns, np.int64), -1, np.zeros(0, np.int64))
    return stats(tots[:, -1, 0], al)


def run_exact(S0, nbl, seed_cell, b, kk, up, mu, t_end, nruns, seed0):
    hs, al = [], []
    for r in range(nruns):
        S, a = Z2.exact_grid2(S0, nbl, seed_cell, b, kk, up, Z2.NU, mu, t_end, seed0 + r)
        hs.append(S.sum())
        al.append(a)
    return stats(hs, al)


def main():
    t00 = time.time()
    # ---- T1
    nr = 200000
    S0 = np.array([[200]], np.int32)
    nb1 = -np.ones((1, 4), np.int32)
    mu1 = np.zeros((1, 4))
    zwins, undecided, chunk = 0, 0, 1000
    for s0 in range(0, nr, chunk):
        seeds = np.arange(s0, s0 + chunk, dtype=np.int64)
        tots, _, _, _, al, _ = Z2.many(S0, 0, np.array([Z2.BETA]), np.array([0.6 * Z2.BETA]), np.array([0]), Z2.NU,
                                       mu1, nb1, 10.0, 3000.0, seeds, np.full(chunk, -1.0), 0, 0,
                                       np.zeros(chunk, np.int64), -1, np.zeros(0, np.int64))
        zwins += int((tots[:, -1, 0] == 0).sum())
        undecided += int(((tots[:, -1, 0] > 0) & al).sum())
    zw = zwins / nr
    RES['T1_room_alpha06_zombies_win'] = {'paper': 0.40, 'here': zw, 'runs': nr, 'undecided_after_3000h': undecided}
    print('T1', RES['T1_room_alpha06_zombies_win'], flush=True)
    assert abs(zw - 0.40) < 0.006

    # ---- T2 / T3 / T4 on the pieces of test_split.py
    pieces = {'rural': (int(P['ys'].min()) + 0, 0, 0), }
    # the same pieces as ../08-zombies/test_split.py: pick by the most populated cell near reference places
    names = list(P['city_names'])
    cidx = P['city_idx']
    ys, xs = P['ys'], P['xs']
    ny = cidx[names.index('New York, NY')]
    pieces = {
        'manhattan_5x5_12h': (ys[ny] - 2, xs[ny] - 2, 5, 12.0, 1500),
        'suburb_20x20_48h': (ys[cidx[names.index('Allentown, PA')]] - 10, xs[cidx[names.index('Allentown, PA')]] - 10, 20, 48.0, 1500),
        'rural_20x20_120h': (ys[cidx[names.index('Billings, MT')]] - 10, xs[cidx[names.index('Billings, MT')]] - 10, 20, 120.0, 2000),
    }
    lv, p, bl, kl = Z2.ladder()
    L0 = lv.index(0)
    for name, (r0, c0, size, T, nruns) in pieces.items():
        sel, nbl = patch(r0, c0, size)
        pop = P['pop'][sel].astype(np.int32)
        seed_cell = int(np.argmax(pop))
        mu_f = P['mu_flat'][sel] * (nbl >= 0)
        mu_t = P['mu_terrain'][sel] * (nbl >= 0)
        res = {'cells': len(sel), 'people': int(pop.sum())}
        t0 = time.time()
        # T2: one class
        S1 = pop[:, None].copy()
        b1, k1, u1 = np.array([Z2.BETA]), np.array([Z2.ALPHA * Z2.BETA]), np.array([0])
        res['new_split'] = run_split(S1, nbl, seed_cell, b1, k1, u1, mu_f, T, nruns, 1000)
        res['new_exact'] = run_exact(S1, nbl, seed_cell, b1, k1, u1, mu_f, T, nruns // 2, 50000)
        # old engine
        hs, al = [], []
        for r in range(nruns):
            tot, _, _, _, a = zsim.us_run(pop.astype(np.int64), nbl, np.array([seed_cell], np.int64), Z2.BETA,
                                          Z2.ALPHA * Z2.BETA, Z2.NU, Z2.MU, 0.1, T, np.zeros(0), 777000 + r,
                                          np.ones(len(sel)), np.zeros(len(sel), np.uint8))
            hs.append(tot[-1, 0])
            al.append(a)
        res['old_split'] = stats(hs, al)
        res['z_new_vs_old'] = z(res['new_split'], res['old_split'])
        res['z_new_vs_exact'] = z(res['new_split'], res['new_exact'])
        # T3: three identical classes
        S3 = np.stack([pop // 3, pop // 3, pop - 2 * (pop // 3)], 1).astype(np.int32)
        res['three_identical'] = run_split(S3, nbl, seed_cell, np.full(3, Z2.BETA), np.full(3, Z2.ALPHA * Z2.BETA),
                                           np.arange(3), mu_f, T, nruns, 1000)
        res['z_three_vs_one'] = z(res['three_identical'], res['new_split'])
        # T4: unequal fighters (thirds at L = -4, 0, +4), learning, police, terrain
        K = len(lv) + 1
        S4 = np.zeros((len(sel), K), np.int32)
        pol = P['police'][sel].astype(np.int32)
        civ = pop - pol
        third = civ // 3
        S4[:, lv.index(-4)] = third
        S4[:, L0] = third
        S4[:, lv.index(4)] = civ - 2 * third
        S4[:, K - 1] = pol
        bp, kp = Z2.armed(10.0)
        b4 = np.append(bl, bp)
        k4 = np.append(kl, kp)
        up4 = np.append(np.minimum(np.arange(len(lv)) + 1, len(lv) - 1), K - 1)
        res['mixed_split'] = run_split(S4, nbl, seed_cell, b4, k4, up4, mu_t, T, nruns, 2000)
        res['mixed_exact'] = run_exact(S4, nbl, seed_cell, b4, k4, up4, mu_t, T, nruns // 2, 60000)
        res['z_mixed_split_vs_exact'] = z(res['mixed_split'], res['mixed_exact'])
        res['seconds'] = round(time.time() - t0)
        RES['T2_T4_' + name] = res
        print(name, json.dumps(res), flush=True)
    RES['seconds'] = round(time.time() - t00)
    json.dump(RES, open(os.path.join(HERE, 'test2.json'), 'w'), indent=1)
    print('done', RES['seconds'], 's')


if __name__ == '__main__':
    main()
