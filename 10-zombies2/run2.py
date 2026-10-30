"""Every number of the zombie sequel comes from this file (engine: zsim2.py; inputs: prep.py; engine checks: test2.py).

All runs start like the first video's story: one person bitten in the busiest 3 km square of lower Manhattan.
An outbreak "takes off" if it is still going after 3 days (as in the first video).

A. The first video again, on the new engine (one class of people, flat ground, nobody responds); take-off seeds.
B. One change at a time (240 seeds each; take-off share, and averages over the runs that took off):
   B1 people are unequal fighters (a third win 24% of their fights, a third 44% like the movies, a third 64%)
   B2 survivors learn: every zombie you destroy raises your share of won fights by 5 points (up to 94%)
   B3 both
   B4 police: 1 in 415 people (BJS 2018) with guns, destroying zombies 10 or 100 times as fast as a civilian;
      and the x100 cop without the reach of a gun (close combat)
   B5 terrain (Tobler's hiking function on NOAA ETOPO1), and with every slope tripled
C. The army: a quarantine ring 15 km beyond the farthest zombie, with 60,000 soldiers (about the military force on the
   Gulf Coast 8 days after Katrina, DoD, 6 Sept 2005), formed on day 1 to 14, soldiers destroying zombies 2 to 100
   times as fast as a civilian; then bigger armies on day 8. Same seeds as A, so each army run is paired with the
   identical run without the army (they are the same until the ring forms).
D. Everything at once.
E. (29/10, after review r2) When the ring breaks, and the army that moves: every 24 h the soldiers still standing are
   spread evenly along the ring again (zsim2 rebal_h), the best case for an army that can reach any part of the ring.

Run: python run2.py [--quick] [--resume]
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
import zsim2 as Z2   # noqa: E402

OUT = os.getenv('ZOMBIE2_OUT', os.path.join(HERE, 'out'))   # prep.npz (prep.py) is read here; maps are written here
QUICK = '--quick' in sys.argv
RESUME = '--resume' in sys.argv
DAY = 24.0
T_END = (30 if QUICK else 120) * DAY
DT = 0.1
SNAP_DAYS = [1, 2, 3, 5, 7, 8, 10, 14, 21, 28, 42, 60, 90, 120]
REPORT_DAYS = [1, 3, 7, 14, 28, 60, 90, 120]
N_SEARCH = 96 if QUICK else 600       # seeds tried for take-off (3 days)
N_TAKEOFF = 6 if QUICK else 48        # took-off seeds used for A and the army grid
N_POP = 48 if QUICK else 240          # seeds for scenarios that change who fights (take-off changes)
SOLDIERS = 60000
RING_M, RING_W = 5, 1                 # margin (cells) between the reached area and the ring; ring width (cells)
WATCH = ['Allentown, PA', 'Philadelphia, PA', 'Scranton, PA', 'Boston, MA', 'Washington, DC', 'Pittsburgh, PA',
         'Cleveland, OH', 'Charlotte, NC', 'Detroit, MI', 'Chicago, IL', 'Atlanta, GA', 'Nashville, TN', 'St. Louis, MO']

P = np.load(os.path.join(OUT, 'prep.npz'))
NAMES = list(P['city_names'])
NYC = int(P['city_idx'][NAMES.index('New York, NY')])
POP = P['pop'].astype(np.int64)
TOTAL = int(POP.sum())
NBR = P['nbr']
WIDX = np.array([int(P['city_idx'][NAMES.index(w)]) for w in WATCH], np.int64)
LV, PL, BL, KL = Z2.ladder()

RES = {}
LOGF = None


def log(*a):
    global LOGF
    if LOGF is None:
        LOGF = open(os.path.join(HERE, 'run2_quick.log' if QUICK else 'run2.log'), 'a' if RESUME else 'w', encoding='utf-8')
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    LOGF.write(s + '\n')
    LOGF.flush()


def save():
    with open(os.path.join(HERE, 'resultados2_quick.json' if QUICK else 'resultados2.json'), 'w', encoding='utf-8') as f:
        json.dump(RES, f, indent=1, ensure_ascii=False)


def setup(kind, police_x=10.0, police_close=False, soldier_x=10.0):
    """Classes (S0, b, kk, up), the soldiers' class (-1 = none) and the civilian levels for a scenario."""
    pol = P['police'].astype(np.int64)
    civ_l, learn = [0], False
    pol_on = kind in ('police', 'all')
    sold_on = kind in ('army', 'all')
    if kind in ('unequal', 'unequal_learning', 'all'):
        civ_l = [-4, 0, 4]
    if kind in ('learning', 'unequal_learning', 'all'):
        learn = True
    levels = list(range(min(civ_l), LV[-1] + 1)) if learn else civ_l
    li = [LV.index(L) for L in levels]
    b, kk = list(BL[li]), list(KL[li])
    up = [min(i + 1, len(levels) - 1) if learn else i for i in range(len(levels))]
    K = len(levels)
    civ = POP - pol if pol_on else POP.copy()
    cols = [np.zeros(len(POP), np.int64) for _ in range(K)]
    if len(civ_l) == 1:
        cols[levels.index(0)] = civ
    else:
        # each person is weak, average or strong with probability 1/3 (one fixed draw for all runs)
        draw = np.random.default_rng(20261028).multinomial(civ, [1 / 3, 1 / 3, 1 / 3])
        for j, L in enumerate([-4, 0, 4]):
            cols[levels.index(L)] = draw[:, j]
    if pol_on:
        bp, kp = Z2.close_combat(police_x) if police_close else Z2.armed(police_x)
        b.append(bp)
        kk.append(kp)
        up.append(K)
        cols.append(pol)
        K += 1
    ks = -1
    if sold_on:
        bs, kss = Z2.armed(soldier_x)
        b.append(bs)
        kk.append(kss)
        up.append(K)
        cols.append(np.zeros(len(POP), np.int64))
        ks = K
        K += 1
    S0 = np.stack(cols, 1).astype(np.int32)
    return S0, np.array(b), np.array(kk), np.array(up, np.int64), ks, levels


def check_conservation(tots, ks, n_sold, nrings):
    """People + exposed + zombies + destroyed zombies = population (+ the soldiers once the ring is built)."""
    tot = tots[:, :, 0] + tots[:, :, 1] + tots[:, :, 2] + tots[:, :, 3]
    if ks >= 0:
        tot = tot + tots[:, :, 6 + ks]
    extra = np.where(nrings > 0, n_sold, 0)
    final = tot[:, -1] - extra
    assert np.all(tot[:, 0] == TOTAL), 'conservation at t=0'
    assert np.all(final == TOTAL), f'conservation at the end: {np.unique(final - TOTAL)}'


def summarize(tots, wts, sel_mask, rstats=None, nrings=None, ks=-1, n_sold=0, base_hum=None):
    """Means over the runs in sel_mask (the runs that took off)."""
    sel = np.nonzero(sel_mask)[0]
    out = {'runs': int(len(sel_mask)), 'took_off': int(len(sel)), 'takeoff_share': float(len(sel) / len(sel_mask))}
    if len(sel) == 0:
        return out
    hum = tots[sel][:, :, 0] / TOTAL
    for d in REPORT_DAYS:
        h = int(d * DAY)
        if h < tots.shape[1]:
            out[f'humans_day{d}'] = float(hum[:, h].mean())
            out[f'humans_day{d}_min'] = float(hum[:, h].min())
            out[f'humans_day{d}_max'] = float(hum[:, h].max())
            out[f'expected_lost_day{d}'] = float(out['takeoff_share'] * (1 - hum[:, h].mean()))
    half = []
    for r in range(len(sel)):
        x = np.nonzero(hum[r] < 0.5)[0]
        half.append(float(x[0] / DAY) if len(x) else float('inf'))
    out['half_lost_share'] = float(np.isfinite(half).mean())
    out['half_lost_day_median_of_those'] = float(np.median([h for h in half if np.isfinite(h)])) if np.isfinite(half).any() else None
    out['alive_at_end'] = float((tots[sel][:, -1, 1] + tots[sel][:, -1, 2] > 0).mean())
    out['destroyed_per_bite'] = float(tots[sel][:, -1, 5].sum() / max(1, tots[sel][:, -1, 4].sum()))
    reach = {}
    for i, w in enumerate(WATCH):
        v = wts[sel, i] / DAY
        fin = v[np.isfinite(v)]
        reach[w] = {'reached_share': float(np.isfinite(v).mean()),
                    'median_day_of_reached': float(np.median(fin)) if len(fin) else None}
    out['first_contact'] = reach
    if rstats is not None:
        rs = rstats[sel]
        built = nrings[sel] > 0
        out['ring_built_share'] = float(built.mean())
        out['ring_cells_median'] = float(np.median(nrings[sel][built])) if built.any() else None
        if built.any():
            out['people_inside_zone_median'] = float(np.median(rs[built, 0]))
            out['people_outside_ring_median'] = float(np.median(rs[built, 1]))
            out['lost_outside_median'] = float(np.median(rs[built, 2]))
            out['lost_outside_mean'] = float(rs[built, 2].mean())
            out['outside_outbreak_alive_at_end_share'] = float(rs[built, 3].mean())
            out['held_share'] = float((rs[built, 3] == 0).mean())
            if ks >= 0:
                left = tots[sel][built][:, -1, 6 + ks]
                out['soldiers_lost_share'] = float(1 - left.mean() / n_sold)
        if base_hum is not None:
            for d in REPORT_DAYS:
                h = int(d * DAY)
                if h < tots.shape[1]:
                    out[f'saved_vs_no_army_day{d}'] = float((tots[sel][:, h, 0] - base_hum[sel][:, h]).mean())
    return out


def run_many(S0, b, kk, up, mu, seeds, ring_T=-1.0, n_sold=0, ks=-1, t_end=T_END, rebal_h=0.0):
    nr = len(seeds)
    rt = np.full(nr, float(ring_T))
    ns = np.full(nr, n_sold, np.int64)
    t0 = time.time()
    tots, wts, nrings, br, al, rs = Z2.many(S0, NYC, b, kk, up, Z2.NU, mu, NBR, DT, t_end, np.asarray(seeds, np.int64),
                                            rt, RING_M, RING_W, ns, ks, WIDX, float(rebal_h))
    check_conservation(tots, ks, n_sold, nrings)
    log(f'    {nr} runs, {time.time() - t0:.0f}s')
    return tots, wts, nrings, br, al, rs


def alive72(tots):
    h = int(3 * DAY)
    return (tots[:, h, 1] + tots[:, h, 2]) > 0


def breach_stats(tots, br, ks, n_sold):
    """When the ring first let a zombie into a square never reached before (days), and the share of the army still
    standing at that hour, over the runs where that happened."""
    ok_ = np.isfinite(br)
    out = {'breached': int(ok_.sum()), 'runs': int(len(br))}
    if ok_.any():
        d = br[ok_] / DAY
        st = np.array([tots[r, min(int(br[r]), tots.shape[1] - 1), 6 + ks] for r in np.nonzero(ok_)[0]]) / n_sold
        out.update(breach_day_min=float(d.min()), breach_day_median=float(np.median(d)), breach_day_max=float(d.max()),
                   standing_at_breach_min=float(st.min()), standing_at_breach_median=float(np.median(st)),
                   standing_at_breach_max=float(st.max()))
    return out


def showcase(name, S0, b, kk, up, mu, seed, ring_T=-1.0, n_sold=0, ks=-1, rebal_h=0.0):
    """One run with the maps (humans and zombies per cell on the snapshot days)."""
    snap_t = np.array([d * DAY for d in SNAP_DAYS if d * DAY <= T_END])
    t0 = time.time()
    tot, sS, sZ, wt, nring, br, al, tr, rs = Z2.one(S0, NYC, b, kk, up, Z2.NU, mu, NBR, DT, T_END, int(seed), snap_t,
                                                    float(ring_T), RING_M, RING_W, int(n_sold), ks, WIDX,
                                                    float(rebal_h))
    np.savez_compressed(os.path.join(OUT, f'run_{name}.npz'), tot=tot, snapS=sS, snapZ=sZ, snap_days=snap_t / DAY,
                        t_reach=tr, seed=seed, ring_T=ring_T, n_sold=n_sold, ring_cells=nring, breach=br,
                        ring_stats=rs, ring_m=RING_M, ring_w=RING_W, rebal_h=rebal_h)
    log(f'    showcase {name}: seed {seed}, {time.time() - t0:.0f}s, humans at end {tot[-1, 0] / TOTAL:.4f}, '
        f'ring {nring}, ring stats {np.round(rs).tolist()}')


def main():
    global RES
    t00 = time.time()
    path = os.path.join(HERE, 'resultados2_quick.json' if QUICK else 'resultados2.json')
    if RESUME and os.path.exists(path):
        RES = json.load(open(path, encoding='utf-8'))
    RES['meta'] = {'total_people': TOTAL, 'patient_zero_cell_people': int(POP[NYC]), 'days': T_END / DAY,
                   'soldiers': SOLDIERS, 'ring_margin_km': RING_M * 3, 'ring_width_km': RING_W * 3,
                   'police_total': int(P['police'].sum()), 'watch': WATCH,
                   'police_threshold_x': 0.25 / (int(P['police'].sum()) / TOTAL) + 1,
                   'ring_crossing': 'out of the ring only by crossing the square: chance exp(-kill rate / step rate)',
                   'moving_army': 'section E only: every 24 h the soldiers still standing are spread evenly along the ring'}
    flat, terr, terr3 = P['mu_flat'], P['mu_terrain'], P['mu_terrain3']

    # ------------------------------------------------ A. the first video on the new engine; take-off seeds
    S0, b, kk, up, ks, _ = setup('base')
    if 'A' not in RES:
        log('A: take-off search')
        seeds = np.arange(10000, 10000 + N_SEARCH)
        tots, _, _, _, _, _ = run_many(S0, b, kk, up, flat, seeds, t_end=3 * DAY)
        a72 = alive72(tots)
        RES['A'] = {'search_runs': N_SEARCH, 'took_off': int(a72.sum()), 'takeoff_share': float(a72.mean()),
                    'takeoff_seeds': [int(s) for s in seeds[a72]]}
        save()
    good = np.array(RES['A']['takeoff_seeds'], np.int64)[:N_TAKEOFF]
    bh_path = os.path.join(OUT, 'base_hum_quick.npy' if QUICK else 'base_hum.npy')
    if RESUME and 'A_base' in RES:
        # the same 48 runs as before (deterministic per seed); without the cache only new sections lose "saved"
        base_hum = np.load(bh_path) if os.path.exists(bh_path) else None
    else:
        log('A: base, took-off seeds, 120 days')
        tots, wts, _, _, _, _ = run_many(S0, b, kk, up, flat, good)
        base_hum = tots[:, :, 0].copy()
        np.save(bh_path, base_hum)
    if 'A_base' not in RES:
        RES['A_base'] = summarize(tots, wts, np.ones(len(good), bool))
        RES['A_base']['first_video_hero'] = {'humans_day7': 0.931, 'humans_day28': 0.815, 'humans_day60': 0.709,
                                             'humans_day120': 0.443, 'half_lost_day': 103}
        save()
        showcase('base', S0, b, kk, up, flat, good[0])

    # ------------------------------------------------ B. one change at a time
    pop_seeds = np.arange(20000, 20000 + N_POP)
    for key, kind, kw, mu in [('B0_base_same_seeds', 'base', {}, flat),
                              ('B1_unequal', 'unequal', {}, flat),
                              ('B2_learning', 'learning', {}, flat),
                              ('B3_unequal_learning', 'unequal_learning', {}, flat),
                              ('B4_police_x10', 'police', {'police_x': 10.0}, flat),
                              ('B4_police_x100', 'police', {'police_x': 100.0}, flat),
                              ('B4_police_x100_close_combat', 'police', {'police_x': 100.0, 'police_close': True}, flat),
                              ('B5_terrain', 'base', {}, terr),
                              ('B5_terrain_slopes_x3', 'base', {}, terr3)]:
        if key in RES:
            continue
        log(key)
        S0, b, kk, up, ks, levels = setup(kind, **kw)
        tots, wts, _, _, _, _ = run_many(S0, b, kk, up, mu, pop_seeds)
        a72 = alive72(tots)
        RES[key] = summarize(tots, wts, a72)
        RES[key]['classes'] = {'levels': levels, 'win_share': [round(float(x), 4) for x in (kk / (b + kk))]}
        sel = np.nonzero(a72)[0]
        if kind in ('learning', 'unequal_learning') and len(sel):
            fin = tots[sel][:, -1, 6:6 + len(levels)].mean(0)
            ini = S0.sum(0)[:len(levels)].astype(float)
            p = kk[:len(levels)] / (b[:len(levels)] + kk[:len(levels)])
            RES[key]['avg_win_share_start'] = float((ini * p).sum() / ini.sum())
            RES[key]['survivors_by_level_end'] = {str(L): float(x) for L, x in zip(levels, fin)}
            if kind == 'learning':
                RES[key]['veterans_1plus'] = float(fin[1:].sum())
                RES[key]['veterans_5plus'] = float(fin[5:].sum())
                RES[key]['veterans_top'] = float(fin[-1])
        if kind == 'unequal' and len(sel):
            ini = S0.sum(0)[:3].astype(float)
            fin = tots[sel][:, -1, 6:9].mean(0)
            bit = ini - fin
            RES[key]['bitten_by_class'] = {'weak': float(bit[0]), 'average': float(bit[1]), 'strong': float(bit[2])}
            RES[key]['bitten_weak_over_strong'] = float(bit[0] / max(bit[2], 1))
        save()
        if key in ('B3_unequal_learning', 'B5_terrain_slopes_x3') and len(sel):
            showcase(key, S0, b, kk, up, mu, pop_seeds[sel[0]])

    # ------------------------------------------------ C. the army ring (same seeds as A: paired with no army)
    if 'C' not in RES:
        RES['C'] = {}
    # 29/10, after review r1: the ring now stops zombies that try to walk through it (zsim2.us_run2), so this section
    # was run again; x2 and the x10 days other than 8 were dropped (not in the video), 3 million x10 added (same
    # soldiers x deadliness as 300 thousand x100). The leaky v1 is described in reviews/r1.md.
    combos = [(10.0, 8, ns) for ns in [SOLDIERS, 300000, 1300000, 3000000]]
    combos += [(100.0, d, SOLDIERS) for d in [1, 2, 3, 5, 8, 14]]
    combos += [(100.0, 8, ns) for ns in [300000, 1300000]]
    for sx, day, ns in combos:
        key = f'x{int(sx)}_day{day}_soldiers{ns}'
        if key in RES['C']:
            continue
        log('C', key)
        S0, b, kk, up, ks, _ = setup('army', soldier_x=sx)
        tots, wts, nr_, br_, _, rs = run_many(S0, b, kk, up, flat, good, ring_T=day * DAY, n_sold=ns, ks=ks)
        RES['C'][key] = summarize(tots, wts, np.ones(len(good), bool), rs, nr_, ks, ns, base_hum)
        save()
    if 'C_showcase' not in RES:
        for sx in [10.0, 100.0]:
            S0, b, kk, up, ks, _ = setup('army', soldier_x=sx)
            showcase(f'army_katrina_x{int(sx)}', S0, b, kk, up, flat, good[0], ring_T=8 * DAY, n_sold=SOLDIERS, ks=ks)
        S0, b, kk, up, ks, _ = setup('army', soldier_x=100.0)
        showcase('army_300k_x100', S0, b, kk, up, flat, good[0], ring_T=8 * DAY, n_sold=300000, ks=ks)
        RES['C_showcase'] = True
        save()

    # ------------------------------------------------ D. everything at once (army on day 8, soldiers and police x10)
    if 'D_all' not in RES:
        log('D_all')
        S0, b, kk, up, ks, levels = setup('all', police_x=10.0, soldier_x=10.0)
        tots, wts, nr_, br_, _, rs = run_many(S0, b, kk, up, terr, pop_seeds, ring_T=8 * DAY, n_sold=SOLDIERS, ks=ks)
        a72 = alive72(tots)
        RES['D_all'] = summarize(tots, wts, a72, rs, nr_, ks, SOLDIERS)
        save()
        if a72.any():
            showcase('all', S0, b, kk, up, terr, pop_seeds[np.nonzero(a72)[0][0]], ring_T=8 * DAY, n_sold=SOLDIERS,
                     ks=ks)
    # the same people and police, no army (what the army adds in D_all)
    if 'D_no_army' not in RES:
        log('D_no_army')
        S0, b, kk, up, ks, levels = setup('all', police_x=10.0, soldier_x=10.0)
        tots, wts, _, _, _, _ = run_many(S0, b, kk, up, terr, pop_seeds)
        RES['D_no_army'] = summarize(tots, wts, alive72(tots))
        save()
    # ------------------------------------------------ E. when the ring breaks; the army that moves (29/10, review r2)
    if 'E' not in RES:
        RES['E'] = {}
    for sx in [10.0, 100.0]:
        key = f'breach_x{int(sx)}_day8_soldiers{SOLDIERS}'
        if key in RES['E']:
            continue
        log('E', key)
        S0, b, kk, up, ks, _ = setup('army', soldier_x=sx)
        tots, wts, nr_, br_, _, rs = run_many(S0, b, kk, up, flat, good, ring_T=8 * DAY, n_sold=SOLDIERS, ks=ks,
                                              t_end=24 * DAY)
        RES['E'][key] = breach_stats(tots, br_, ks, SOLDIERS)
        save()
    for sx, ns in [(100.0, SOLDIERS), (10.0, SOLDIERS), (10.0, 1300000)]:
        key = f'moving_x{int(sx)}_day8_soldiers{ns}'
        if key in RES['E']:
            continue
        log('E', key)
        S0, b, kk, up, ks, _ = setup('army', soldier_x=sx)
        tots, wts, nr_, br_, _, rs = run_many(S0, b, kk, up, flat, good, ring_T=8 * DAY, n_sold=ns, ks=ks,
                                              rebal_h=24.0)
        RES['E'][key] = summarize(tots, wts, np.ones(len(good), bool), rs, nr_, ks, ns, base_hum)
        RES['E'][key].update(breach_stats(tots, br_, ks, ns))
        RES['E'][key]['zombies_end_median'] = float(np.median(tots[:, -1, 2]))
        save()
    if 'E_showcase' not in RES:
        S0, b, kk, up, ks, _ = setup('army', soldier_x=100.0)
        showcase('army_katrina_x100_moving', S0, b, kk, up, flat, good[0], ring_T=8 * DAY, n_sold=SOLDIERS, ks=ks,
                 rebal_h=24.0)
        RES['E_showcase'] = True
        save()
    RES['seconds'] = round(time.time() - t00)
    save()
    log('done', RES['seconds'], 's')


if __name__ == '__main__':
    main()
