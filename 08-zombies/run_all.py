"""Every number in the zombie video comes from this file.

The model is the one of Alemi, Bierbaum, Myers & Sethna (Cornell, 2015), "You Can Run, You Can Hide", re-built
from the paper (zsim.py) and run on the 2020 Census (grid.py).

A. Checks against the paper
   A1. 200 people in one room, alpha 0.6: zombies win 40% of the time; the formula says 81 zombies at the end.
   A2. One person per square: the outbreak dies out above alpha_c = 0.4373.
   A3. The United States, one random patient zero, 28 days: about 21% of outbreaks are still going at 7 and at 28 days
       (paper: 1,467 and 1,458 of 7,000), and the one-month risk is highest in northeastern Pennsylvania.
   A4. One in a million infected at the start: most Americans are zombies by the end of week one, and remote parts of
       Montana and Nevada are still untouched after four months.
B. The stories in the video: patient zero in Manhattan (4 months), the cities it reaches and when.
C. One rule changed at a time, same starting people and seeds: faster zombies, better fighters, hiding.

Run: python run_all.py            (hours on 16 threads; --quick for a small version)
Writes resultados.json here and the maps (risk maps, the long runs) to out/.
"""
import json
import os
import sys
import time

import numba as nb
import numpy as np

import zsim
from cities import city_cells

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get('ZOMBIE_OUT', os.path.join(HERE, 'out'))
QUICK = '--quick' in sys.argv
RESUME = '--resume' in sys.argv     # keep the sections already in resultados.json (and the risk maps on disk)
SEED = 2031
DT = 0.1
DAY = 24.0
B, K, NU, MU = zsim.BETA, zsim.ALPHA * zsim.BETA, zsim.NU, zsim.MU
NE_PA = {42069: 'Lackawanna', 42079: 'Luzerne', 42089: 'Monroe', 42025: 'Carbon', 42107: 'Schuylkill',
         42095: 'Northampton', 42077: 'Lehigh', 42103: 'Pike', 42127: 'Wayne', 42115: 'Susquehanna',
         42131: 'Wyoming', 42037: 'Columbia', 42093: 'Montour', 42097: 'Northumberland', 42113: 'Sullivan',
         42015: 'Bradford'}
SNAP_DAYS = [1, 2, 3, 4, 5, 6, 7, 10, 14, 21, 28, 42, 60, 90, 120]

LOG = None          # opened on the first log() call, so that importing this file (extra_hide.py does) leaves the log alone
RES = {}


def log(*a):
    global LOG
    if LOG is None:
        LOG = open(os.path.join(HERE, 'run_quick.log' if QUICK else 'run_all.log'), 'a' if RESUME else 'w', encoding='utf-8')
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + '\n')
    LOG.flush()


def check(ok, what):
    """Stops the full run when a check against the paper fails; the --quick run only warns (small samples)."""
    if not ok:
        log('CHECK FAILED:', what)
        if not QUICK:
            raise AssertionError(what)


def save():
    with open(os.path.join(HERE, 'resultados_quick.json' if QUICK else 'resultados.json'), 'w', encoding='utf-8') as f:
        json.dump(RES, f, indent=1, ensure_ascii=False)


@nb.njit(parallel=True, cache=True)
def full_many(pop, nbr, seed_cells, seed_off, betas, kappas, nu, mus, dt, t_end, snap_t, seeds, s_mult, block):
    """A few complete runs in parallel; run r starts with seed_cells[seed_off[r]:seed_off[r+1]] exposed."""
    nr = len(seeds)
    n = len(pop)
    nh = int(t_end) + 1
    snaps = np.zeros((nr, len(snap_t), n), np.int32)
    t_reach = np.zeros((nr, n), np.float32)
    t_half = np.zeros((nr, n), np.float32)
    totals = np.zeros((nr, nh, 4), np.int64)
    alive = np.zeros(nr, np.bool_)
    for r in nb.prange(nr):
        sc = seed_cells[seed_off[r]:seed_off[r + 1]]
        tot, sn, tr, th, al = zsim.us_run(pop, nbr, sc, betas[r], kappas[r], nu, mus[r], dt, t_end, snap_t, seeds[r], s_mult[r], block)
        snaps[r] = sn
        t_reach[r] = tr
        t_half[r] = th
        totals[r] = tot
        alive[r] = al
    return snaps, t_reach, t_half, totals, alive


def pack(lists):
    off = np.zeros(len(lists) + 1, np.int64)
    off[1:] = np.cumsum([len(x) for x in lists])
    return np.concatenate([np.asarray(x, np.int64) for x in lists]), off


def curve_stats(tot, total0):
    """Share of the starting humans still human at given days, and the hours until 50 / 90 / 99% are gone."""
    s = tot[:, 0] / total0
    out = {f'day{d}': float(s[min(int(d * DAY), len(s) - 1)]) for d in (1, 2, 3, 5, 7, 14, 21, 28, 60, 90, 120) if d * DAY < len(s)}
    for q in (0.5, 0.9, 0.99):
        k = np.flatnonzero(s <= 1 - q)
        out[f'hours_to_{int(q * 100)}pct_lost'] = int(k[0]) if len(k) else None
    return out


def main():
    t00 = time.time()
    g = np.load(os.path.join(OUT, 'grid.npz'))
    meta = json.load(open(os.path.join(OUT, 'grid.json')))
    land2d, pop2d, county2d = g['land'], g['pop'], g['county']
    idx, ys, xs, nbr = zsim.land_index(land2d)
    pop = pop2d[ys, xs].astype(np.int32)
    county = county2d[ys, xs]
    state = county // 1000
    n = len(pop)
    total0 = int(pop.sum())
    one = np.ones(n)
    noblock = np.zeros(n, np.uint8)
    rpath = os.path.join(HERE, 'resultados_quick.json' if QUICK else 'resultados.json')
    if RESUME and os.path.exists(rpath):
        RES.update(json.load(open(rpath, encoding='utf-8')))
        log('resuming with', [k for k in RES])
    RES['grid'] = meta
    np.savez_compressed(os.path.join(OUT, 'land_index.npz'), ys=ys.astype(np.int32), xs=xs.astype(np.int32), pop=pop, county=county)
    cells = city_cells(pop2d, land2d, meta)
    city_idx = {k: int(idx[v]) for k, v in cells.items()}
    rng = np.random.default_rng(SEED)

    # ------------------------------------------------------------ A1 well mixed
    nr = 20_000 if QUICK else 200_000
    zf, sf = zsim.wellmixed(199, 1, 0.6, nr, SEED)
    win = zf > 0
    a1 = {'runs': nr, 'zombies_win': float(win.mean()), 'mean_final_zombies_when_they_win': float(zf[win].mean()),
          'formula_final_zombies': 1 + 0.4 * 199, 'paper_zombies_win': 0.40, 'paper_formula': 81}
    sweep = {}
    for a in (0.2, 0.4, 0.6, 0.8, 0.9, 1.0, 1.1, 1.25):
        z_, _ = zsim.wellmixed(9999, 1, a, nr // 10, SEED + int(a * 1000))
        sweep[str(a)] = float((z_ > 0).mean())
    a1['zombies_win_by_alpha_10000_people'] = sweep
    RES['A1_well_mixed'] = a1
    log('A1', json.dumps(a1))
    check(abs(a1['zombies_win'] - 0.40) < 0.01, 'A1 zombies win 40%')
    save()

    # ------------------------------------------------------------ A2 lattice
    alphas = np.round(np.arange(0.425, 0.4501, 0.0025), 4)
    sizes = (256, 512, 1024)
    nrl = 300 if QUICK else 3000
    pe = {}
    for L in sizes:
        pe[L] = []
        for a in alphas:
            b_, e_ = zsim.lattice_many(L, float(a), nrl, SEED + L)
            pe[L].append(float(e_.mean()))
    # 2D percolation: P(reach edge) * L^(beta/nu) with beta/nu = 5/48 is the same for every L at the critical point
    q = {L: np.array(pe[L]) * L ** (5 / 48) for L in sizes}
    d = q[1024] - q[512]
    k = np.flatnonzero(np.sign(d[:-1]) != np.sign(d[1:]))
    ac = float(alphas[k[0]] - d[k[0]] * (alphas[k[0] + 1] - alphas[k[0]]) / (d[k[0] + 1] - d[k[0]])) if len(k) else None
    a2 = {'alphas': alphas.tolist(), 'reach_edge': {str(L): pe[L] for L in sizes}, 'runs_each': nrl,
          'alpha_c_estimate': ac, 'paper_alpha_c': 0.437344654}
    for a, name in ((0.40, '040'), (0.4373, '0437'), (0.47, '047')):
        best, bs = -1, 0
        for s in range(64):
            b_, e_, _ = zsim.lattice(1024, a, 9000 + s, False)
            if b_ > best:
                best, bs = b_, s
        b_, e_, st = zsim.lattice(1024, a, 9000 + bs, True)
        np.savez_compressed(os.path.join(OUT, f'lattice_{name}.npz'), state=st.reshape(1024, 1024), alpha=a, bitten=b_)
        a2[f'picture_{name}'] = {'seed': 9000 + bs, 'bitten': int(b_), 'reached_edge': bool(e_)}
    RES['A2_lattice'] = a2
    log('A2', json.dumps(a2))
    check(ac is not None and 0.43 < ac < 0.445, 'A2 lattice threshold')
    save()

    # ------------------------------------------------------------ A3 one random patient zero, 28 days
    nruns = 160 if QUICK else int(os.environ.get('ZRUNS', 4000))
    seeds_p0 = rng.choice(n, size=nruns, p=pop / total0)
    if RESUME and 'A3_patient_zero_ensemble' in RES and os.path.exists(os.path.join(OUT, 'risk_maps.npz')):
        log('A3 kept from the previous run')
    else:
        A3D = (3, 7) if QUICK else (7, 28)
        snap_t = np.array([A3D[0] * DAY, A3D[1] * DAY])
        risk_sum = np.zeros((2, n))
        over_sum = np.zeros((2, n))   # 'overrun' (paper's wording): the cell lost at least half its people
        alive_n = np.zeros(2, np.int64)
        alive_flags = np.zeros((nruns, 2), bool)
        lost_frac = np.zeros((nruns, 2))
        batch = 64
        t0 = time.time()
        for b0 in range(0, nruns, batch):
            sc = seeds_p0[b0:b0 + batch]
            snaps, al, tots = zsim.us_many(pop, nbr, sc, B, K, NU, MU, DT, A3D[1] * DAY, snap_t, SEED * 1000 + b0, one, noblock)
            for r in range(len(sc)):
                for q_ in range(2):
                    if al[r, q_]:
                        lost = 1.0 - snaps[r, q_] / np.maximum(pop, 1)
                        risk_sum[q_] += lost
                        over_sum[q_] += lost >= 0.5
                        alive_n[q_] += 1
                alive_flags[b0 + r] = al[r]
                lost_frac[b0 + r] = 1.0 - snaps[r, :, :].sum(axis=1) / total0
            log(f'  A3 {b0 + len(sc)}/{nruns} runs, alive at 7d {alive_n[0]}, at 28d {alive_n[1]}, {time.time() - t0:.0f}s')
        risk = risk_sum / np.maximum(alive_n, 1)[:, None]
        over = over_sum / np.maximum(alive_n, 1)[:, None]
        np.savez_compressed(os.path.join(OUT, 'risk_maps.npz'), risk7=risk[0].astype(np.float32), risk28=risk[1].astype(np.float32), over7=over[0].astype(np.float32), over28=over[1].astype(np.float32),
                            alive7=alive_n[0], alive28=alive_n[1], seeds=seeds_p0, alive=alive_flags)
        cpop = {}
        for c in np.unique(county):
            m = county == c
            cpop[int(c)] = (int(pop[m].sum()), float((risk[0, m] * pop[m]).sum() / max(pop[m].sum(), 1)),
                            float((risk[1, m] * pop[m]).sum() / max(pop[m].sum(), 1)))
        big = sorted([c for c, v in cpop.items() if v[0] >= 50_000], key=lambda c: -cpop[c][2])
        top28 = [(c, cpop[c][0], round(cpop[c][2], 4)) for c in big[:25]]
        top7 = [(c, cpop[c][0], round(cpop[c][1], 4)) for c in sorted([c for c, v in cpop.items() if v[0] >= 50_000], key=lambda c: -cpop[c][1])[:25]]
        nepa_rank = {NE_PA[c]: big.index(c) + 1 for c in NE_PA if c in big}
        # where patient zero starts: the people of their cell, for outbreaks that took off and those that did not
        sp = pop[seeds_p0]
        a3 = {'runs': nruns, 'alive_7d': float(alive_flags[:, 0].mean()), 'alive_28d': float(alive_flags[:, 1].mean()),
              'paper_alive_7d': 1467 / 7000, 'paper_alive_28d': 1458 / 7000,
              'mean_share_lost_28d_if_alive': float(lost_frac[alive_flags[:, 1], 1].mean()) if alive_flags[:, 1].any() else None,
              'counties_50k_ranked_by_28d_risk': top28, 'counties_50k_ranked_by_7d_risk': top7,
              'northeast_pennsylvania_rank_28d_among_%d' % len(big): nepa_rank,
              'patient_zero_cell_people_median_took_off': float(np.median(sp[alive_flags[:, 0]])) if alive_flags[:, 0].any() else None,
              'patient_zero_cell_people_median_died_out': float(np.median(sp[~alive_flags[:, 0]]))}
        for lo, hi in ((0, 100), (100, 1000), (1000, 10_000), (10_000, 10 ** 7)):
            m = (sp >= lo) & (sp < hi)
            a3[f'took_off_if_patient_zero_cell_has_{lo}_to_{hi}_people'] = [int(m.sum()), float(alive_flags[m, 0].mean()) if m.any() else None]
        RES['A3_patient_zero_ensemble'] = a3
        log('A3', json.dumps(a3))
        check(0.17 < a3['alive_7d'] < 0.25, 'A3 share of outbreaks alive at 7 days')
        save()

    # ------------------------------------------------------------ A4 + B: long runs (4 months)
    T = (21 if QUICK else 120) * DAY
    snap_t = np.array([d * DAY for d in SNAP_DAYS if d * DAY <= T])
    runs, labels = [], []
    for s in range(1 if QUICK else 4):                    # one in a million
        r_ = np.random.default_rng(SEED + 100 + s)
        k_ = r_.poisson(total0 / 1e6)
        runs.append(r_.choice(n, size=k_, p=pop / total0))
        labels.append(f'one_in_a_million_{s}')
    # patient zero in Manhattan: first seed whose outbreak is still going after 3 days
    ny = city_idx['New York, NY']
    hero = None
    for s in range(200):
        tot, _, _, _, al = zsim.us_run(pop, nbr, np.array([ny]), B, K, NU, MU, DT, 72.0, np.array([72.0]), SEED + 500 + s, one, noblock)
        if al:
            hero = SEED + 500 + s
            break
    runs.append([ny])
    labels.append('patient_zero_manhattan')
    seed_cells, off = pack(runs)
    seeds = np.array([SEED + 200 + i for i in range(len(runs) - 1)] + [hero], np.int64)
    if RESUME and 'B_long_runs' in RES:
        log('B kept from the previous run')
    else:
        smult = np.ones((len(runs), n))
        t0 = time.time()
        nr_ = len(runs)
        snaps, t_reach, t_half, totals, alive = full_many(pop, nbr, seed_cells, off, np.full(nr_, B), np.full(nr_, K), NU, np.full(nr_, MU),
                                                          DT, T, snap_t, seeds, smult, noblock)
        log(f'  long runs: {time.time() - t0:.0f}s')
        b = {'snap_days': [float(x / DAY) for x in snap_t], 'hero_seed': hero, 'manhattan_tries_before_takeoff': hero - (SEED + 500) + 1}
        for i, lab in enumerate(labels):
            np.savez_compressed(os.path.join(OUT, f'run_{lab}.npz'), snaps=snaps[i], t_reach=t_reach[i], t_half=t_half[i],
                                totals=totals[i], seeds=np.asarray(runs[i]))
            info = {'start_people': len(runs[i]), 'curve': curve_stats(totals[i], total0), 'alive_at_end': bool(alive[i])}
            never = ~np.isfinite(t_reach[i])
            info['cells_never_reached'] = int(never.sum())
            info['people_in_cells_never_reached'] = int(pop[never].sum())
            for st_, nm in ((30, 'Montana'), (32, 'Nevada'), (56, 'Wyoming'), (16, 'Idaho'), (38, 'North Dakota'), (49, 'Utah'), (35, 'New Mexico')):
                m = state == st_
                info[f'{nm}_people_never_reached'] = int(pop[m & never].sum())
                info[f'{nm}_share_still_human'] = float(snaps[i][-1][m].sum() / pop[m].sum())
            by_state = {}
            for st_ in np.unique(state):
                m = state == st_
                by_state[int(st_)] = round(float(snaps[i][-1][m].sum() / pop[m].sum()), 4)
            info['share_still_human_by_state_fips'] = by_state
            info['cities_hours_to_first_contact'] = {c: (None if not np.isfinite(t_reach[i][j]) else round(float(t_reach[i][j]), 1)) for c, j in city_idx.items()}
            info['cities_hours_to_half_gone'] = {c: (None if not np.isfinite(t_half[i][j]) else round(float(t_half[i][j]), 1)) for c, j in city_idx.items()}
            b[lab] = info
            log(lab, json.dumps({k: v for k, v in info.items() if k not in ('share_still_human_by_state_fips',)}))
        # where to hide: share of each county still human at the end, averaged over the one-in-a-million runs
        ucnty = np.unique(county)
        share = np.zeros((len(labels), len(ucnty)))
        cp = np.array([pop[county == c].sum() for c in ucnty])
        for i in range(len(labels)):
            for k_, c_ in enumerate(ucnty):
                m = county == c_
                share[i, k_] = snaps[i][-1][m].sum() / max(pop[m].sum(), 1)
        om = [i for i, l in enumerate(labels) if l.startswith('one_in_a_million')]
        avg = share[om].mean(axis=0)
        order = np.argsort(-avg)
        b['safest_counties_one_in_a_million'] = [(int(ucnty[k_]), int(cp[k_]), round(float(avg[k_]), 4)) for k_ in order if cp[k_] >= 5000][:30]
        b['safest_counties_over_50k'] = [(int(ucnty[k_]), int(cp[k_]), round(float(avg[k_]), 4)) for k_ in order if cp[k_] >= 50000][:20]
        hi = labels.index('patient_zero_manhattan')
        order = np.argsort(-share[hi])
        b['safest_counties_manhattan_start'] = [(int(ucnty[k_]), int(cp[k_]), round(float(share[hi, k_]), 4)) for k_ in order if cp[k_] >= 5000][:30]
        np.savez_compressed(os.path.join(OUT, 'county_share_human_end.npz'), county=ucnty, people=cp, share=share, labels=np.array(labels))
        RES['B_long_runs'] = b
        save()

    # ------------------------------------------------------------ C one change at a time (one in a million, 28 days)
    T = (14 if QUICK else 28) * DAY
    base_seeds = runs[:2] if not QUICK else runs[:1]
    variants = [('paper', B, K, MU, 1.0), ('zombies_3x_faster', 3 * B, 3 * K, 3 * MU, 1.0), ('zombies_10x_faster', 10 * B, 10 * K, 10 * MU, 1.0),
                ('everyone_hides_half_the_encounters', B, K, MU, 0.5)]
    for a in (0.9, 1.0, 1.1, 1.25):
        variants.append((f'humans_fight_alpha_{a}', B, a * B, MU, 1.0))
    c = {}
    rl, bet, kap, mus_, sms, sds, names = [], [], [], [], [], [], []
    for name, beta, kappa, mu, sm in variants:
        for i, s_ in enumerate(base_seeds):
            rl.append(s_); bet.append(beta); kap.append(kappa); mus_.append(mu); sms.append(sm); sds.append(seeds[i]); names.append(name)
    seed_cells, off = pack(rl)
    sm_ = np.array(sms)[:, None] * np.ones((1, n))
    t0 = time.time()
    _, _, _, tots, al = full_many(pop, nbr, seed_cells, off, np.array(bet), np.array(kap), NU, np.array(mus_), DT, T, np.array([T]),
                                  np.array(sds, np.int64), sm_, noblock)
    log(f'  C variants: {time.time() - t0:.0f}s')
    for i, name in enumerate(names):
        c.setdefault(name, []).append(curve_stats(tots[i], total0))
    for name in c:
        log('  C', name, json.dumps(c[name]))
    # patient zero: does the outbreak take off? (random patient zero, 4 days)
    pz = {}
    nz = 100 if QUICK else 800
    sc = seeds_p0[:nz]
    for a in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1):
        _, al, _ = zsim.us_many(pop, nbr, sc, B, a * B, NU, MU, DT, 96.0, np.array([96.0]), SEED * 7 + int(a * 100), one, noblock)
        pz[str(a)] = float(al[:, 0].mean())
        log(f'  C patient zero alpha {a}: takes off {pz[str(a)]:.3f}')
    c['patient_zero_takes_off_by_alpha_96h'] = pz
    RES['C_one_change'] = c
    RES['seconds'] = round(time.time() - t00)
    save()
    log('done', round(time.time() - t00), 's')


if __name__ == '__main__':
    main()
