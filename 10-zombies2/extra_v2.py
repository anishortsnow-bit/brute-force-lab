"""Extra numbers for script v2 (29/10, after review r1): zombies alive on ring day and at day 120 for the rings that
held, and the soldiers left. Same seeds and settings as run2.py (C and D_all), so the runs are the same ones that are
summarized in resultados2.json. Also the bites and destroyed zombies up to day 8 for everything-together and for the
first video's base (review r2, b29). Writes extra_v2.json next to this file; keys already there are kept.
Run: python extra_v2.py"""
import json
import os

import numpy as np

import run2 as R

HERE = os.path.dirname(os.path.abspath(__file__))
R.RES = json.load(open(os.path.join(HERE, 'resultados2.json'), encoding='utf-8'))
good = np.array(R.RES['A']['takeoff_seeds'], np.int64)[:R.N_TAKEOFF]
pop_seeds = np.arange(20000, 20000 + R.N_POP)
flat, terr = R.P['mu_flat'], R.P['mu_terrain']
path = os.path.join(HERE, 'extra_v2.json')
res = json.load(open(path, encoding='utf-8')) if os.path.exists(path) else {}
for name, kind, sx, day, ns, mu, seeds in [('x100_day1_60k', 'army', 100., 1, 60000, flat, good),
                                           ('x100_day14_60k', 'army', 100., 14, 60000, flat, good),
                                           ('x100_day8_300k', 'army', 100., 8, 300000, flat, good),
                                           ('x10_day8_3M', 'army', 10., 8, 3000000, flat, good),
                                           ('D_all', 'all', 10., 8, 60000, terr, pop_seeds)]:
    if name in res:
        continue
    S0, b, kk, up, ks, _ = R.setup(kind, police_x=10.0, soldier_x=sx)
    tots, wts, nr_, br_, al, rs = R.run_many(S0, b, kk, up, mu, seeds, ring_T=day * R.DAY, n_sold=ns, ks=ks)
    sel = R.alive72(tots) if kind == 'all' else np.ones(len(seeds), bool)
    t = tots[sel]
    held = rs[sel, 3] == 0
    zi = t[:, -1, 1] + t[:, -1, 2]          # exposed + zombies at day 120 (all inside the ring when it held)
    sold = t[:, -1, 6 + ks]
    h = held.any()
    res[name] = dict(runs=int(sel.sum()), held=int(held.sum()),
                     zombies_ringday_median=float(np.median(t[:, int(day * 24), 2])),
                     zombies_day28_median=float(np.median(t[:, 28 * 24, 2])),
                     zombies_end_held_median=float(np.median(zi[held])) if h else None,
                     zombies_end_held_min=float(zi[held].min()) if h else None,
                     zombies_end_held_max=float(zi[held].max()) if h else None,
                     outbreak_dead_share_held=float((zi[held] == 0).mean()) if h else None,
                     soldiers_left_held_median=float(np.median(sold[held])) if h else None)
    print(name, res[name], flush=True)
# up to day 8 (before the ring): bites and destroyed zombies, everything-together vs the first video's base
for name, kind, mu, seeds in [('day8_base', 'base', flat, good), ('day8_all', 'all', terr, pop_seeds)]:
    if name in res:
        continue
    S0, b, kk, up, ks, _ = R.setup(kind, police_x=10.0, soldier_x=10.0)
    tots, wts, _, _, _, _ = R.run_many(S0, b, kk, up, mu, seeds, t_end=8 * R.DAY)
    t = tots[R.alive72(tots)] if kind == 'all' else tots
    res[name] = dict(runs=int(len(t)), bites_median=float(np.median(t[:, -1, 4])),
                     destroyed_median=float(np.median(t[:, -1, 5])), zombies_median=float(np.median(t[:, -1, 2])),
                     destroyed_per_bite=float(t[:, -1, 5].sum() / t[:, -1, 4].sum()))
    print(name, res[name], flush=True)
json.dump(res, open(path, 'w'), indent=1)
