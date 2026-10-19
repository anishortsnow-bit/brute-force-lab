"""Section C2 of the results: hiding and fast zombies with 8 starting draws each (C used 2, too few to tell hiding apart).

Same model and settings as run_all.py: one in a million infected at the start, 28 days. The first four starting draws
are the ones of the four long runs in run_all.py. Adds 'C2_eight_draws' to resultados.json.
Run: python extra_hide.py      (about 15 minutes on 16 threads)
"""
import json
import os
import time

import numpy as np

import zsim
from run_all import DAY, DT, HERE, OUT, SEED, B, K, MU, NU, curve_stats, full_many, pack


def main():
    g = np.load(os.path.join(OUT, 'grid.npz'))
    _, ys, xs, nbr = zsim.land_index(g['land'])
    pop = g['pop'][ys, xs].astype(np.int32)
    n, total0 = len(pop), int(pop.sum())
    starts = []
    for s in range(8):
        r_ = np.random.default_rng(SEED + 100 + s)
        k_ = r_.poisson(total0 / 1e6)
        starts.append(r_.choice(n, size=k_, p=pop / total0))
    variants = [('paper', B, K, MU, 1.0), ('everyone_hides_half_the_encounters', B, K, MU, 0.5),
                ('zombies_10x_faster', 10 * B, 10 * K, 10 * MU, 1.0)]
    rl, bet, kap, mus, sms, sds, names = [], [], [], [], [], [], []
    for name, beta, kappa, mu, sm in variants:
        for i, s_ in enumerate(starts):
            rl.append(s_); bet.append(beta); kap.append(kappa); mus.append(mu); sms.append(sm); sds.append(SEED + 200 + i); names.append(name)
    seed_cells, off = pack(rl)
    t0 = time.time()
    _, _, _, tots, _ = full_many(pop, nbr, seed_cells, off, np.array(bet), np.array(kap), NU, np.array(mus), DT, 28 * DAY,
                                 np.array([28 * DAY]), np.array(sds, np.int64), np.array(sms)[:, None] * np.ones((1, n)), np.zeros(n, np.uint8))
    out = {'seconds': round(time.time() - t0), 'starting_draws': 8}
    for name in dict.fromkeys(names):
        cs = [curve_stats(tots[i], total0) for i, nm in enumerate(names) if nm == name]
        h50 = [c['hours_to_50pct_lost'] for c in cs]
        out[name] = {'runs': cs, 'hours_to_50pct_lost': h50, 'days_to_50pct_lost_mean': float(np.mean(h50)) / 24,
                     'day7_still_human_mean': float(np.mean([c['day7'] for c in cs])),
                     'day28_still_human_mean': float(np.mean([c['day28'] for c in cs]))}
        print(name, h50, round(out[name]['days_to_50pct_lost_mean'], 2), round(out[name]['day28_still_human_mean'], 4), flush=True)
    path = os.path.join(HERE, 'resultados.json')
    res = json.load(open(path, encoding='utf-8'))
    res['C2_eight_draws'] = out
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=1, ensure_ascii=False)


if __name__ == '__main__':
    main()
