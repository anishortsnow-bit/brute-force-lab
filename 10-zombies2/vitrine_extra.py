"""Showcase maps that run2.py did not save: the police, unequal fighters, learning alone and real terrain.

Same rule as run2.showcase for block B: the first seed of 20000, 20001, ... whose outbreak took off (still alive after
3 days) in that scenario. For the x100 police no outbreak takes off in 240 seeds (results), so its map is seed 20000,
the outbreak dying in New York. Maps only: every number on screen comes from resultados2.json.

Out: <ZOMBIE2_OUT>/run_<name>.npz (same fields as run2.showcase; ZOMBIE2_OUT as in run2.py, default out/)
Run: python vitrine_extra.py [name ...]
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
_argv, sys.argv = sys.argv, [sys.argv[0]]       # run2 reads --quick/--resume from argv
import zsim2 as Z2                               # noqa: E402
import run2 as R                                 # noqa: E402
sys.argv = _argv
R.LOGF = open(os.path.join(HERE, 'vitrine_extra.log'), 'a', encoding='utf-8')   # never touch run2.log

DAY = 24.0
CASES = {
    'B1_unequal': ('unequal', {}, 'mu_flat'),
    'B2_learning': ('learning', {}, 'mu_flat'),
    'B4_police_x10': ('police', {'police_x': 10.0}, 'mu_flat'),
    'B4_police_x100': ('police', {'police_x': 100.0}, 'mu_flat'),
    'B4_police_x100_close_combat': ('police', {'police_x': 100.0, 'police_close': True}, 'mu_flat'),
    'B5_terrain': ('base', {}, 'mu_terrain'),
}


def took_off_seed(S0, b, kk, up, mu, ks, tries=40):
    seeds = np.arange(20000, 20000 + tries)
    tots, _, _, _, _, _ = R.run_many(S0, b, kk, up, mu, seeds, t_end=3 * DAY + 1, ks=ks)
    ok = np.nonzero(R.alive72(tots))[0]
    return int(seeds[ok[0]]) if len(ok) else None


def main(names):
    for name in names:
        kind, kw, mu_key = CASES[name]
        S0, b, kk, up, ks, _ = R.setup(kind, **kw)
        mu = R.P[mu_key]
        seed = took_off_seed(S0, b, kk, up, mu, ks)
        print(name, 'seed', seed, flush=True)
        R.showcase(name, S0, b, kk, up, mu, 20000 if seed is None else seed)


if __name__ == '__main__':
    main(sys.argv[1:] or list(CASES))
