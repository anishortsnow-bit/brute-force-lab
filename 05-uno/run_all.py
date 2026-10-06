"""Every run behind video 05, "How Much of UNO Is Luck?". Results go to resultados.json (one entry per run, so
an interrupted run resumes where it stopped) and run_all.log. Seed 2026.

The luck meter: the best player I could build plays a player who picks a random legal card. If skill decided
every game it would win them all; if luck decided every game, half. Model a game as decided by skill with
probability s and by a coin flip otherwise: the win rate is p = s + (1 - s) / 2, so s = 2p - 1 and the luck
share is 1 - s = 2(1 - p). With n players the coin flip becomes 1/n: s = (p - 1/n) / (1 - 1/n).
"""
import json
import os
import sys
import time
from fractions import Fraction

import numpy as np

import engine as E
import ttt

SEED = 2026
T = E.uno_table()
T8 = E.c8_table()
X = E.expert_flags(stage=1)                     # two-stage search: every option K/4 deals, the best three the rest
K_MAIN = int(os.environ.get('UNO_K', 2048))     # deals per option for the search player (tune*.log: the strength
Z_MAIN = float(os.environ.get('UNO_Z', 0.0))    # levels off near K = 2048; with K this large, z = 0 is best)
CHUNKS = 24 * 8
HERE = os.path.dirname(os.path.abspath(__file__))
RJ = os.path.join(HERE, 'resultados.json')
LOG = open(os.path.join(HERE, 'run_all.log'), 'a', encoding='utf-8')
RES = json.load(open(RJ)) if os.path.exists(RJ) else {}
NAMES = {0: 'random', 1: 'casual', 2: 'expert', 3: 'search', 4: 'cheat', 5: 'sida_random', 6: 'sida_smart',
         7: 'sida_stupid'}


def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + '\n')
    LOG.flush()


def save():
    json.dump(RES, open(RJ, 'w'), indent=1)


def skill(p, n=2):
    return (p - 1 / n) / (1 - 1 / n)


def summarize(res, NP):
    w = res[:, E.O_WIN]
    n = len(res)
    p = float((w == 1).mean())
    d = dict(n=n, p=p, se=float(np.sqrt(p * (1 - p) / n)), unfinished=int((w < 0).sum()),
             skill=skill(p, NP), steps=float(res[:, E.O_STEPS].mean()))
    first = res[:, E.O_FIRST] == 1
    d['p_first'] = float((w[first] == 1).mean()) if first.any() else None
    d['p_second'] = float((w[~first] == 1).mean()) if (~first).any() else None
    d['share_first'] = float(first.mean())
    for k, col in (('turns', E.O_TURNS), ('forced', E.O_FORCED), ('one', E.O_ONE), ('multi', E.O_MULTI),
                   ('drawn', E.O_DRAWN), ('played', E.O_PLAYED), ('penalty', E.O_PEN), ('voluntary', E.O_VOL),
                   ('sdec', E.O_SDEC), ('sagree', E.O_SAGREE), ('sspread', E.O_SSPREAD)):
        d[k] = int(res[:, col].sum())
    d['pts_when_win'] = float(res[w == 1, E.O_PTS].mean()) if (w == 1).any() else None
    d['pts_when_lose'] = float(res[w == 0, E.O_PTS].mean()) if (w == 0).any() else None
    # the deal: win rate by wild cards held at the start (focal and, with two players, the other player)
    d['by_wilds'] = {int(k): [int(((res[:, E.O_W0] == k) & (w == 1)).sum()), int((res[:, E.O_W0] == k).sum())]
                     for k in range(9) if (res[:, E.O_W0] == k).any()}
    d['by_wd4'] = {int(k): [int(((res[:, E.O_WD40] == k) & (w == 1)).sum()), int((res[:, E.O_WD40] == k).sum())]
                   for k in range(5) if (res[:, E.O_WD40] == k).any()}
    d['by_actions'] = {int(k): [int(((res[:, E.O_ACT0] == k) & (w == 1)).sum()), int((res[:, E.O_ACT0] == k).sum())]
                       for k in range(10) if (res[:, E.O_ACT0] == k).any()}
    if NP == 2:
        diff = res[:, E.O_W0] - res[:, E.O_OW0]
        d['by_wild_diff'] = {int(k): [int(((diff == k) & (w == 1)).sum()), int((diff == k).sum())]
                             for k in range(-8, 9) if (diff == k).any()}
    lens = res[:, E.O_STEPS]
    d['steps_pct'] = [float(np.percentile(lens, q)) for q in (1, 10, 50, 90, 99)]
    return d


def h2h(key, R, lineup, games, K=None, z=None, Xv=X, Tv=T, seed=0, opol=E.P_EXPERT):
    if key in RES:
        return RES[key]
    K = K_MAIN if K is None else K
    z = Z_MAIN if z is None else z
    t = time.time()
    res = E.run_batch(R, Tv, np.array(lineup, np.int32), K, E.P_EXPERT, opol, z, Xv, SEED + seed, 0, games, CHUNKS)
    d = summarize(res, int(R[E.R_NP]))
    d.update(lineup=[NAMES[x] for x in lineup], K=K if (3 in lineup or 4 in lineup) else 0, z=z, rules=R.tolist(),
             sec=round(time.time() - t, 1))
    RES[key] = d
    save()
    log(f'{key}: {d["p"]:.4%} +- {d["se"]:.4%}  skill {d["skill"]:.2%}  first {d["p_first"]:.2%} second '
        f'{d["p_second"]:.2%}  n={games:,}  {d["sec"]}s')
    return d


def matches(key, R, lineup, n, target=500, K=None, z=None, seed=0):
    if key in RES:
        return RES[key]
    K = K_MAIN if K is None else K
    z = Z_MAIN if z is None else z
    t = time.time()
    res = E.run_matches(R, T, np.array(lineup, np.int32), K, E.P_EXPERT, E.P_EXPERT, z, X, SEED + seed, 0, n, target,
                        CHUNKS)
    p = float(res[:, E.M_WIN].mean())
    g = res[:, E.M_GAMES]
    d = dict(n=n, p=p, se=float(np.sqrt(p * (1 - p) / n)), games_mean=float(g.mean()),
             games_pct=[float(np.percentile(g, q)) for q in (10, 50, 90)],
             game_p=float(res[:, E.M_FGAMES].sum() / g.sum()), lineup=[NAMES[x] for x in lineup], target=target,
             sec=round(time.time() - t, 1))
    RES[key] = d
    save()
    log(f'{key}: match {p:.3%} +- {d["se"]:.3%}  games/match {d["games_mean"]:.2f}  game win {d["game_p"]:.3%}  {d["sec"]}s')
    return d


def step_checks():
    log('== 1. checks')
    # tic-tac-toe exact values (random v random 58.49% / 28.81% / 12.70% draws, the textbook numbers)
    rr = ttt.random_vs_random()
    assert abs(float(rr['X']) - 0.584921) < 1e-6 and abs(float(rr['O']) - 0.288095) < 1e-6
    best = {}
    for me in 'XO':
        w_, d_ = ttt.value('.........', 'X', me)
        best[me] = float(w_ + d_ / 2)
    pt = (best['X'] + best['O']) / 2
    RES['ttt'] = dict(as_x=best['X'], as_o=best['O'], p=pt, skill=2 * pt - 1)
    log(f'tic-tac-toe: best v random {pt:.4%} (as X {best["X"]:.4%}, as O {best["O"]:.4%}), skill {2 * pt - 1:.2%}')
    # symmetry
    d = h2h('random_random', E.official(2), [0, 0], 10_000_000, seed=1)
    assert abs(d['p'] - 0.5) < 4 * d['se']
    # Sidajaya et al. (2024): 31.7% (forward) and 24.7% (reverse) against three random bots, 4 players
    R4 = E.sidajaya(4)
    vals = []
    for tie in range(5):
        Xs = E.expert_flags()
        Xs[E.X_SIDATIE] = tie
        vals.append(h2h(f'sida_forward_tie{tie}', R4, [6, 5, 5, 5], 2_500_000, Xv=Xs, seed=2)['p'])
    rev = h2h('sida_reverse', R4, [7, 5, 5, 5], 2_500_000, seed=3)['p']
    fwd = h2h('sida_forward_randomtie', R4, [6, 5, 5, 5], 2_500_000, seed=4)['p']
    log(f'Sidajaya forward {min(vals):.2%} to {max(vals):.2%} by colour tie order (random tie {fwd:.2%}; paper 31.7%), '
        f'reverse {rev:.3%} (paper 24.7%)')
    assert min(vals) < 0.317 < max(vals) and abs(rev - 0.247) < 0.0015


def step_ladder():
    log('== 2. ladder, two players, official rules')
    R = E.official(2)
    h2h('casual_random', R, [1, 0], 10_000_000, seed=10)
    h2h('expert_random', R, [2, 0], 10_000_000, seed=11)
    h2h('expert_casual', R, [2, 1], 10_000_000, seed=12)
    h2h('search_random', R, [3, 0], 40_000, seed=13)
    h2h('search_casual', R, [3, 1], 10_000, seed=14)
    h2h('search_expert', R, [3, 2], 10_000, seed=15)
    h2h('cheat_random', R, [4, 0], 10_000, seed=16)
    h2h('cheat_expert', R, [4, 2], 6_000, seed=17)
    h2h('cheat_search', R, [4, 3], 4_000, seed=18)


def step_deal():
    log('== 3. the deal and the turns (expert v expert)')
    h2h('expert_expert', E.official(2), [2, 2], 20_000_000, seed=20)
    h2h('casual_casual', E.official(2), [1, 1], 10_000_000, seed=21)


def step_matches():
    log('== 4. matches to 500 points')
    R = E.official(2)
    matches('m_casual_random', R, [1, 0], 1_000_000, seed=30)
    matches('m_expert_random', R, [2, 0], 1_000_000, seed=31)
    matches('m_expert_casual', R, [2, 1], 1_000_000, seed=32)
    matches('m_search_random', R, [3, 0], 2_000, seed=33)
    matches('m_search_casual', R, [3, 1], 2_000, seed=34)


def step_players():
    log('== 5. number of players (one skilled player, the rest random)')
    for n in (3, 4, 5, 6, 8, 10):
        R = E.official(n)
        lu = [2] + [0] * (n - 1)
        h2h(f'p{n}_expert_random', R, lu, 4_000_000, seed=40 + n)
        h2h(f'p{n}_casual_random', R, [1] + [0] * (n - 1), 4_000_000, seed=60 + n)
        h2h(f'p{n}_random', R, [0] * n, 4_000_000, seed=100 + n)
    for n in (3, 4, 6, 10):
        h2h(f'p{n}_search_random', E.official(n), [3] + [0] * (n - 1), 3_000, seed=80 + n)


def step_rules():
    log('== 6. house rules and Crazy Eights')
    variants = {'stack': E.rules(stack=1), 'drawuntil': E.rules(drawuntil=1, voldraw=0),
                'stack_drawuntil': E.rules(stack=1, drawuntil=1, voldraw=0)}
    for name, R in variants.items():
        h2h(f'{name}_expert_random', R, [2, 0], 4_000_000, seed=120)
        h2h(f'{name}_search_random', R, [3, 0], 10_000, seed=121)
    R8 = E.crazy_eights(2)
    h2h('c8_expert_random', R8, [2, 0], 4_000_000, Tv=T8, seed=130)
    h2h('c8_casual_random', R8, [1, 0], 4_000_000, Tv=T8, seed=131)
    h2h('c8_search_random', R8, [3, 0], 10_000, Tv=T8, seed=132)


def step_ablation():
    log('== 7. which habits matter (expert with one habit switched off, against random and against the full expert)')
    R = E.official(2)
    for f in ('holdwild', 'color', 'attack', 'chain', 'void', 'points'):
        Xf = E.expert_flags(**{f: 0})
        h2h(f'abl_{f}_random', R, [2, 0], 10_000_000, Xv=Xf, seed=140)
    Xn = E.expert_flags(sdraw=0, stage=1)
    h2h('search_nodraw_random', R, [3, 0], 10_000, Xv=Xn, seed=13)


def step_k():
    log('== 8. how strength grows with the number of rollouts')
    R = E.official(2)
    for K, n in ((8, 20_000), (32, 20_000), (128, 20_000), (512, 10_000), (8192, 4_000)):
        h2h(f'k{K}_search_random', R, [3, 0], n, K=K, seed=150)


STEPS = dict(checks=step_checks, ladder=step_ladder, deal=step_deal, matches=step_matches, players=step_players,
             rules=step_rules, ablation=step_ablation, k=step_k)

if __name__ == '__main__':
    todo = sys.argv[1:] or list(STEPS)
    log(f'run_all {time.strftime("%Y-%m-%d %H:%M")}  K={K_MAIN} z={Z_MAIN}  steps {todo}')
    t0 = time.time()
    for s in todo:
        STEPS[s]()
    save()
    log(f'done in {(time.time() - t0) / 60:.1f} min')
