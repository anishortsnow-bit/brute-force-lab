"""Experiment 03: the perfect strategy for Guess Who?, under three ways of ending the game.

1. Solves the game exactly (exact.py) for the official rules, the "free guess" house rule and
   the "instant win" house rule, and checks the published theorems against the result.
2. Plays every strategy against every other one: 100 million games per pairing (sim.py),
   on the board from the video and, as a cross-check, on the classic board's feature list.
Writes resultados.json next to this file.
"""
import json, os, time
from fractions import Fraction as F
import numpy as np

import exact as X
import sim, board, classic

HERE = os.path.dirname(os.path.abspath(__file__))
N = 24
GAMES = 100_000_000
SEED = 2026
R = {'games_per_pairing': GAMES, 'seed': SEED}


def fr(x):
    return {'frac': f'{x.numerator}/{x.denominator}', 'value': float(x)}


def table(pol, lo=1):
    T = np.zeros((N + 1, N + 1), np.int64)
    for n in range(lo, N + 1):
        for m in range(lo, N + 1):
            T[n, m] = pol(n, m)
    return T


# ------------------------------------------------------------------ 1. exact solution
t0 = time.time()
SOL = {r: X.solve(r, 40) for r in X.RULES}
P_OFF, B_OFF = SOL['official']
P_FREE, B_FREE = SOL['free_guess']
P_INS, B_INS = SOL['instant']
# published results must agree with the solver
assert all(X.cushing(n, m) in B_OFF[n][m] for n in range(1, 41) for m in range(1, 41))
assert all(X.nica(n, m) in B_INS[n][m] and X.nica_value(n, m) == P_INS[n][m] for n in range(2, 41) for m in range(2, 41))
assert abs(float(P_OFF[23][23]) - 0.5595) < 5e-5 and abs(float(P_FREE[23][23]) - 0.6597) < 5e-5      # O'Neill (2021)


def halver_free(n, m):
    """Binary search for the free-guess rule: with two suspects left, the question is a guess."""
    return 0 if (n <= 2 or m == 1) else n // 2


def halver_instant(n, m):
    return n // 2


perfect_official = X.cushing
perfect_free = X.table_policy(B_FREE, lambda n, m: n // 2)
bold = X.nica

STRATS = {
    'official': {'halver': X.halver, 'perfect': perfect_official, 'bold': bold, 'stubborn': X.halver_stubborn},
    'free_guess': {'halver': halver_free, 'perfect': perfect_free},
    'instant': {'halver': halver_instant, 'perfect': bold},
}
R['exact'] = {}
for rule in X.RULES:
    P, BEST = SOL[rule]
    lo = 2 if rule == 'instant' else 1
    S = STRATS[rule]
    match = {a: {b: fr(X.evaluate(rule, S[a], S[b], N)[N][N]) for b in S} for a in S}
    grid_p = [[float(P[n][m]) if n >= lo and m >= lo else None for m in range(N + 1)] for n in range(N + 1)]
    grid_best = [[BEST[n][m] if n >= lo and m >= lo else None for m in range(N + 1)] for n in range(N + 1)]
    canon = [[int(S['perfect'](n, m)) if n >= lo and m >= lo else None for m in range(N + 1)] for n in range(N + 1)]
    not_half = [(n, m) for n in range(lo, N + 1) for m in range(lo, N + 1) if S['halver'](n, m) not in BEST[n][m]]
    R['exact'][rule] = {
        'first_player_24': fr(P[N][N]), 'first_player_23': fr(P[23][23]), 'first_move_options': BEST[N][N],
        'matchups_first_player_wins': match, 'win_prob': grid_p, 'best_moves': grid_best, 'perfect_move': canon,
        'halving_not_optimal': not_half, 'positions': (N + 1 - lo) ** 2,
        'second_player_start_for_fair_game': {m: float(P[N][m]) for m in range(12, N + 1)},
    }
    print(f'{rule:11s} first player wins {P[N][N]} = {float(P[N][N]):.4f};  halving is a mistake in '
          f'{len(not_half)} of {(N + 1 - lo) ** 2} positions')
assert X.evaluate('official', X.halver, perfect_official, N)[N][N] == P_OFF[N][N]          # halving loses nothing
assert X.evaluate('official', perfect_official, X.halver, N)[N][N] == P_OFF[N][N]
# the worked example: 16 suspects against 4
R['example_16_vs_4'] = {
    'official_any_split': fr(P_OFF[16][4]),
    'instant_perfect_vs_perfect': fr(P_INS[16][4]),
    'instant_halving_vs_halver': fr(X.evaluate('instant', halver_instant, halver_instant, N)[16][4]),
    'instant_bold_vs_halver': fr(X.evaluate('instant', bold, halver_instant, N)[16][4]),
}
print('exact part:', round(time.time() - t0, 1), 's')

# ------------------------------------------------------------------ 2. tournament
Z = np.zeros((N + 1, N + 1), np.int64)
FEAT = {'random_feature': (sim.RANDOM_FEATURE, Z), 'best_feature': (sim.BEST_FEATURE, Z)}
LINEUP = {
    'official': dict(FEAT, halver=(sim.TABLE, table(X.halver)), perfect=(sim.TABLE, table(perfect_official)),
                     bold=(sim.TABLE, table(bold))),
    'free_guess': {'halver': (sim.TABLE, table(halver_free)), 'perfect': (sim.TABLE, table(perfect_free))},
    'instant': {'halver': (sim.TABLE, table(halver_instant, 2)), 'perfect': (sim.TABLE, table(bold, 2))},
}
RULE_ID = {'official': sim.OFFICIAL, 'free_guess': sim.FREE_GUESS, 'instant': sim.INSTANT}
sim.tournament(1000, 1, 0, 0, Z, 0, Z, board.MASKS)        # compile


def run(rule, players, masks, tag):
    out = {}
    for a in players:
        out[a] = {}
        for b in players:
            t = time.time()
            w, h = sim.tournament(GAMES, SEED, RULE_ID[rule], players[a][0], players[a][1], players[b][0], players[b][1], masks)
            assert h.sum() == GAMES and h[sim.MAXT - 1] == 0
            turns = float((h * np.arange(sim.MAXT)).sum() / GAMES)
            out[a][b] = {'first_wins': int(w), 'rate': w / GAMES, 'mean_turns': turns, 'turn_hist': h[:40].tolist()}
            print(f'{tag:8s} {rule:10s} {a:15s} vs {b:15s} first wins {w / GAMES:.5f}  ({time.time() - t:.1f}s)', flush=True)
    return out


t0 = time.time()
R['tournament'] = {rule: run(rule, LINEUP[rule], board.MASKS, 'board') for rule in X.RULES}
cl = {k: v for k, v in LINEUP['official'].items() if k != 'bold'}
R['tournament_classic_features'] = run('official', cl, classic.MASKS, 'classic')
pairings = sum(len(v) ** 2 for v in LINEUP.values()) + len(cl) ** 2
R['total_games'] = pairings * GAMES
print('pairings', pairings, 'games', R['total_games'], 'seconds', round(time.time() - t0))

# simulation against the exact values (strategies that only depend on the two counts)
chk = []
for rule in X.RULES:
    for a in STRATS[rule]:
        for b in STRATS[rule]:
            if a in LINEUP[rule] and b in LINEUP[rule]:
                ex = R['exact'][rule]['matchups_first_player_wins'][a][b]['value']
                mc = R['tournament'][rule][a][b]['rate']
                chk.append({'rule': rule, 'first': a, 'second': b, 'exact': ex, 'simulated': mc, 'diff': mc - ex})
R['check'] = chk
print('largest gap between simulation and exact value:', max(abs(c['diff']) for c in chk))

# how many questions each kind of player needs, with nobody racing
R['solo'] = {}
for tag, masks in (('board', board.MASKS), ('classic', classic.MASKS)):
    R['solo'][tag] = {}
    for a in ('random_feature', 'best_feature', 'halver'):
        h = sim.solo_many(GAMES, SEED + 1, LINEUP['official'][a][0], LINEUP['official'][a][1], masks)
        R['solo'][tag][a] = {'mean': float((h * np.arange(sim.MAXT)).sum() / GAMES), 'hist': h[:20].tolist()}
        print('solo', tag, a, round(R['solo'][tag][a]['mean'], 4))

json.dump(R, open(os.path.join(HERE, 'resultados.json'), 'w'), indent=1)
print('done')
