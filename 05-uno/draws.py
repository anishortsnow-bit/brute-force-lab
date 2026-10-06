"""When does the best player draw on purpose? Replays games of the main run (search_random in run_all.py: same
seed, same game numbers, same settings) and sorts every deliberate draw by what else the player could have played.
Writes draws.json."""
import json
import sys
from collections import Counter

import numpy as np

import engine as E
import run_all as RA
import trace as TR

T = TR.T
R = E.official(2)
SEED = RA.SEED + 13


def replay(g):
    seats = [3, 0] if g % 2 == 0 else [0, 3]
    G = TR.Game(R, seats, SEED, g, K=RA.K_MAIN, z=RA.Z_MAIN, X=RA.X)
    G.play()
    return G, g % 2


def classify(e):
    """What the player could have played instead of drawing."""
    legal = e['legal']
    col = e['color']
    kinds = [T[c, 2] for c in legal]
    if all(k >= E.K_WILD for k in kinds):
        return 'only wild cards' + (' (a Wild Draw Four)' if any(k == E.K_WD4 for k in kinds) else '')
    if any(T[c, 0] == col for c in legal):
        return 'a card of the same colour'
    return 'only colour changes'


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    turns = Counter()
    cats = Counter()
    keep = Counter()
    wins_by = Counter()
    games_with = 0
    wild_early = Counter()
    for g in range(n):
        G, me = replay(g)
        won = G.S[E.H_WIN] == me
        had = False
        for e in G.events:
            if e['kind'] != 'decision' or e['cur'] != me:
                continue
            if e['phase'] == E.PH_TURN:
                turns['decisions'] += 1
                if e['action'] < 0:
                    cats[classify(e)] += 1
                    had = True
                    opp = len(e['hands'][1 - me])
                    turns['draw_opp_' + ('1-2' if opp <= 2 else '3-5' if opp <= 5 else '6+')] += 1
                    if e['void'][1 - me][e['color']] >= opp > 0:
                        turns['draw_opp_void'] += 1
                elif T[e['action'] >> 2, 2] >= E.K_WILD:
                    nonwild = any(T[c, 2] < E.K_WILD for c in e['legal'])
                    wild_early['wild played, other cards playable' if nonwild else 'wild played, nothing else'] += 1
            elif e['phase'] == E.PH_AFTER:
                kd = T[e['drawn'], 2]
                lab = 'wild' if kd >= E.K_WILD else 'other'
                keep[(lab, 'kept' if e['action'] < 0 else 'played')] += 1
        games_with += had
        wins_by['with' if had else 'without', bool(won)] += 1
    out = dict(games=n, deliberate_draws=sum(cats.values()), by_alternative=dict(cats), turns=dict(turns),
               after_draw={f'{a} {b}': v for (a, b), v in keep.items()}, wilds=dict(wild_early),
               games_with_draw=games_with, wins={f'{a} {b}': v for (a, b), v in wins_by.items()})
    print(json.dumps(out, indent=1))
    json.dump(out, open('draws.json', 'w'), indent=1)
