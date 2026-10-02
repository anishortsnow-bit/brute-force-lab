"""Records real games move by move for the animations (careful player, or the solver's winning line).

Each state is {"cols": [[card, ...] x7], "hid": [..7], "stock": [cards], "found": [..4]}; each move is the
klondike.py tuple (type, a, b, k). Cards use klondike's encoding (suit*13 + rank, rank 0 = ace).
"""
import json, sys
import numpy as np
import klondike as K


def new_state(n):
    cols = np.zeros((7, K.MAXC), np.int64); lens = np.zeros(7, np.int64); hid = np.zeros(7, np.int64)
    talon = np.zeros(52, np.int64); found = np.zeros(4, np.int64)
    K.deal(n, cols, lens, hid, talon, found)
    return cols, lens, hid, talon, found


def snap(cols, lens, hid, talon, found):
    return {'cols': [[int(x) for x in cols[c, :lens[c]]] for c in range(7)], 'hid': [int(x) for x in hid],
            'stock': [int(i) for i in np.nonzero(talon)[0]], 'found': [int(x) for x in found]}


def careful_game(n, strategy=2, seed=0):
    st = new_state(n); out = np.zeros((K.MAXM, 4), np.int64); rng = np.random.default_rng(seed)
    states = [snap(*st)]; moves = []
    while True:
        nm = K.gen_moves(out, *st, False)
        if nm == 0:
            break
        m = int(rng.integers(nm)) if strategy == 0 else 0 if strategy == 1 else K.careful_pick(out, nm, *st)
        mv = [int(x) for x in out[m]]
        K.apply(mv[0], mv[1], mv[2], mv[3], *st)
        moves.append(mv); states.append(snap(*st))
    return {'deal': n, 'won': sum(states[-1]['found']) == 52, 'moves': moves, 'states': states}


def solver_line(n, limit=20_000_000, tbits=24):
    """Re-runs the solver on one deal and reports the result, positions searched and depth of the win."""
    st = new_state(n); start = snap(*st)
    moves = np.zeros((K.MAXD, K.MAXM, 4), np.int64); ast = np.zeros((K.MAXD, K.MAXA, 4), np.int64)
    keys = np.zeros(1 << tbits, np.uint64); gens = np.zeros(1 << tbits, np.int32); ctr = np.zeros(2, np.int64)
    r = K.search(0, *st, keys, gens, 1, moves, ast, ctr, limit)
    return {'deal': n, 'result': int(r), 'nodes': int(ctr[0]), 'depth': int(ctr[1]), 'start': start}


if __name__ == '__main__':
    n = int(sys.argv[1]); strat = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    g = careful_game(n, strat)
    print(n, 'won' if g['won'] else 'lost', len(g['moves']), 'moves, cards up:', sum(g['states'][-1]['found']))
    json.dump(g, open(f'game_{n}_s{strat}.json', 'w'))
