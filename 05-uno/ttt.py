"""Tic-tac-toe on the luck meter, solved exactly: the best possible player against a player who picks a random
empty square. The best player maximises its score (win = 1, draw = 1/2), which is what the meter counts."""
from fractions import Fraction
from functools import lru_cache

LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]


def winner(b):
    for a, c, d in LINES:
        if b[a] != '.' and b[a] == b[c] == b[d]:
            return b[a]
    return None


@lru_cache(None)
def value(b, turn, me):
    """Exact (P(me wins), P(draw)) when 'me' plays the best response and the other player is uniformly random."""
    w = winner(b)
    if w:
        return (Fraction(1), Fraction(0)) if w == me else (Fraction(0), Fraction(0))
    empty = [i for i in range(9) if b[i] == '.']
    if not empty:
        return Fraction(0), Fraction(1)
    nxt = 'O' if turn == 'X' else 'X'
    kids = [value(b[:i] + turn + b[i + 1:], nxt, me) for i in empty]
    if turn == me:
        return max(kids, key=lambda v: v[0] + v[1] / 2)
    n = len(kids)
    return sum(k[0] for k in kids) / n, sum(k[1] for k in kids) / n


def random_vs_random(b='.........', turn='X'):
    w = winner(b)
    if w:
        return {w: Fraction(1)}
    empty = [i for i in range(9) if b[i] == '.']
    if not empty:
        return {'draw': Fraction(1)}
    out = {}
    nxt = 'O' if turn == 'X' else 'X'
    for i in empty:
        for k, v in random_vs_random(b[:i] + turn + b[i + 1:], nxt).items():
            out[k] = out.get(k, 0) + v / len(empty)
    return out


if __name__ == '__main__':
    res = {}
    for me in 'XO':
        w, d = value('.........', 'X', me)
        res[me] = (w, d)
        print(f'best as {me}: win {float(w):.4%}  draw {float(d):.4%}  lose {float(1 - w - d):.4%}  score {float(w + d / 2):.4%}')
    p = (res['X'][0] + res['X'][1] / 2 + res['O'][0] + res['O'][1] / 2) / 2
    print(f'both seats: score {float(p):.4%}  skill share 2p-1 = {float(2 * p - 1):.4%}  luck {float(2 - 2 * p):.4%}')
    rr = random_vs_random()
    print('random v random:', {k: f'{float(v):.4%}' for k, v in rr.items()})
