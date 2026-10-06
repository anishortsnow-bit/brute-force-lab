"""Guess Who? solved exactly, as a race between two shrinking lists of suspects.

A position is (n, m): the player to move still has n suspects, the opponent has m.
A question splits the mover's n suspects into b ("yes") and n - b ("no"); the mystery
person is in the "yes" group with probability b / n. Any split is possible, because a
question can be about any set of faces ("Is your person one of these six?").

Three rule sets:
  official    A guess uses your turn. A wrong guess loses the game. (Hasbro's rules;
              the model of Cushing, Gipp, Levick, Rickinson & Stewart, arXiv:2508.00799,
              and Variant I of O'Neill, PLOS ONE 2021.)
  free_guess  A guess uses your turn, but a wrong guess only removes that face.
              (Variant II of O'Neill 2021.)
  instant     The moment you are down to one suspect, you win. No extra turn needed.
              (The model of Nica, arXiv:1509.03327.)

All probabilities are exact fractions. Moves: 0 = guess, b >= 1 = ask a question with
b suspects on the "yes" side (1 <= b <= n // 2).
"""
from fractions import Fraction as F
import json, math, os, sys

RULES = ('official', 'free_guess', 'instant')


def move_values(rule, n, m, P):
    """Win probability of every legal move at (n, m), given the table P of optimal values."""
    out = {}
    if rule == 'instant':
        for b in range(1, n // 2 + 1):
            v = F(0)
            for k in (b, n - b):
                v += F(k, n) * (F(1) if k == 1 else 1 - P[m][k])
            out[b] = v
        return out
    if n == 1:
        return {0: F(1)}
    if rule == 'official':
        out[0] = F(1, n)
    else:
        out[0] = F(1, n) + F(n - 1, n) * (1 - P[m][n - 1])
    for b in range(1, n // 2 + 1):
        out[b] = 1 - F(b, n) * P[m][b] - F(n - b, n) * P[m][n - b]
    return out


def solve(rule, N=24):
    """P[n][m] = chance that the player to move wins with best play on both sides.
    BEST[n][m] = every move that reaches that value."""
    lo = 2 if rule == 'instant' else 1
    P = [[None] * (N + 1) for _ in range(N + 1)]
    BEST = [[None] * (N + 1) for _ in range(N + 1)]
    for s in range(2 * lo, 2 * N + 1):
        for n in range(lo, N + 1):
            m = s - n
            if m < lo or m > N:
                continue
            vals = move_values(rule, n, m, P)
            top = max(vals.values())
            P[n][m] = top
            BEST[n][m] = sorted(k for k, v in vals.items() if v == top)
    return P, BEST


def evaluate(rule, pol_a, pol_b, N=24):
    """Exact chance that A wins when A moves first from (n, m), for two fixed strategies.
    A strategy is a function (n, m) -> move. Returns the table E[n][m]."""
    lo = 2 if rule == 'instant' else 1
    EA = [[None] * (N + 1) for _ in range(N + 1)]      # A to move, A has n, B has m
    EB = [[None] * (N + 1) for _ in range(N + 1)]      # B to move, B has n, A has m

    def step(pol, n, m, other):
        if rule == 'instant':
            b = pol(n, m)
            v = F(0)
            for k in (b, n - b):
                v += F(k, n) * (F(1) if k == 1 else 1 - other[m][k])
            return v
        if n == 1:
            return F(1)
        b = pol(n, m)
        if b == 0:
            if rule == 'official':
                return F(1, n)
            return F(1, n) + F(n - 1, n) * (1 - other[m][n - 1])
        return 1 - F(b, n) * other[m][b] - F(n - b, n) * other[m][n - b]

    for s in range(2 * lo, 2 * N + 1):
        for n in range(lo, N + 1):
            m = s - n
            if m < lo or m > N:
                continue
            EA[n][m] = step(pol_a, n, m, EB)
            EB[n][m] = step(pol_b, n, m, EA)
    return EA


# ------------------------------------------------------------------ strategies
def halver(n, m):
    """Binary search: always split as evenly as possible. Guess only when one suspect is left,
    or (official rules) as a last resort when the opponent is about to win."""
    if n == 1 or m == 1:
        return 0
    return n // 2


def halver_stubborn(n, m):
    """Binary search that never takes a desperate guess."""
    return 0 if n == 1 else n // 2


def n_less(n):
    return 1 if n == 2 else n // 4 + (n + 1) // 4


def cushing(n, m):
    """Theorem 1.1 of Cushing et al. (2025): an optimal move under the official rules."""
    if n == 1 or m == 1:
        return 0
    if (n, m) == (4, 4):
        return 1
    if (n, m) == (6, 4):
        return 3
    if (n, m) == (10, 4):
        return 5
    return n_less(n)


def nica(n, m):
    """Theorem 1.1 of Nica (2016): optimal under the 'instant' rule. Bold when behind."""
    if n == 1 or m == 1:
        return 0
    k = int(math.floor(math.log2(m - 1))) if m > 1 else 0
    if n >= 2 ** (k + 1) + 1:
        return 2 ** k
    return n // 2


def nica_value(n, m):
    """Closed form for the mover's win probability in Nica's model."""
    k = 0
    while not (2 ** k + 1 <= m <= 2 ** (k + 1)):
        k += 1
    if n >= 2 ** (k + 1) + 1:
        return F(2 ** (k + 1), n) - F(2, 3) * F(2 ** (2 * k + 1) + 1, n * m)
    k = 0
    while not (2 ** k + 1 <= n <= 2 ** (k + 1)):
        k += 1
    return 1 - F(2 ** k, m) + F(2, 3) * F(2 ** (2 * k) + 2, n * m)


def table_policy(BEST, prefer):
    """Turn a table of optimal moves into one strategy, breaking ties with `prefer`."""
    def pol(n, m):
        opts = BEST[n][m]
        p = prefer(n, m)
        return p if p in opts else opts[-1]
    return pol


if __name__ == '__main__':
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    res = {}
    for rule in RULES:
        P, BEST = solve(rule, N)
        res[rule] = (P, BEST)
        print(f'{rule:11s} first player wins from (24,24): {P[24][24]} = {float(P[24][24]):.6f}'
              f'   from (23,23): {float(P[23][23]):.6f}   best first move: {BEST[24][24]}')
