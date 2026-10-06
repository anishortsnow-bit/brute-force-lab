"""Monte Carlo tournament for Guess Who? on a real board (Numba).

Every game has two real mystery people and two real lists of suspects (24-bit masks). A question
is a set of faces; the answer comes from the opponent's mystery person. Nothing in the simulation
knows about the exact solution in exact.py, so the two can be checked against each other.

Players:
  RANDOM_FEATURE  asks a random single-feature question that still splits its suspects
  BEST_FEATURE    asks the single-feature question whose split is closest to half
  TABLE           follows a table move[n][m] (0 = guess, b = ask about b of its suspects);
                  this covers the halver, the perfect players and the bold player

Rules: 0 official (a wrong guess loses), 1 free guess (a wrong guess only costs the turn),
       2 instant (down to one suspect = you win on the spot).
"""
import numpy as np
from numba import njit, prange

OFFICIAL, FREE_GUESS, INSTANT = 0, 1, 2
RANDOM_FEATURE, BEST_FEATURE, TABLE = 0, 1, 2
NFACES = 24
FULL = (1 << NFACES) - 1
MAXT = 64
U = np.uint64


@njit(inline='always')
def mix64(z):
    z = (z ^ (z >> U(30))) * U(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> U(27))) * U(0x94D049BB133111EB)
    return z ^ (z >> U(31))


@njit(inline='always')
def popcount(x):
    x = x - ((x >> 1) & 0x55555555)
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
    x = (x + (x >> 4)) & 0x0F0F0F0F
    return (x * 0x01010101 >> 24) & 0xFF


@njit(inline='always')
def lowest(c, b):
    """Mask with the b lowest set bits of c."""
    out = 0
    for _ in range(b):
        bit = c & -c
        out |= bit
        c ^= bit
    return out


@njit(inline='always')
def kth_bit(c, k):
    """Index of the k-th set bit of c (k = 0 is the lowest)."""
    for _ in range(k):
        c &= c - 1
    bit = c & -c
    i = 0
    while bit > 1:
        bit >>= 1
        i += 1
    return i


@njit
def play(game, seed, rule, kind_a, tab_a, kind_b, tab_b, masks):
    """Plays one game. Returns (winner, turns): winner 0 = the player who moved first."""
    s = U(seed) * U(0xD1342543DE82EF95) + U(game) * U(0x9E3779B97F4A7C15) + U(0x632BE59BD9B4E019)
    s += U(0x9E3779B97F4A7C15); person_a = int(mix64(s) % U(NFACES))     # A's mystery person
    s += U(0x9E3779B97F4A7C15); person_b = int(mix64(s) % U(NFACES))
    sus_a = FULL                                                          # A's suspects for B's person
    sus_b = FULL
    nq = masks.shape[0]
    turn = 0
    t = 0
    while True:
        t += 1
        if turn == 0:
            c = sus_a; target = person_b; kind = kind_a; n = popcount(sus_a); m = popcount(sus_b)
        else:
            c = sus_b; target = person_a; kind = kind_b; n = popcount(sus_b); m = popcount(sus_a)
        q = 0
        if kind == TABLE:
            b = tab_a[n, m] if turn == 0 else tab_b[n, m]
            if b > 0:
                q = lowest(c, b)
        elif n > 1 and m > 1:
            best = -1; seen = 0
            for f in range(nq):
                k = popcount(c & masks[f])
                if k == 0 or k == n:
                    continue
                score = 0 if kind == RANDOM_FEATURE else abs(2 * k - n)
                if best < 0 or score < best:
                    best = score; seen = 1; q = c & masks[f]
                elif score == best:
                    seen += 1
                    s += U(0x9E3779B97F4A7C15)
                    if int(mix64(s) % U(seen)) == 0:
                        q = c & masks[f]
        if q == 0:                                                        # guess
            g = 0
            if n > 1:
                s += U(0x9E3779B97F4A7C15); g = int(mix64(s) % U(n))
            face = kth_bit(c, g)
            if face == target:
                return turn, t
            if rule == OFFICIAL:
                return 1 - turn, t
            c ^= 1 << face
        else:
            if (q >> target) & 1:
                c &= q
            else:
                c &= ~q
            if rule == INSTANT and popcount(c) == 1:
                return turn, t
        if turn == 0:
            sus_a = c
        else:
            sus_b = c
        turn = 1 - turn
        if t >= MAXT - 1:
            return -1, t


@njit(parallel=True)
def tournament(games, seed, rule, kind_a, tab_a, kind_b, tab_b, masks, chunks=500):
    """Plays `games` games (a multiple of `chunks`) with A moving first.
    Returns (wins of A, histogram of game length in turns)."""
    per = games // chunks
    wins = np.zeros(chunks, np.int64)
    hist = np.zeros((chunks, MAXT), np.int64)
    for ch in prange(chunks):
        w = 0
        for g in range(per):
            who, t = play(ch * per + g, seed, rule, kind_a, tab_a, kind_b, tab_b, masks)
            if who == 0:
                w += 1
            hist[ch, t] += 1
        wins[ch] = w
    return wins.sum(), hist.sum(axis=0)


@njit
def solo(game, seed, kind, tab, masks):
    """Number of questions one player needs to get from 24 suspects down to one (nobody racing)."""
    s = U(seed) * U(0xD1342543DE82EF95) + U(game) * U(0x9E3779B97F4A7C15) + U(0x632BE59BD9B4E019)
    s += U(0x9E3779B97F4A7C15); target = int(mix64(s) % U(NFACES))
    c = FULL
    nq = masks.shape[0]
    t = 0
    while popcount(c) > 1:
        n = popcount(c)
        q = 0
        if kind == TABLE:
            q = lowest(c, max(1, tab[n, NFACES]))
        else:
            best = -1; seen = 0
            for f in range(nq):
                k = popcount(c & masks[f])
                if k == 0 or k == n:
                    continue
                score = 0 if kind == RANDOM_FEATURE else abs(2 * k - n)
                if best < 0 or score < best:
                    best = score; seen = 1; q = c & masks[f]
                elif score == best:
                    seen += 1
                    s += U(0x9E3779B97F4A7C15)
                    if int(mix64(s) % U(seen)) == 0:
                        q = c & masks[f]
        if (q >> target) & 1:
            c &= q
        else:
            c &= ~q
        t += 1
    return t


@njit(parallel=True)
def solo_many(games, seed, kind, tab, masks, chunks=500):
    per = games // chunks
    hist = np.zeros((chunks, MAXT), np.int64)
    for ch in prange(chunks):
        for g in range(per):
            hist[ch, solo(ch * per + g, seed, kind, tab, masks)] += 1
    return hist.sum(axis=0)
