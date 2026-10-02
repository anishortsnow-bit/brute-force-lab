"""Klondike (draw 1, unlimited passes through the stock) in Numba.

Two kinds of players:
  * play():  strategies that only see what a human sees (face-down tableau cards stay unknown).
             In draw-1 with unlimited redeals every stock/waste card is reachable, so the stock is
             modelled as a set of cards that can be played at any time.
  * solve(): "thoughtful" solver that knows every card. Depth-first search with a transposition
             table, safe auto-moves to the foundations and a node limit per deal. Deals that hit the
             limit are reported as unknown, never as won or lost.

Cards: c = suit * 13 + rank, rank 0 = ace ... 12 = king; suits 0,2 black and 1,3 red.
Deal n is reproducible: the shuffle uses splitmix64 seeded with n.
"""
import numpy as np
from numba import njit, prange

MAXC = 20      # cards in one column (6 face-down + 13 face-up, plus slack)
MAXD = 700     # max search depth
MAXM = 160     # max moves at one node
MAXA = 64      # max auto-moves at one node
U = np.uint64


@njit(inline='always')
def suit(c):
    return c // 13


@njit(inline='always')
def rank(c):
    return c % 13


@njit(inline='always')
def color(c):
    return (c // 13) & 1


@njit(inline='always')
def mix64(z):
    z = (z ^ (z >> U(30))) * U(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> U(27))) * U(0x94D049BB133111EB)
    return z ^ (z >> U(31))


@njit
def deal(n, cols, lens, hid, talon, found):
    deck = np.arange(52)
    s = U(n) * U(0x9E3779B97F4A7C15) + U(0x632BE59BD9B4E019)
    for i in range(51, 0, -1):
        s = s + U(0x9E3779B97F4A7C15)
        j = int(mix64(s) % U(i + 1))
        t = deck[i]; deck[i] = deck[j]; deck[j] = t
    k = 0
    for c in range(7):
        for r in range(c + 1):
            cols[c, r] = deck[k]; k += 1
        lens[c] = c + 1
        hid[c] = c
    talon[:] = 0
    for i in range(k, 52):
        talon[deck[i]] = 1
    found[:] = 0


@njit(inline='always')
def fits(card, cols, lens, d):
    if lens[d] == 0:
        return rank(card) == 12
    t = cols[d, lens[d] - 1]
    return rank(t) == rank(card) + 1 and color(t) != color(card)


@njit(inline='always')
def safe(card, found):
    """Moving card to its foundation can never hurt: no card still in play would need it as a base."""
    s = suit(card); r = rank(card)
    if found[s] != r:
        return False
    if r <= 1:
        return True
    return found[s ^ 1] >= r and found[s ^ 3] >= r


@njit(inline='always')
def flip_after_take(s, lens, hid):
    if lens[s] > 0 and hid[s] == lens[s]:
        hid[s] -= 1
        return 1
    return 0


# move types: 0 tableau->foundation (a=col, b=card) | 1 stock->foundation (a=card)
#             2 stock->tableau (a=card, b=col)     | 3 tableau->tableau (a=src, b=dst, k=count)
#             4 foundation->tableau (a=suit, b=col)
@njit
def apply(t, a, b, k, cols, lens, hid, talon, found):
    if t == 0:
        lens[a] -= 1; found[suit(b)] += 1
        return flip_after_take(a, lens, hid)
    elif t == 1:
        talon[a] = 0; found[suit(a)] += 1
    elif t == 2:
        talon[a] = 0; cols[b, lens[b]] = a; lens[b] += 1
    elif t == 3:
        for i in range(k):
            cols[b, lens[b] + i] = cols[a, lens[a] - k + i]
        lens[b] += k; lens[a] -= k
        return flip_after_take(a, lens, hid)
    else:
        found[a] -= 1; cols[b, lens[b]] = a * 13 + found[a]; lens[b] += 1
    return 0


@njit
def undo(t, a, b, k, fl, cols, lens, hid, talon, found):
    if t == 0:
        hid[a] += fl
        cols[a, lens[a]] = b; lens[a] += 1; found[suit(b)] -= 1
    elif t == 1:
        talon[a] = 1; found[suit(a)] -= 1
    elif t == 2:
        lens[b] -= 1; talon[a] = 1
    elif t == 3:
        hid[a] += fl
        for i in range(k):
            cols[a, lens[a] + i] = cols[b, lens[b] - k + i]
        lens[a] += k; lens[b] -= k
    else:
        lens[b] -= 1; found[a] += 1


@njit
def gen_moves(out, cols, lens, hid, talon, found, solver):
    """Writes candidate moves into out[:, 0:4] in priority order and returns how many.
    solver=True also includes moves that only make sense with lookahead (foundation->tableau,
    emptying a fully revealed column)."""
    n = 0
    # 1. to the foundations
    for s in range(7):
        if lens[s] > 0:
            c = cols[s, lens[s] - 1]
            if found[suit(c)] == rank(c):
                out[n, 0] = 0; out[n, 1] = s; out[n, 2] = c; out[n, 3] = 1; n += 1
    for su in range(4):
        if found[su] < 13:
            c = su * 13 + found[su]
            if talon[c]:
                out[n, 0] = 1; out[n, 1] = c; out[n, 2] = 0; out[n, 3] = 1; n += 1
    # 2. moves that turn over a face-down card (whole face-up run of a column with face-down cards)
    for s in range(7):
        if hid[s] > 0 and lens[s] > hid[s]:
            i = hid[s]; c = cols[s, i]; k = lens[s] - i
            empty_done = False
            for d in range(7):
                if d != s and fits(c, cols, lens, d):
                    if lens[d] == 0:
                        if empty_done:
                            continue
                        empty_done = True
                    out[n, 0] = 3; out[n, 1] = s; out[n, 2] = d; out[n, 3] = k; n += 1
    # 3. stock -> tableau
    empty_done = False
    for d in range(7):
        if lens[d] == 0:
            if empty_done:
                continue
            empty_done = True
            for su in range(4):
                c = su * 13 + 12
                if talon[c] and n < MAXM - 8 and (not solver or needs_base(c, cols, lens, hid, talon, found)):
                    out[n, 0] = 2; out[n, 1] = c; out[n, 2] = d; out[n, 3] = 1; n += 1
        else:
            top = cols[d, lens[d] - 1]
            r = rank(top) - 1
            if r >= 0:
                for su in range(4):
                    if (su & 1) != color(top):
                        c = su * 13 + r
                        if talon[c] and n < MAXM - 8 and (not solver or needs_base(c, cols, lens, hid, talon, found)):
                            out[n, 0] = 2; out[n, 1] = c; out[n, 2] = d; out[n, 3] = 1; n += 1
    # 4. partial runs that uncover a card ready for the foundations; whole runs that empty a column
    for s in range(7):
        for i in range(hid[s] + 1, lens[s]):
            under = cols[s, i - 1]
            if found[suit(under)] != rank(under):
                continue
            c = cols[s, i]; k = lens[s] - i
            for d in range(7):
                if d != s and lens[d] > 0 and fits(c, cols, lens, d) and n < MAXM - 8:
                    out[n, 0] = 3; out[n, 1] = s; out[n, 2] = d; out[n, 3] = k; n += 1
        # emptying a fully revealed column only helps if some king could then use the space
        if solver and hid[s] == 0 and lens[s] > 0 and king_waiting(cols, lens, hid, talon):
            c = cols[s, 0]; k = lens[s]
            for d in range(7):
                if d != s and lens[d] > 0 and fits(c, cols, lens, d) and n < MAXM - 8:
                    out[n, 0] = 3; out[n, 1] = s; out[n, 2] = d; out[n, 3] = k; n += 1
    # 5. foundation -> tableau, only when the card would then hold something that needs a base
    if solver:
        for su in range(4):
            if found[su] >= 2:
                c = su * 13 + found[su] - 1
                if not needs_base(c, cols, lens, hid, talon, found):
                    continue
                for d in range(7):
                    if lens[d] > 0 and fits(c, cols, lens, d) and n < MAXM - 8:
                        out[n, 0] = 4; out[n, 1] = su; out[n, 2] = d; out[n, 3] = 1; n += 1
    return n


@njit
def king_waiting(cols, lens, hid, talon):
    """A king in the stock, or a king heading a face-up run that sits on face-down cards."""
    for su in range(4):
        if talon[su * 13 + 12]:
            return True
    for s in range(7):
        if hid[s] > 0 and lens[s] > hid[s] and rank(cols[s, hid[s]]) == 12:
            return True
    return False


@njit
def needs_base(c, cols, lens, hid, talon, found):
    """Some visible card of rank(c)-1 and the other colour (stock, face-up tableau, foundation top)
    could be put on c. If none is visible, putting c down now can always wait: in draw-1 the stock
    stays fully available, so this pruning never loses a win."""
    r = rank(c) - 1
    if r < 0:
        return False
    for su in range(4):
        if (su & 1) != color(c):
            x = su * 13 + r
            if talon[x] or found[su] == r + 1:
                return True
            for s in range(7):
                for i in range(hid[s], lens[s]):
                    if cols[s, i] == x:
                        return True
    return False


# ---------------------------------------------------------------- players without peeking
@njit
def careful_pick(out, nm, cols, lens, hid, talon, found):
    """Rules of thumb a careful human uses: safe cards up first, then dig into the column with the
    most face-down cards, play stock cards that let a buried run move, and only then the rest."""
    best = 0; bs = -1
    for m in range(nm):
        t = out[m, 0]; a = out[m, 1]; b = out[m, 2]; k = out[m, 3]
        if t == 0 or t == 1:
            c = b if t == 0 else a
            sc = 100 if safe(c, found) else 30
        elif t == 3:
            if hid[a] > 0 and k == lens[a] - hid[a]:
                sc = 50 + hid[a]
                if lens[b] == 0:
                    sc -= 5          # spend an empty column only when nothing else digs
            else:
                sc = 20
        else:                        # stock -> tableau
            sc = 10
            for s in range(7):
                if hid[s] > 0 and lens[s] > hid[s]:
                    x = cols[s, hid[s]]
                    if rank(x) + 1 == rank(a) and color(x) != color(a):
                        sc = 40
            if rank(a) == 12 and sc < 40:
                sc = 5
        if sc > bs:
            bs = sc; best = m
    return best


@njit
def play(n_deal, strategy, cols, lens, hid, talon, found, out):
    """strategy 0: random sensible move; 1: greedy (first move in priority order); 2: careful.
    Returns (won, cards on foundations, moves made, legal moves at the deal)."""
    deal(n_deal, cols, lens, hid, talon, found)
    first = gen_moves(out, cols, lens, hid, talon, found, False)
    rs = U(n_deal) * U(0xD1B54A32D192ED03) + U(strategy + 1)
    moves = 0
    while True:
        nm = gen_moves(out, cols, lens, hid, talon, found, False)
        if nm == 0:
            break
        if strategy == 0:
            rs = rs + U(0x9E3779B97F4A7C15)
            m = np.int64(mix64(rs) % U(nm))
        elif strategy == 1:
            m = 0
        else:
            m = careful_pick(out, nm, cols, lens, hid, talon, found)
        apply(out[m, 0], out[m, 1], out[m, 2], out[m, 3], cols, lens, hid, talon, found)
        moves += 1
    f = found[0] + found[1] + found[2] + found[3]
    return f == 52, f, moves, first


@njit(parallel=True)
def play_many(start, count, strategy, nthreads):
    won = np.zeros(count, np.bool_)
    fnd = np.zeros(count, np.int8)
    first = np.zeros(count, np.int16)
    per = (count + nthreads - 1) // nthreads
    for t in prange(nthreads):
        cols = np.zeros((7, MAXC), np.int64); lens = np.zeros(7, np.int64); hid = np.zeros(7, np.int64)
        talon = np.zeros(52, np.int64); found = np.zeros(4, np.int64); out = np.zeros((MAXM, 4), np.int64)
        for i in range(t * per, min(count, (t + 1) * per)):
            w, f, mv, fm = play(start + i, strategy, cols, lens, hid, talon, found, out)
            won[i] = w; fnd[i] = f; first[i] = fm
    return won, fnd, first


# ---------------------------------------------------------------- thoughtful solver
@njit
def state_hash(cols, lens, hid, talon, found):
    h = U(0)
    for c in range(7):
        x = U(hid[c] + 1)
        for i in range(lens[c]):
            x = mix64(x * U(131) + U(cols[c, i] + 1))
        h += mix64(x + U(0x1234567))          # sum: the order of the columns does not matter
    t = U(0)
    for i in range(52):
        if talon[i]:
            t |= U(1) << U(i)
    h ^= mix64(t + U(77))
    f = U(found[0] + 14 * found[1] + 196 * found[2] + 2744 * found[3])
    h ^= mix64(f * U(1000003) + U(99))
    return h


@njit
def tt_seen(keys, gens, gen, h):
    mask = keys.size - 1
    i = np.int64(h & U(mask))
    while True:
        if gens[i] != gen:
            gens[i] = gen; keys[i] = h
            return False
        if keys[i] == h:
            return True
        i = np.int64((i + 1) & mask)


@njit
def automove(depth, astack, cols, lens, hid, talon, found):
    na = 0
    changed = True
    while changed and na < MAXA - 1:
        changed = False
        for s in range(7):
            if lens[s] > 0:
                c = cols[s, lens[s] - 1]
                if safe(c, found):
                    fl = apply(0, s, c, 1, cols, lens, hid, talon, found)
                    astack[depth, na, 0] = 0; astack[depth, na, 1] = s; astack[depth, na, 2] = c; astack[depth, na, 3] = fl
                    na += 1; changed = True
        for su in range(4):
            if found[su] < 13:
                c = su * 13 + found[su]
                if talon[c] and safe(c, found):
                    apply(1, c, 0, 1, cols, lens, hid, talon, found)
                    astack[depth, na, 0] = 1; astack[depth, na, 1] = c; astack[depth, na, 2] = 0; astack[depth, na, 3] = 0
                    na += 1; changed = True
    return na


@njit
def undo_auto(depth, na, astack, cols, lens, hid, talon, found):
    for j in range(na - 1, -1, -1):
        undo(astack[depth, j, 0], astack[depth, j, 1], astack[depth, j, 2], 1, astack[depth, j, 3],
             cols, lens, hid, talon, found)


@njit
def search(depth, cols, lens, hid, talon, found, keys, gens, gen, moves, astack, ctr, limit):
    ctr[0] += 1
    if ctr[0] > limit:
        return -1
    if depth >= MAXD - 1:
        return 0
    na = automove(depth, astack, cols, lens, hid, talon, found)
    if found[0] + found[1] + found[2] + found[3] == 52:
        ctr[1] = depth
        return 1
    if tt_seen(keys, gens, gen, state_hash(cols, lens, hid, talon, found)):
        undo_auto(depth, na, astack, cols, lens, hid, talon, found)
        return 0
    nm = gen_moves(moves[depth], cols, lens, hid, talon, found, True)
    for m in range(nm):
        t = moves[depth, m, 0]; a = moves[depth, m, 1]; b = moves[depth, m, 2]; k = moves[depth, m, 3]
        fl = apply(t, a, b, k, cols, lens, hid, talon, found)
        r = search(depth + 1, cols, lens, hid, talon, found, keys, gens, gen, moves, astack, ctr, limit)
        if r != 0:
            return r
        undo(t, a, b, k, fl, cols, lens, hid, talon, found)
    undo_auto(depth, na, astack, cols, lens, hid, talon, found)
    return 0


@njit(parallel=True)
def solve_many(start, count, limit, nthreads, tbits):
    """Returns per deal: result (1 won, 0 proven lost, -1 node limit), nodes searched."""
    res = np.zeros(count, np.int8)
    nodes = np.zeros(count, np.int64)
    per = (count + nthreads - 1) // nthreads
    for t in prange(nthreads):
        cols = np.zeros((7, MAXC), np.int64); lens = np.zeros(7, np.int64); hid = np.zeros(7, np.int64)
        talon = np.zeros(52, np.int64); found = np.zeros(4, np.int64)
        moves = np.zeros((MAXD, MAXM, 4), np.int64); astack = np.zeros((MAXD, MAXA, 4), np.int64)
        keys = np.zeros(1 << tbits, np.uint64); gens = np.zeros(1 << tbits, np.int32)
        ctr = np.zeros(2, np.int64)
        for i in range(t * per, min(count, (t + 1) * per)):
            deal(start + i, cols, lens, hid, talon, found)
            ctr[0] = 0; ctr[1] = 0
            res[i] = search(0, cols, lens, hid, talon, found, keys, gens, i + 1, moves, astack, ctr, limit)
            nodes[i] = ctr[0]
    return res, nodes


@njit(parallel=True)
def solve_ids(ids, limit, nthreads, tbits):
    """Same as solve_many, for an explicit list of deal numbers (second pass on the unknowns)."""
    count = ids.size
    res = np.zeros(count, np.int8)
    nodes = np.zeros(count, np.int64)
    per = (count + nthreads - 1) // nthreads
    for t in prange(nthreads):
        cols = np.zeros((7, MAXC), np.int64); lens = np.zeros(7, np.int64); hid = np.zeros(7, np.int64)
        talon = np.zeros(52, np.int64); found = np.zeros(4, np.int64)
        moves = np.zeros((MAXD, MAXM, 4), np.int64); astack = np.zeros((MAXD, MAXA, 4), np.int64)
        keys = np.zeros(1 << tbits, np.uint64); gens = np.zeros(1 << tbits, np.int32)
        ctr = np.zeros(2, np.int64)
        for i in range(t, count, nthreads):
            deal(ids[i], cols, lens, hid, talon, found)
            ctr[0] = 0; ctr[1] = 0
            res[i] = search(0, cols, lens, hid, talon, found, keys, gens, i + 1, moves, astack, ctr, limit)
            nodes[i] = ctr[0]
    return res, nodes


@njit(parallel=True)
def legal_at_start(start, count):
    """Number of legal moves in the starting position of each deal (any move the rules allow)."""
    out = np.zeros(count, np.int16)
    for i in prange(count):
        cols = np.zeros((7, MAXC), np.int64); lens = np.zeros(7, np.int64); hid = np.zeros(7, np.int64)
        talon = np.zeros(52, np.int64); found = np.zeros(4, np.int64)
        deal(start + i, cols, lens, hid, talon, found)
        n = 0
        for s in range(7):
            c = cols[s, lens[s] - 1]
            if rank(c) == 0:
                n += 1
            for d in range(7):
                if d != s and fits(c, cols, lens, d):
                    n += 1
        for c in range(52):
            if talon[c]:
                if rank(c) == 0:
                    n += 1
                for d in range(7):
                    if fits(c, cols, lens, d):
                        n += 1
        out[i] = n
    return out
