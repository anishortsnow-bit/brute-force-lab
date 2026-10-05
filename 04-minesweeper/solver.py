"""Minesweeper engine for experiment 04: how often does Minesweeper force you to guess?

A board is W x H cells with M mines. Cell i = row * W + col.
Cell state st[i]: 0 = covered, 1 = open, 2 = covered and known to be a mine (flag).

Three levels of deduction, always tried in this order:
  * the simple rules: a number whose flags are all found opens the rest; a number that needs exactly as many
    mines as it has covered neighbours flags them all; a mine counter at zero opens everything;
  * patterns: two numbers that share covered cells are compared (this covers the 1-1 and 1-2-1 patterns);
  * perfect logic: when the simple rules run dry, it enumerates every mine arrangement on the
    frontier that agrees with all the numbers and with the total number of mines. A cell is safe only if it
    holds no mine in any of them. Safety is decided with exact booleans (no floating point).
When no cell is provably safe the position is a FORCED GUESS, and one of four guessing rules picks a cell:
  RANDOM (any covered cell), FRONTIER (a random cell next to a number), SAFEST (lowest exact mine
  probability, random tie-break) and SMART (lowest probability; ties go to the cell with fewest covered
  neighbours, the best chance to open an area).

First-click rules: ANY (the first click can be a mine), SAFE (the first cell is never a mine: classic Windows)
and ZERO (the first cell and its neighbours are never mines, so the first click opens an area).
"""
import math
import numpy as np
from numba import njit, prange

R_ANY, R_SAFE, R_ZERO = 0, 1, 2
S_RANDOM, S_FRONTIER, S_SAFEST, S_SMART = 0, 1, 2, 3
RULES = {'any': R_ANY, 'safe': R_SAFE, 'zero': R_ZERO}
STRATS = {'random': S_RANDOM, 'frontier': S_FRONTIER, 'safest': S_SAFEST, 'smart': S_SMART}

# per-game output columns
O_WIN, O_GUESS, O_STUCK, O_HIDDEN, O_GLOBAL, O_CAPPED, O_FIRSTFRAC, O_OPENED, O_BOOM1, O_BASIC_HIDDEN1, O_IMMEDIATE, O_PAIR = range(12)
NOUT = 12
TIE = 1e-9


def neighbors(W, H):
    N = W * H
    NB = np.full((N, 8), -1, np.int32)
    NBC = np.zeros(N, np.int32)
    for r in range(H):
        for c in range(W):
            i = r * W + c; k = 0
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < H and 0 <= cc < W:
                        NB[i, k] = rr * W + cc; k += 1
            NBC[i] = k
    return NB, NBC


def binom_table():
    B = np.zeros((9, 9))
    for n in range(9):
        for k in range(n + 1):
            B[n, k] = math.comb(n, k)
    return B


# ------------------------------------------------------------------ random numbers (splitmix64)
@njit(inline='always')
def _next(rs):
    s = rs[0] + np.uint64(0x9E3779B97F4A7C15)
    rs[0] = s
    z = (s ^ (s >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


@njit(inline='always')
def _randint(rs, n):
    return np.int64(np.float64(_next(rs) >> np.uint64(11)) * (1.0 / 9007199254740992.0) * n)


@njit(inline='always')
def seed_game(rs, seed, game):
    rs[0] = np.uint64(seed) * np.uint64(0x9E3779B97F4A7C15) + np.uint64(game) * np.uint64(0xD1B54A32D192ED03)
    _next(rs); _next(rs)


# ------------------------------------------------------------------ board
@njit(cache=True)
def make_board(M, rule, first, rs, NB, NBC, mine, num, pool):
    N = mine.shape[0]
    n = 0
    for i in range(N):
        mine[i] = 0
        ex = False
        if rule == R_SAFE and i == first:
            ex = True
        elif rule == R_ZERO:
            if i == first:
                ex = True
            else:
                for k in range(NBC[first]):
                    if NB[first, k] == i:
                        ex = True
        if not ex:
            pool[n] = i; n += 1
    for j in range(M):                      # partial Fisher-Yates: M distinct cells, uniformly
        r = j + _randint(rs, n - j)
        t = pool[j]; pool[j] = pool[r]; pool[r] = t
        mine[pool[j]] = 1
    for i in range(N):
        s = 0
        for k in range(NBC[i]):
            s += mine[NB[i, k]]
        num[i] = s


@njit(cache=True)
def reveal(i, st, num, NB, NBC, stack):
    """Opens a cell that is not a mine, and the whole zero area around it. Returns how many cells opened."""
    if st[i] != 0:
        return 0
    st[i] = 1; stack[0] = i; top = 1; n = 1
    while top > 0:
        top -= 1; c = stack[top]
        if num[c] == 0:
            for k in range(NBC[c]):
                j = NB[c, k]
                if st[j] == 0:
                    st[j] = 1; n += 1; stack[top] = j; top += 1
    return n


@njit(cache=True)
def basic_pass(M, st, num, NB, NBC, stack, cnt):
    """The rules everybody uses. cnt[0] = open cells, cnt[1] = flags. Returns how many deductions were made."""
    N = st.shape[0]
    acts = 0
    changed = True
    while changed:
        changed = False
        for c in range(N):
            if st[c] != 1 or num[c] == 0:
                continue
            u = 0; f = 0
            for k in range(NBC[c]):
                s = st[NB[c, k]]
                if s == 0:
                    u += 1
                elif s == 2:
                    f += 1
            if u == 0:
                continue
            need = num[c] - f
            if need == 0:
                for k in range(NBC[c]):
                    j = NB[c, k]
                    if st[j] == 0:
                        cnt[0] += reveal(j, st, num, NB, NBC, stack)
                changed = True; acts += 1
            elif need == u:
                for k in range(NBC[c]):
                    j = NB[c, k]
                    if st[j] == 0:
                        st[j] = 2; cnt[1] += 1
                changed = True; acts += 1
        if cnt[1] == M:                     # mine counter at zero: everything left is safe
            for i in range(N):
                if st[i] == 0:
                    cnt[0] += reveal(i, st, num, NB, NBC, stack); changed = True; acts += 1
    return acts


@njit(cache=True)
def pair_pass(st, num, NB, NBC, stack, cnt, maxacts=1 << 30):
    """Patterns: compares every two numbers that share covered cells. A has na covered cells and needs nA mines,
    B has nb and needs nB; i cells are shared. With z mines in the shared cells, the cells only A touches hold
    nA - z and the cells only B touches hold nB - z, so the range of z decides what is certain.
    One sweep; returns the number of deductions made."""
    N = st.shape[0]
    acts = 0
    ca = np.empty(8, np.int32); cb = np.empty(8, np.int32)
    stamp = np.zeros(N, np.int32)
    for A in range(N):
        if st[A] != 1 or num[A] == 0:
            continue
        na = 0; fa = 0
        for k in range(NBC[A]):
            j = NB[A, k]
            if st[j] == 0:
                ca[na] = j; na += 1
            elif st[j] == 2:
                fa += 1
        if na == 0:
            continue
        nA = num[A] - fa
        done = False
        for t in range(na):
            if done:
                break
            x = ca[t]
            for k2 in range(NBC[x]):
                B = NB[x, k2]
                if B <= A or st[B] != 1 or stamp[B] == A + 1:
                    continue
                stamp[B] = A + 1
                nb = 0; fb = 0
                for k in range(NBC[B]):
                    j = NB[B, k]
                    if st[j] == 0:
                        cb[nb] = j; nb += 1
                    elif st[j] == 2:
                        fb += 1
                nB = num[B] - fb
                i = 0
                for p in range(na):
                    for q in range(nb):
                        if ca[p] == cb[q]:
                            i += 1
                a = na - i; b = nb - i
                zmin = max(0, nA - a, nB - b); zmax = min(i, nA, nB)
                xmax = nA - zmin; xmin = nA - zmax; ymax = nB - zmin; ymin = nB - zmax
                # 0 = nothing certain, 1 = open, 2 = flag; for A-only, B-only and shared cells
                da = 0; db = 0; dz = 0
                if a > 0 and xmax == 0:
                    da = 1
                elif a > 0 and xmin == a:
                    da = 2
                if b > 0 and ymax == 0:
                    db = 1
                elif b > 0 and ymin == b:
                    db = 2
                if zmin == zmax:
                    if zmin == 0:
                        dz = 1
                    elif zmin == i:
                        dz = 2
                if da == 0 and db == 0 and dz == 0:
                    continue
                for p in range(na):
                    c = ca[p]; shared = False
                    for q in range(nb):
                        if cb[q] == c:
                            shared = True
                    d = dz if shared else da
                    if st[c] == 0:
                        if d == 1:
                            cnt[0] += reveal(c, st, num, NB, NBC, stack)
                        elif d == 2:
                            st[c] = 2; cnt[1] += 1
                for q in range(nb):
                    c = cb[q]; shared = False
                    for p in range(na):
                        if ca[p] == c:
                            shared = True
                    if not shared and st[c] == 0:
                        if db == 1:
                            cnt[0] += reveal(c, st, num, NB, NBC, stack)
                        elif db == 2:
                            st[c] = 2; cnt[1] += 1
                acts += 1
                done = True
                break
        if acts >= maxacts:
            break
    return acts


# ------------------------------------------------------------------ perfect logic
@njit(inline='always')
def _box_range(b, sz, bsig, bnsig, cval, cur, rem):
    lo = 0; hi = sz
    for q in range(bnsig):
        c = bsig[b, q]
        need = cval[c] - cur[c]
        other = rem[c] - sz
        if need - other > lo:
            lo = need - other
        if need < hi:
            hi = need
    return lo, hi


@njit(cache=True)
def full_solve(M, st, num, NB, NBC, BIN, node_cap, safe_out, mine_out, prob, res):
    """Perfect logic on the current position.

    res[0] = cells proven safe (listed in safe_out), res[1] = cells proven to be mines (mine_out),
    res[2] = 1 if some safe cell could only be proven with the total mine count,
    res[3] = 1 if a frontier piece was too big to enumerate (then nothing in it is called safe),
    res[4] = search nodes used, res[5] = number of frontier pieces, res[6] = arrangements of the frontier (capped).
    When no cell is proven safe, prob[i] gets the exact chance that covered cell i holds a mine.
    """
    N = st.shape[0]
    # 1. constraints: open cells with covered neighbours
    cons_id = np.full(N, -1, np.int32)
    cons_cell = np.empty(N, np.int32)
    cval = np.empty(N, np.int32)
    ncons = 0
    known = 0; nunk = 0
    for c in range(N):
        s = st[c]
        if s == 2:
            known += 1
            continue
        if s == 0:
            nunk += 1
            continue
        u = 0; f = 0
        for k in range(NBC[c]):
            t = st[NB[c, k]]
            if t == 0:
                u += 1
            elif t == 2:
                f += 1
        if u > 0:
            cons_id[c] = ncons; cons_cell[ncons] = c; cval[ncons] = num[c] - f; ncons += 1
    Mr = M - known
    # 2. frontier cells and the sorted list of numbers each one touches
    sig = np.empty((N, 8), np.int32)
    sign = np.zeros(N, np.int32)
    front = np.empty(N, np.int32)
    nf = 0
    for i in range(N):
        if st[i] != 0:
            continue
        n = 0
        for k in range(NBC[i]):
            j = NB[i, k]
            if cons_id[j] >= 0:
                sig[i, n] = cons_id[j]; n += 1
        if n > 0:
            for a in range(1, n):
                v = sig[i, a]; b = a - 1
                while b >= 0 and sig[i, b] > v:
                    sig[i, b + 1] = sig[i, b]; b -= 1
                sig[i, b + 1] = v
            sign[i] = n; front[nf] = i; nf += 1
    U = nunk - nf
    # 3. boxes: frontier cells that touch exactly the same numbers are interchangeable
    box_of = np.full(N, -1, np.int32)
    bsize = np.zeros(nf + 1, np.int32)
    brep = np.empty(nf + 1, np.int32)
    nbox = 0
    for a in range(nf):
        i = front[a]
        if box_of[i] >= 0:
            continue
        b = nbox; nbox += 1
        box_of[i] = b; bsize[b] = 1; brep[b] = i
        c0 = cons_cell[sig[i, 0]]
        for k in range(NBC[c0]):
            j = NB[c0, k]
            if j == i or st[j] != 0 or box_of[j] >= 0 or sign[j] != sign[i]:
                continue
            same = True
            for q in range(sign[i]):
                if sig[j, q] != sig[i, q]:
                    same = False
                    break
            if same:
                box_of[j] = b; bsize[b] += 1
    bsig = np.empty((nbox + 1, 8), np.int32)
    bnsig = np.zeros(nbox + 1, np.int32)
    cbox = np.empty((ncons + 1, 8), np.int32)
    cnb = np.zeros(ncons + 1, np.int32)
    csize = np.zeros(ncons + 1, np.int32)
    for b in range(nbox):
        i = brep[b]
        bnsig[b] = sign[i]
        for q in range(sign[i]):
            c = sig[i, q]
            bsig[b, q] = c
            cbox[c, cnb[c]] = b; cnb[c] += 1; csize[c] += bsize[b]
    # 4. independent pieces of the frontier (breadth-first from a far end, so that numbers close early)
    comp_of = np.full(nbox + 1, -1, np.int32)
    seen = np.full(nbox + 1, -1, np.int32)
    order = np.empty(nbox + 1, np.int32)
    cstart = np.zeros(nbox + 2, np.int32)
    ncomp = 0; pos = 0
    tmp = np.empty(nbox + 1, np.int32)
    for b0 in range(nbox):
        if comp_of[b0] >= 0:
            continue
        # first pass: find the piece and its last-reached box
        seen[b0] = ncomp; tmp[0] = b0; h = 0; tl = 1
        while h < tl:
            b = tmp[h]; h += 1
            for q in range(bnsig[b]):
                c = bsig[b, q]
                for t in range(cnb[c]):
                    b2 = cbox[c, t]
                    if seen[b2] != ncomp:
                        seen[b2] = ncomp; tmp[tl] = b2; tl += 1
        start = tmp[tl - 1]
        cstart[ncomp] = pos
        comp_of[start] = ncomp; order[pos] = start; h = pos; pos += 1
        while h < pos:
            b = order[h]; h += 1
            for q in range(bnsig[b]):
                c = bsig[b, q]
                for t in range(cnb[c]):
                    b2 = cbox[c, t]
                    if comp_of[b2] < 0:
                        comp_of[b2] = ncomp; order[pos] = b2; pos += 1
        ncomp += 1
    cstart[ncomp] = pos
    res[5] = ncomp
    # 5. enumerate every arrangement of each piece (boxes take 0..size mines, weight C(size, m))
    K = Mr + 1
    ccnt = np.zeros((ncomp + 1, K))
    cfeas = np.zeros((ncomp + 1, K), np.bool_)
    capped = np.zeros(ncomp + 1, np.bool_)
    wmine = np.zeros((nbox + 1, K))
    canm = np.zeros((nbox + 1, K), np.bool_)
    cans = np.zeros((nbox + 1, K), np.bool_)
    canm_loc = np.zeros(nbox + 1, np.bool_)        # a mine here fits the numbers, ignoring the mine counter
    cur = np.zeros(ncons + 1, np.int32)
    rem = csize.copy()
    mcur = np.empty(nbox + 1, np.int32)
    hid = np.empty(nbox + 1, np.int32)
    asg = np.zeros(nbox + 1, np.bool_)
    wpre = np.empty(nbox + 2)
    kpre = np.empty(nbox + 2, np.int32)
    total_nodes = 0
    narr = 0.0
    any_capped = False
    for cc in range(ncomp):
        s0 = cstart[cc]; nb = cstart[cc + 1] - s0
        wpre[0] = 1.0; kpre[0] = 0
        b = order[s0]
        lo, hi = _box_range(b, bsize[b], bsig, bnsig[b], cval, cur, rem)
        d = 0; mcur[0] = lo - 1; hid[0] = hi; asg[0] = False
        nodes = 0
        while d >= 0:
            b = order[s0 + d]
            sz = bsize[b]
            if asg[d]:
                m = mcur[d]
                for q in range(bnsig[b]):
                    c = bsig[b, q]; cur[c] -= m; rem[c] += sz
                asg[d] = False
            mcur[d] += 1
            if mcur[d] > hid[d]:
                d -= 1
                continue
            m = mcur[d]
            for q in range(bnsig[b]):
                c = bsig[b, q]; cur[c] += m; rem[c] -= sz
            asg[d] = True
            nodes += 1
            if nodes > node_cap:
                capped[cc] = True
                break
            wpre[d + 1] = wpre[d] * BIN[sz, m]
            kpre[d + 1] = kpre[d] + m
            if d == nb - 1:
                k = kpre[nb]; w = wpre[nb]
                for e in range(nb):
                    if mcur[e] > 0:
                        canm_loc[order[s0 + e]] = True
                if k > Mr:                              # more mines than are left: fits the numbers, not the counter
                    continue
                ccnt[cc, k] += w; cfeas[cc, k] = True
                for e in range(nb):
                    bb = order[s0 + e]; me = mcur[e]
                    wmine[bb, k] += w * me
                    if me > 0:
                        canm[bb, k] = True
                    if me < bsize[bb]:
                        cans[bb, k] = True
                continue
            d += 1
            b2 = order[s0 + d]
            lo, hi = _box_range(b2, bsize[b2], bsig, bnsig[b2], cval, cur, rem)
            mcur[d] = lo - 1; hid[d] = hi; asg[d] = False
        total_nodes += nodes
        if capped[cc]:
            any_capped = True
            for e in range(nb):                         # undo the half-finished search
                if asg[e]:
                    bb = order[s0 + e]
                    for q in range(bnsig[bb]):
                        c = bsig[bb, q]; cur[c] -= mcur[e]; rem[c] += bsize[bb]
                    asg[e] = False
            ncell = 0
            for e in range(nb):
                ncell += bsize[order[s0 + e]]
            for k in range(K):                          # know nothing: every count is possible
                if k <= ncell:
                    cfeas[cc, k] = True
                    ccnt[cc, k] = math.exp(math.lgamma(ncell + 1) - math.lgamma(k + 1) - math.lgamma(ncell - k + 1))
                for e in range(nb):
                    bb = order[s0 + e]
                    canm[bb, k] = True; cans[bb, k] = True; canm_loc[bb] = True
                    wmine[bb, k] = ccnt[cc, k] * k * bsize[bb] / max(ncell, 1)
        else:
            sm = 0.0
            for k in range(K):
                sm += ccnt[cc, k]
            narr += sm
    res[3] = 1 if any_capped else 0
    res[4] = total_nodes
    res[6] = np.int64(min(narr, 9.0e18))
    # 6. combine the pieces with the cells nobody touches (U cells, Mr - k mines among them)
    pre_b = np.zeros((ncomp + 1, K), np.bool_)
    suf_b = np.zeros((ncomp + 2, K), np.bool_)
    pre_b[0, 0] = True
    suf_b[ncomp, 0] = True
    for cc in range(ncomp):
        for a in range(K):
            if pre_b[cc, a]:
                for k in range(K - a):
                    if cfeas[cc, k]:
                        pre_b[cc + 1, a + k] = True
    for cc in range(ncomp - 1, -1, -1):
        for a in range(K):
            if suf_b[cc + 1, a]:
                for k in range(K - a):
                    if cfeas[cc, k]:
                        suf_b[cc, a + k] = True
    nsafe = 0; nmine = 0; glob = 0
    oth_b = np.zeros(K, np.bool_)
    g_b = np.zeros(K, np.bool_)
    for cc in range(ncomp):
        for a in range(K):
            oth_b[a] = False
        for a in range(K):
            if pre_b[cc, a]:
                for c2 in range(K - a):
                    if suf_b[cc + 1, c2]:
                        oth_b[a + c2] = True
        for k in range(K):
            g_b[k] = False
            for so in range(K - k):
                if oth_b[so]:
                    r = Mr - k - so
                    if 0 <= r <= U:
                        g_b[k] = True
                        break
        for e in range(cstart[cc], cstart[cc + 1]):
            bb = order[e]
            cm = False; cs = False; lm = canm_loc[bb]
            for k in range(K):
                if canm[bb, k] and g_b[k]:
                    cm = True
                if cans[bb, k] and g_b[k]:
                    cs = True
            if not cm:
                if lm:
                    glob = 1
                for a in range(nf):
                    i = front[a]
                    if box_of[i] == bb:
                        safe_out[nsafe] = i; nsafe += 1
            elif not cs:
                for a in range(nf):
                    i = front[a]
                    if box_of[i] == bb:
                        mine_out[nmine] = i; nmine += 1
    if U > 0:
        int_m = False; int_s = False
        for s in range(K):
            if pre_b[ncomp, s]:
                r = Mr - s
                if 1 <= r <= U:
                    int_m = True
                if 0 <= r <= U - 1:
                    int_s = True
        if not int_m or not int_s:
            for i in range(N):
                if st[i] == 0 and box_of[i] < 0:
                    if not int_m:
                        safe_out[nsafe] = i; nsafe += 1
                    else:
                        mine_out[nmine] = i; nmine += 1
            if not int_m:
                glob = 1
    res[0] = nsafe; res[1] = nmine; res[2] = glob
    if nsafe > 0:
        return
    # 7. no safe cell: exact mine probabilities (floating point, scaled to avoid overflow)
    logC = np.full(K, -np.inf)
    for r in range(min(U, Mr) + 1):
        logC[r] = math.lgamma(U + 1) - math.lgamma(r + 1) - math.lgamma(U - r + 1)
    pre_f = np.zeros((ncomp + 1, K))
    suf_f = np.zeros((ncomp + 2, K))
    pre_f[0, 0] = 1.0
    suf_f[ncomp, 0] = 1.0
    nrm = np.empty(ncomp + 1)
    for cc in range(ncomp):
        mx = 0.0
        for k in range(K):
            if ccnt[cc, k] > mx:
                mx = ccnt[cc, k]
        nrm[cc] = mx if mx > 0 else 1.0
    for cc in range(ncomp):
        mx = 0.0
        for a in range(K):
            if pre_f[cc, a] > 0:
                for k in range(K - a):
                    if ccnt[cc, k] > 0:
                        pre_f[cc + 1, a + k] += pre_f[cc, a] * ccnt[cc, k] / nrm[cc]
        for a in range(K):
            if pre_f[cc + 1, a] > mx:
                mx = pre_f[cc + 1, a]
        if mx > 0:
            for a in range(K):
                pre_f[cc + 1, a] /= mx
    for cc in range(ncomp - 1, -1, -1):
        mx = 0.0
        for a in range(K):
            if suf_f[cc + 1, a] > 0:
                for k in range(K - a):
                    if ccnt[cc, k] > 0:
                        suf_f[cc, a + k] += suf_f[cc + 1, a] * ccnt[cc, k] / nrm[cc]
        for a in range(K):
            if suf_f[cc, a] > mx:
                mx = suf_f[cc, a]
        if mx > 0:
            for a in range(K):
                suf_f[cc, a] /= mx
    oth_f = np.zeros(K)
    fk = np.zeros(K)
    lw = np.empty(K * K)
    for cc in range(ncomp):
        for a in range(K):
            oth_f[a] = 0.0
        for a in range(K):
            if pre_f[cc, a] > 0:
                for c2 in range(K - a):
                    if suf_f[cc + 1, c2] > 0:
                        oth_f[a + c2] += pre_f[cc, a] * suf_f[cc + 1, c2]
        # fk[k] = sum over the other pieces' mines so of oth_f[so] * C(U, Mr - k - so), on a common scale
        L = -np.inf
        for k in range(K):
            for so in range(K - k):
                r = Mr - k - so
                v = -np.inf
                if oth_f[so] > 0 and 0 <= r <= U:
                    v = math.log(oth_f[so]) + logC[r]
                lw[k * K + so] = v
                if v > L:
                    L = v
        for k in range(K):
            acc = 0.0
            for so in range(K - k):
                v = lw[k * K + so]
                if v > -np.inf:
                    acc += math.exp(v - L)
            fk[k] = acc
        den = 0.0
        for k in range(K):
            den += ccnt[cc, k] * fk[k]
        for e in range(cstart[cc], cstart[cc + 1]):
            bb = order[e]
            nm = 0.0
            for k in range(K):
                nm += wmine[bb, k] * fk[k]
            p = nm / (den * bsize[bb]) if den > 0 else 0.5
            for a in range(nf):
                i = front[a]
                if box_of[i] == bb:
                    prob[i] = p
    if U > 0:
        L = -np.inf
        for s in range(K):
            r = Mr - s
            if pre_f[ncomp, s] > 0 and 0 <= r <= U:
                v = math.log(pre_f[ncomp, s]) + logC[r]
                if v > L:
                    L = v
        T = 0.0; E = 0.0
        for s in range(K):
            r = Mr - s
            if pre_f[ncomp, s] > 0 and 0 <= r <= U:
                w = math.exp(math.log(pre_f[ncomp, s]) + logC[r] - L)
                T += w; E += w * r
        pint = E / (T * U)
        for i in range(N):
            if st[i] == 0 and box_of[i] < 0:
                prob[i] = pint


# ------------------------------------------------------------------ one game
@njit(cache=True)
def choose_guess(strategy, st, NB, NBC, prob, rs, cand):
    """Picks the cell to click in a forced guess. Returns (cell, best probability on the board, tied cells)."""
    N = st.shape[0]
    pmin = 2.0
    for i in range(N):
        if st[i] == 0 and prob[i] < pmin:
            pmin = prob[i]
    n = 0
    if strategy == S_RANDOM:
        for i in range(N):
            if st[i] == 0:
                cand[n] = i; n += 1
    elif strategy == S_FRONTIER:
        for i in range(N):
            if st[i] == 0:
                fr = False
                for k in range(NBC[i]):
                    if st[NB[i, k]] == 1:
                        fr = True
                        break
                if fr:
                    cand[n] = i; n += 1
        if n == 0:
            for i in range(N):
                if st[i] == 0:
                    cand[n] = i; n += 1
    else:
        best_u = 99
        for i in range(N):
            if st[i] == 0 and prob[i] <= pmin + TIE:
                if strategy == S_SMART:
                    u = 0
                    for k in range(NBC[i]):
                        if st[NB[i, k]] == 0:
                            u += 1
                    if u < best_u:
                        best_u = u; n = 0
                    elif u > best_u:
                        continue
                cand[n] = i; n += 1
    ntie = 0
    for i in range(N):
        if st[i] == 0 and prob[i] <= pmin + TIE:
            ntie += 1
    return cand[_randint(rs, n)], pmin, ntie


@njit(cache=True)
def play(M, rule, first, strategy, rs, NB, NBC, BIN, node_cap, mine, num, st, pool, stack, prob, sbuf, mbuf, res, cand,
         out, gp, gpmin, gfrac, gcell, gnunk, gmr, gtie):
    """Plays one game. out[...] gets the per-game summary; the g* arrays get one row per forced guess (up to their length)."""
    N = st.shape[0]
    S = N - M
    gmax = gp.shape[0]
    make_board(M, rule, first, rs, NB, NBC, mine, num, pool)
    for i in range(N):
        st[i] = 0
    for k in range(out.shape[0]):
        out[k] = 0
    out[O_FIRSTFRAC] = -1
    if mine[first] == 1:
        out[O_BOOM1] = 1
        return
    cnt = np.zeros(2, np.int64)
    cnt[0] += reveal(first, st, num, NB, NBC, stack)
    opened_first = cnt[0]
    ng = 0
    while True:
        if cnt[0] == S:
            out[O_WIN] = 1
            break
        basic_pass(M, st, num, NB, NBC, stack, cnt)
        if cnt[0] == S:
            out[O_WIN] = 1
            break
        out[O_STUCK] += 1
        if pair_pass(st, num, NB, NBC, stack, cnt) > 0:
            out[O_PAIR] += 1
            if ng == 0:
                out[O_BASIC_HIDDEN1] = 1
            continue
        full_solve(M, st, num, NB, NBC, BIN, node_cap, sbuf, mbuf, prob, res)
        if res[3]:
            out[O_CAPPED] = 1
        for q in range(res[1]):
            st[mbuf[q]] = 2; cnt[1] += 1
        if res[0] > 0:
            out[O_HIDDEN] += 1
            if ng == 0:
                out[O_BASIC_HIDDEN1] = 1
            if res[2]:
                out[O_GLOBAL] += 1
            for q in range(res[0]):
                cnt[0] += reveal(sbuf[q], st, num, NB, NBC, stack)
            continue
        g, pmin, ntie = choose_guess(strategy, st, NB, NBC, prob, rs, cand)
        if ng == 0:
            out[O_FIRSTFRAC] = int(10000 * cnt[0] / S)
            if cnt[0] == opened_first and cnt[1] == 0:
                out[O_IMMEDIATE] = 1
        if ng < gmax:
            nunk = 0
            for i in range(N):
                if st[i] == 0:
                    nunk += 1
            gp[ng] = prob[g]; gpmin[ng] = pmin; gfrac[ng] = cnt[0] / S; gcell[ng] = g
            gnunk[ng] = nunk; gmr[ng] = M - cnt[1]; gtie[ng] = ntie
        ng += 1
        if mine[g] == 1:
            break
        cnt[0] += reveal(g, st, num, NB, NBC, stack)
    out[O_GUESS] = ng
    out[O_OPENED] = cnt[0]


@njit(parallel=True, cache=True)
def run_batch(W, H, M, rule, first, strategy, seed, g0, ngames, nthreads, node_cap, NB, NBC, BIN,
              OUT, GP, GPMIN, GFRAC, GCELL, GNUNK, GMR, GTIE):
    N = W * H
    chunk = (ngames + nthreads - 1) // nthreads
    for t in prange(nthreads):
        mine = np.zeros(N, np.int8); num = np.zeros(N, np.int8); st = np.zeros(N, np.int8)
        pool = np.empty(N, np.int32); stack = np.empty(N + 8, np.int32); prob = np.zeros(N)
        sbuf = np.empty(N, np.int32); mbuf = np.empty(N, np.int32); res = np.zeros(8, np.int64)
        cand = np.empty(N, np.int32); rs = np.zeros(1, np.uint64)
        for g in range(t * chunk, min(ngames, (t + 1) * chunk)):
            seed_game(rs, seed, g0 + g)
            play(M, rule, first, strategy, rs, NB, NBC, BIN, node_cap, mine, num, st, pool, stack, prob, sbuf, mbuf, res,
                 cand, OUT[g], GP[g], GPMIN[g], GFRAC[g], GCELL[g], GNUNK[g], GMR[g], GTIE[g])
