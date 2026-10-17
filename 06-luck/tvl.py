"""Talent vs Luck: the agent-based model of Pluchino, Biondo & Rapisarda (2018), plus one-rule-at-a-time variants.

The original model (Advances in Complex Systems 21, 1850014; arXiv:1802.07068):
  - N = 1000 people at fixed random positions in a 201 x 201 world that wraps around (a torus)
  - talent ~ Normal(0.6, 0.1), truncated to [0, 1], fixed for life
  - everyone starts with 10 units of capital
  - 500 event points, half lucky and half unlucky, each moving 2 patches in a random direction every step
  - an event point within 1 patch of a person is an event for that person:
      lucky   -> capital doubles with probability equal to the person's talent
      unlucky -> capital is cut in half
  - 80 steps of six months: a 40-year career

Everything is seeded: the same seed gives the same world.
"""
import math

import numpy as np
from numba import njit, prange

NPAR = 16
L_, NE_, STEPS_, PL_, MOVE_, RAD_, UP_, DOWN_, INDEP_, OPP_, PAY_, C0_, MT_, ST_, START_, SHIELD_ = range(NPAR)

TB = 50          # talent bins of 0.02
NLK = 31         # net luck (lucky - unlucky events), clipped to -15..15
LKO = 15
CB = 801         # log2(capital / start) in bins of 0.05, from -20 to +20
NS = 40          # scalar accumulators


def params(**kw):
    """Parameter vector. Defaults are the original model."""
    P = np.zeros(NPAR)
    P[L_] = 201.0        # world size (patches)
    P[NE_] = 500         # event points
    P[STEPS_] = 80       # six-month steps
    P[PL_] = 0.5         # share of lucky event points
    P[MOVE_] = 2.0       # distance an event point moves per step
    P[RAD_] = 1.0        # an event point this close to a person is an event
    P[UP_] = 2.0         # a lucky event multiplies capital by this (if the person's talent lets them use it)
    P[DOWN_] = 2.0       # an unlucky event divides capital by this
    P[INDEP_] = 0.0      # > 0: no map; every step each person gets each kind of event with this probability
    P[OPP_] = 0.0        # 1: talent attracts opportunities (lucky events arrive in proportion to talent)
    P[PAY_] = 0.0        # paycheck: this much capital times talent is added every step
    P[C0_] = 10.0        # starting capital
    P[MT_] = 0.6         # mean talent
    P[ST_] = 0.1         # standard deviation of talent
    P[START_] = 0.0      # > 0: unequal start (lognormal with this sigma, same mean)
    P[SHIELD_] = 0.0     # 1: talent softens bad luck (an unlucky event divides by DOWN ** (1 - talent))
    names = dict(L=L_, NE=NE_, steps=STEPS_, pL=PL_, move=MOVE_, radius=RAD_, up=UP_, down=DOWN_, indep=INDEP_, opp=OPP_,
                 pay=PAY_, C0=C0_, mT=MT_, sT=ST_, start=START_, shield=SHIELD_)
    for k, v in kw.items():
        P[names[k]] = v
    return P


@njit(cache=True)
def _seed(s):
    np.random.seed(s)


@njit(cache=True)
def population(N, P, tal, ax, ay):
    """Draw the people: talent (truncated normal) and a fixed position."""
    mT = P[MT_]
    sT = P[ST_]
    L = P[L_]
    for i in range(N):
        t = np.random.normal(mT, sT)
        while t < 0.0 or t > 1.0:
            t = np.random.normal(mT, sT)
        tal[i] = t
        ax[i] = np.random.random() * L
        ay[i] = np.random.random() * L


@njit(cache=True)
def simulate(N, P, tal, ax, ay, cap, nl, nu, used, rec, h_cap, h_ev, h_ex, h_ey):
    """One career for everyone. Fills cap (final capital), nl / nu (lucky / unlucky events received) and
    used (lucky events turned into a gain). With rec=True also records the history:
    h_cap[step, person], h_ev[step, person] (bit 1 lucky received, 2 lucky used, 4 unlucky), h_ex / h_ey[step, event]."""
    L = P[L_]
    NE = int(P[NE_])
    steps = int(P[STEPS_])
    nlucky = int(round(NE * P[PL_]))
    move = P[MOVE_]
    rad = P[RAD_]
    up = P[UP_]
    down = P[DOWN_]
    q = P[INDEP_]
    opp = P[OPP_] > 0
    pay = P[PAY_]
    mT = P[MT_]
    ssd = P[START_]
    shield = P[SHIELD_] > 0
    for i in range(N):
        if ssd > 0:
            cap[i] = P[C0_] * math.exp(np.random.normal(0.0, ssd) - 0.5 * ssd * ssd)
        else:
            cap[i] = P[C0_]
        nl[i] = 0
        nu[i] = 0
        used[i] = 0
    if rec:
        for i in range(N):
            h_cap[0, i] = cap[i]
            h_ev[0, i] = 0

    if q > 0:                                   # independent luck: no map
        for s in range(steps):
            for i in range(N):
                code = 0
                ql = q * tal[i] / mT if opp else q
                if np.random.random() < ql:
                    nl[i] += 1
                    code |= 1
                    if np.random.random() < tal[i]:
                        cap[i] *= up
                        used[i] += 1
                        code |= 2
                if np.random.random() < q:
                    nu[i] += 1
                    cap[i] /= (down ** (1.0 - tal[i])) if shield else down
                    code |= 4
                if pay > 0:
                    cap[i] += pay * tal[i]
                if rec:
                    h_ev[s + 1, i] = code
                    h_cap[s + 1, i] = cap[i]
        return

    rmax = rad * (math.sqrt(1.0 / mT) if opp else 1.0)
    nc = int(L / rmax)
    cs = L / nc
    head = np.full(nc * nc, -1, np.int32)
    nxt = np.empty(N, np.int32)
    for i in range(N):
        cx = int(ax[i] / cs)
        cy = int(ay[i] / cs)
        if cx >= nc:
            cx = nc - 1
        if cy >= nc:
            cy = nc - 1
        cidx = cx * nc + cy
        nxt[i] = head[cidx]
        head[cidx] = i
    ex = np.random.random(NE) * L
    ey = np.random.random(NE) * L
    hitl = np.zeros(N, np.uint8)
    hitu = np.zeros(N, np.uint8)
    touched = np.empty(N, np.int32)
    r2 = rad * rad
    half = L / 2
    if rec:
        for e in range(NE):
            h_ex[0, e] = ex[e]
            h_ey[0, e] = ey[e]
    for s in range(steps):
        nt = 0
        for e in range(NE):
            ang = np.random.random() * 2.0 * math.pi
            ex[e] = (ex[e] + move * math.cos(ang)) % L
            ey[e] = (ey[e] + move * math.sin(ang)) % L
            cx = int(ex[e] / cs)
            cy = int(ey[e] / cs)
            if cx >= nc:
                cx = nc - 1
            if cy >= nc:
                cy = nc - 1
            lucky = e < nlucky
            for dx in range(-1, 2):
                gx = (cx + dx) % nc
                for dy in range(-1, 2):
                    gy = (cy + dy) % nc
                    i = head[gx * nc + gy]
                    while i >= 0:
                        ddx = abs(ax[i] - ex[e])
                        if ddx > half:
                            ddx = L - ddx
                        ddy = abs(ay[i] - ey[e])
                        if ddy > half:
                            ddy = L - ddy
                        d2 = ddx * ddx + ddy * ddy
                        lim = r2
                        if lucky and opp:
                            lim = r2 * tal[i] / mT
                        if d2 <= lim:
                            if hitl[i] == 0 and hitu[i] == 0:
                                touched[nt] = i
                                nt += 1
                            if lucky:
                                hitl[i] = 1
                            else:
                                hitu[i] = 1
                        i = nxt[i]
        if rec:
            for i in range(N):
                h_ev[s + 1, i] = 0
        for j in range(nt):
            i = touched[j]
            code = 0
            if hitl[i]:
                nl[i] += 1
                code |= 1
                if np.random.random() < tal[i]:
                    cap[i] *= up
                    used[i] += 1
                    code |= 2
            if hitu[i]:
                nu[i] += 1
                cap[i] /= (down ** (1.0 - tal[i])) if shield else down
                code |= 4
            hitl[i] = 0
            hitu[i] = 0
            if rec:
                h_ev[s + 1, i] = code
        if pay > 0:
            for i in range(N):
                cap[i] += pay * tal[i]
        if rec:
            for i in range(N):
                h_cap[s + 1, i] = cap[i]
            for e in range(NE):
                h_ex[s + 1, e] = ex[e]
                h_ey[s + 1, e] = ey[e]


@njit(cache=True)
def _cbin(c, c0):
    v = math.log2(c / c0)
    b = int(round((v + 20.0) / 0.05))
    if b < 0:
        b = 0
    if b >= CB:
        b = CB - 1
    return b


@njit(parallel=True, cache=True)
def batch(seed, nchunk, per, N, P):
    """nchunk * per independent worlds of N people. Returns histograms and sums (see run_all.py for what each one is).
    'Rich' = the richest 1% of each world (ties broken at random)."""
    H_tl = np.zeros((nchunk, TB, NLK, 2), np.int64)       # talent bin x net luck x rich
    H_dl = np.zeros((nchunk, 10, NLK, 2), np.int64)       # talent decile (rank in the world) x net luck x rich
    H_lu = np.zeros((nchunk, 25, 25, 2), np.int64)        # lucky events x unlucky events x rich
    H_c = np.zeros((nchunk, CB), np.int64)                # final capital
    H_grp = np.zeros((nchunk, 5, CB), np.int64)           # final capital by talent group
    H_best = np.zeros((nchunk, TB), np.int64)             # talent of the richest person of each world
    H_rtp = np.zeros((nchunk, 100), np.int64)             # talent percentile of the rich
    H_rlp = np.zeros((nchunk, 100), np.int64)             # luck percentile of the rich
    H_rtl = np.zeros((nchunk, 10, 10), np.int64)          # talent decile x luck decile of the rich
    S = np.zeros((nchunk, NS))
    nrich = N // 100
    c0 = P[C0_]
    for ch in prange(nchunk):
        tal = np.empty(N)
        ax = np.empty(N)
        ay = np.empty(N)
        cap = np.empty(N)
        nl = np.empty(N, np.int32)
        nu = np.empty(N, np.int32)
        used = np.empty(N, np.int32)
        d2 = np.empty((1, 1))
        d8 = np.empty((1, 1), np.uint8)
        trank = np.empty(N, np.int32)
        rich = np.empty(N, np.uint8)
        wl = np.empty(NLK, np.int64)
        for r in range(per):
            np.random.seed(seed + ch * per + r)
            population(N, P, tal, ax, ay)
            simulate(N, P, tal, ax, ay, cap, nl, nu, used, False, d2, d8, d2, d2)
            key = cap * (1.0 + 1e-7 * np.random.random(N))
            order = np.argsort(key)
            torder = np.argsort(tal)
            for j in range(N):
                trank[torder[j]] = j
                rich[j] = 0
            for j in range(N - nrich, N):
                rich[order[j]] = 1
            for k in range(NLK):
                wl[k] = 0
            total = 0.0
            for i in range(N):
                n = nl[i] - nu[i]
                if n < -LKO:
                    n = -LKO
                if n > LKO:
                    n = LKO
                wl[n + LKO] += 1
                total += cap[i]
            # shares and Gini
            acc = 0.0
            gsum = 0.0
            s1 = 0.0
            s10 = 0.0
            s20 = 0.0
            sb50 = 0.0
            for j in range(N):
                cj = cap[order[j]]
                gsum += (j + 1) * cj
                if j < N // 2:
                    sb50 += cj
                if j >= N - nrich:
                    s1 += cj
                if j >= N - N // 10:
                    s10 += cj
                if j >= N - N // 5:
                    s20 += cj
            S[ch, 0] += 1
            S[ch, 1] += s1 / total
            S[ch, 2] += s10 / total
            S[ch, 3] += s20 / total
            S[ch, 4] += 2.0 * gsum / (N * total) - (N + 1.0) / N
            S[ch, 28] += sb50 / total
            S[ch, 29] += N
            S[ch, 30] += total
            ibest = order[N - 1]
            imt = torder[N - 1]
            tb = tal[ibest]
            S[ch, 20] += tb
            S[ch, 21] += tb * tb
            if tb > 0.8:
                S[ch, 19] += 1
            H_best[ch, min(TB - 1, int(tb / 0.02))] += 1
            S[ch, 10] += cap[imt]
            S[ch, 22] += math.log2(cap[order[N - nrich]] / c0)
            if rich[imt]:
                S[ch, 23] += 1
            if imt == ibest:
                S[ch, 24] += 1
            if cap[ibest] > S[ch, 25]:
                S[ch, 25] = cap[ibest]
            S[ch, 26] += cap[ibest]
            cum = 0
            for i in range(N):
                t = tal[i]
                ci = cap[i]
                n = nl[i] - nu[i]
                if n < -LKO:
                    n = -LKO
                if n > LKO:
                    n = LKO
                ri = rich[i]
                tbin = min(TB - 1, int(t / 0.02))
                dec = trank[i] * 10 // N
                H_tl[ch, tbin, n + LKO, ri] += 1
                H_dl[ch, dec, n + LKO, ri] += 1
                H_lu[ch, min(24, nl[i]), min(24, nu[i]), ri] += 1
                b = _cbin(ci, c0)
                H_c[ch, b] += 1
                tr = trank[i]
                if tr < N // 100:
                    H_grp[ch, 0, b] += 1
                if tr < N // 10:
                    H_grp[ch, 1, b] += 1
                if N * 45 // 100 <= tr < N * 55 // 100:
                    H_grp[ch, 2, b] += 1
                if tr >= N - N // 10:
                    H_grp[ch, 3, b] += 1
                if tr >= N - N // 100:
                    H_grp[ch, 4, b] += 1
                if ci < c0:
                    S[ch, 5] += 1
                elif ci == c0:
                    S[ch, 6] += 1
                else:
                    S[ch, 7] += 1
                if t > 0.7:
                    S[ch, 8] += 1
                    if ci > c0:
                        S[ch, 9] += 1
                if abs(t - P[MT_]) < 0.01:
                    S[ch, 11] += ci
                    S[ch, 12] += 1
                S[ch, 13] += nl[i]
                S[ch, 14] += nu[i]
                S[ch, 35] += used[i]
                if nl[i] > S[ch, 15]:
                    S[ch, 15] = nl[i]
                if nu[i] > S[ch, 16]:
                    S[ch, 16] = nu[i]
                if nl[i] == 0 and nu[i] == 0:
                    S[ch, 17] += 1
                elif nl[i] == 0 or nu[i] == 0:
                    S[ch, 18] += 1
                if ri:
                    S[ch, 36] += 1
                    S[ch, 27] += t
                    S[ch, 33] += nl[i]
                    S[ch, 34] += nu[i]
                    if tr < N // 2:
                        S[ch, 31] += 1
                    if nl[i] - nu[i] <= 0:
                        S[ch, 32] += 1
                    less = 0
                    for k in range(n + LKO):
                        less += wl[k]
                    lp = (less + 0.5 * wl[n + LKO]) / N
                    lb = min(99, int(lp * 100))
                    H_rtp[ch, tr * 100 // N] += 1
                    H_rlp[ch, lb] += 1
                    H_rtl[ch, dec, min(9, int(lp * 10))] += 1
    return H_tl, H_dl, H_lu, H_c, H_grp, H_best, H_rtp, H_rlp, H_rtl, S


@njit(parallel=True, cache=True)
def replay_pairs(seed, nchunk, per, N, P):
    """The same people live two careers: same talents and positions, new events. For the richest 1% of the first
    life: how many are in the richest 1% of the second, and where they land (percentile of the second life)."""
    H_ov = np.zeros((nchunk, N // 100 + 1), np.int64)     # how many of the first life's rich are rich again
    H_pos = np.zeros((nchunk, 100), np.int64)             # percentile in life B of the people who were rich in life A
    H_dec = np.zeros((nchunk, 10, 2), np.int64)           # talent decile of the rich in A x rich again in B
    S = np.zeros((nchunk, 8))
    nrich = N // 100
    c0 = P[C0_]
    for ch in prange(nchunk):
        tal = np.empty(N)
        ax = np.empty(N)
        ay = np.empty(N)
        capa = np.empty(N)
        capb = np.empty(N)
        nl = np.empty(N, np.int32)
        nu = np.empty(N, np.int32)
        used = np.empty(N, np.int32)
        d2 = np.empty((1, 1))
        d8 = np.empty((1, 1), np.uint8)
        rankb = np.empty(N, np.int32)
        trank = np.empty(N, np.int32)
        for r in range(per):
            w = ch * per + r
            np.random.seed(seed + w)
            population(N, P, tal, ax, ay)
            torder = np.argsort(tal)
            for j in range(N):
                trank[torder[j]] = j
            np.random.seed(seed + 1_000_000_000 + 2 * w)
            simulate(N, P, tal, ax, ay, capa, nl, nu, used, False, d2, d8, d2, d2)
            oa = np.argsort(capa * (1.0 + 1e-7 * np.random.random(N)))
            np.random.seed(seed + 1_000_000_000 + 2 * w + 1)
            simulate(N, P, tal, ax, ay, capb, nl, nu, used, False, d2, d8, d2, d2)
            ob = np.argsort(capb * (1.0 + 1e-7 * np.random.random(N)))
            for j in range(N):
                rankb[ob[j]] = j
            ov = 0
            for j in range(N - nrich, N):
                i = oa[j]
                again = rankb[i] >= N - nrich
                if again:
                    ov += 1
                H_pos[ch, rankb[i] * 100 // N] += 1
                H_dec[ch, trank[i] * 10 // N, 1 if again else 0] += 1
                S[ch, 1] += 1
                if capb[i] < c0:
                    S[ch, 2] += 1
                if capb[i] <= c0:
                    S[ch, 3] += 1
            H_ov[ch, ov] += 1
            S[ch, 0] += 1
    return H_ov, H_pos, H_dec, S


def history(seed, life_seed=None, N=1000, P=None):
    """One world with everything recorded (for the pictures and the worked example)."""
    P = params() if P is None else P
    steps = int(P[STEPS_])
    NE = int(P[NE_])
    tal = np.empty(N)
    ax = np.empty(N)
    ay = np.empty(N)
    cap = np.empty(N)
    nl = np.empty(N, np.int32)
    nu = np.empty(N, np.int32)
    used = np.empty(N, np.int32)
    h_cap = np.empty((steps + 1, N))
    h_ev = np.zeros((steps + 1, N), np.uint8)
    h_ex = np.empty((steps + 1, NE))
    h_ey = np.empty((steps + 1, NE))
    _seeded_history(seed, -1 if life_seed is None else life_seed, N, P, tal, ax, ay, cap, nl, nu, used, h_cap, h_ev, h_ex, h_ey)
    return dict(tal=tal, ax=ax, ay=ay, cap=cap, nl=nl, nu=nu, used=used, h_cap=h_cap, h_ev=h_ev, h_ex=h_ex, h_ey=h_ey)


@njit(cache=True)
def _seeded_history(seed, life_seed, N, P, tal, ax, ay, cap, nl, nu, used, h_cap, h_ev, h_ex, h_ey):
    np.random.seed(seed)
    population(N, P, tal, ax, ay)
    if life_seed >= 0:
        np.random.seed(life_seed)
    simulate(N, P, tal, ax, ay, cap, nl, nu, used, True, h_cap, h_ev, h_ex, h_ey)


@njit(cache=True)
def world(seed, N, P, tal, ax, ay, cap, nl, nu, used, rich):
    """World number `seed` exactly as batch() plays it, including the random tie-break for the richest 1%."""
    d2 = np.empty((1, 1))
    d8 = np.empty((1, 1), np.uint8)
    np.random.seed(seed)
    population(N, P, tal, ax, ay)
    simulate(N, P, tal, ax, ay, cap, nl, nu, used, False, d2, d8, d2, d2)
    order = np.argsort(cap * (1.0 + 1e-7 * np.random.random(N)))
    for i in range(N):
        rich[i] = 0
    for j in range(N - N // 100, N):
        rich[order[j]] = 1


@njit(cache=True)
def lives(seed, first_life_seed, nlives, N, P, rich_count, caps, richk, keep):
    """One population, many careers. rich_count[i] = lives in which person i ended in the richest 1%.
    caps[k, i] = final capital and richk[k, :] = the richest 1% in life k, for the first `keep` lives. Returns the talents."""
    tal = np.empty(N)
    ax = np.empty(N)
    ay = np.empty(N)
    cap = np.empty(N)
    nl = np.empty(N, np.int32)
    nu = np.empty(N, np.int32)
    used = np.empty(N, np.int32)
    d2 = np.empty((1, 1))
    d8 = np.empty((1, 1), np.uint8)
    np.random.seed(seed)
    population(N, P, tal, ax, ay)
    nrich = N // 100
    for i in range(N):
        rich_count[i] = 0
    for k in range(nlives):
        np.random.seed(first_life_seed + k)
        simulate(N, P, tal, ax, ay, cap, nl, nu, used, False, d2, d8, d2, d2)
        o = np.argsort(cap * (1.0 + 1e-7 * np.random.random(N)))
        for j in range(N - nrich, N):
            rich_count[o[j]] += 1
        if k < keep:
            for i in range(N):
                caps[k, i] = cap[i]
            for j in range(nrich):
                richk[k, j] = o[N - 1 - j]
    return tal
