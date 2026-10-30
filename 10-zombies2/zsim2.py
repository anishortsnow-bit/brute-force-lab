"""Zombie outbreak, part 2: the Cornell model of the first video (../08-zombies/zsim.py; Alemi, Bierbaum, Myers & Sethna,
"You Can Run, You Can Hide", arXiv:1503.01104, 2015) with the things the viewers asked for.

The paper has no police, no army, no differences between people, no learning and no terrain. Everything below that goes
beyond it is our own extrapolation. With one class of people (bite rate beta, kill rate kappa = 0.8 beta), flat terrain
and no soldiers this is exactly the model of the first video (checked in run2.py, section A).

Per cell: S[c, k] humans of class k, E exposed (bitten, turning), Z zombies, R destroyed zombies.
Each class k has its own bite rate b[k] and kill rate kk[k] per human-zombie pair:
    S_k + Z -> E + Z           at b[k]  * S_k * Z
    S_k + Z -> S_up[k] + R     at kk[k] * S_k * Z     (the human who destroyed the zombie may move up a class: learning)
    E -> Z                     at nu * E
Zombies step to each land neighbour d at rate mu[c, d] (terrain; flat = the paper's mu). People never move between
cells (as in the paper), except the evacuation of the ring cells when the army arrives.

The army ring: at time ring_T every land cell already reached by a zombie, plus a margin of ring_m cells, is the
quarantine zone; the next ring_w cells around it (on the land graph, so the ring follows the coast) get n_sold soldiers
in total (class ks), and their civilians are moved one cell outward. Soldiers stay where they are, unless rebal_h > 0: then every rebal_h
hours after the ring forms, the soldiers still standing are pooled and spread evenly along the ring again (a perfectly
organized army that can reach any part of the ring in time; review r2, 29/10). A zombie stepping
out of a ring cell to the outside has to cross the soldiers' square first: it gets through with chance
exp(-sum_k kk[k] S[c, k] / q[c]) (q[c] = its total step rate, so 1/q[c] is the time to cross a cell); otherwise it stays.

Inside a cell the reactions run event by event (exact), with tau-leaping only while humans and zombies are both
numerous, as in zsim.py. Zombies step between cells every dt hours (operator splitting, checked against exact_grid2).
Random numbers: xoshiro256** seeded through splitmix64, one stream per run (the block of our werewolf code, video 09; Numba's
np.random seeded with consecutive integers showed a small bias there).
"""
import math

import numba as nb
import numpy as np

BETA = 3.6e-3
ALPHA = 0.8
MU = 0.0914
NU = 2.0
GAMMA = BETA * (1.0 + ALPHA)      # meetings per hour per pair = the paper's bite rate + kill rate
P_MOVIE = ALPHA / (1.0 + ALPHA)   # 4/9: share of fights a human wins in the movies

LEAP_MIN = 200
LEAP_EPS = 0.03


# ------------------------------------------------------------------ random numbers

@nb.njit(inline='always')
def _rotl(x, r):
    return (x << np.uint64(r)) | (x >> np.uint64(64 - r))


@nb.njit
def _next(st):
    out = _rotl(st[1] * np.uint64(5), 7) * np.uint64(9)
    t = st[1] << np.uint64(17)
    st[2] ^= st[0]
    st[3] ^= st[1]
    st[1] ^= st[2]
    st[0] ^= st[3]
    st[2] ^= t
    st[3] = _rotl(st[3], 45)
    return out


@nb.njit
def _seed(seed):
    st = np.zeros(4, dtype=np.uint64)
    z = np.uint64(seed)
    for i in range(4):
        z += np.uint64(0x9E3779B97F4A7C15)
        x = z
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        st[i] = x ^ (x >> np.uint64(31))
    return st


@nb.njit
def _uniform(st):
    return np.float64(_next(st) >> np.uint64(11)) * (1.0 / 9007199254740992.0)


@nb.njit
def _randint(st, m):
    return np.int64((_next(st) >> np.uint64(11)) % np.uint64(m))


@nb.njit
def _normal(st):
    u1 = _uniform(st)
    u2 = _uniform(st)
    return math.sqrt(-2.0 * math.log(1.0 - u1)) * math.cos(2.0 * math.pi * u2)


@nb.njit
def _binom(st, n, p):
    """Binomial draw: exact inversion while the variance is at most 30, normal approximation above (as in zsim.py)."""
    if n <= 0 or p <= 0.0:
        return 0
    if p >= 1.0:
        return n
    m = n * p
    v = m * (1.0 - p)
    if v > 30.0:
        k = int(math.floor(m + math.sqrt(v) * _normal(st) + 0.5))
        return 0 if k < 0 else (n if k > n else k)
    if p > 0.5:
        return n - _binom(st, n, 1.0 - p)
    q = 1.0 - p
    u = _uniform(st)
    pk = math.exp(n * math.log(q))
    f = pk
    k = 0
    r = p / q
    while u > f and k < n:
        pk *= r * (n - k) / (k + 1)
        k += 1
        f += pk
        if pk < 1e-300 and k > m:
            break
    return k


@nb.njit
def _poisson(st, lam):
    if lam <= 0.0:
        return 0
    if lam > 30.0:
        k = int(math.floor(lam + math.sqrt(lam) * _normal(st) + 0.5))
        return 0 if k < 0 else k
    u = _uniform(st)
    pk = math.exp(-lam)
    f = pk
    k = 0
    while u > f:
        k += 1
        pk *= lam / k
        f += pk
        if pk < 1e-300 and k > lam:
            break
    return k


# ------------------------------------------------------------------ one cell

@nb.njit(cache=True)
def _cell2(st, S, c, e, z, r, b, kk, up, nu, dt, rb, rk, nb_, nk_):
    """Reactions inside cell c for dt hours; S[c] is updated in place. rb, rk, nb_, nk_ are scratch arrays of size K.
    Returns (e, z, r, bites, kills)."""
    K = S.shape[1]
    t = 0.0
    bites = 0
    kills = 0
    while True:
        stot = 0
        for k in range(K):
            stot += S[c, k]
        if stot == 0:
            if e > 0:
                x = _binom(st, e, 1.0 - math.exp(-nu * (dt - t)))
                e -= x
                z += x
            break
        zf = float(z)
        a1 = 0.0
        a2 = 0.0
        bmax = 0.0
        for k in range(K):
            sk = S[c, k]
            if sk > 0 and z > 0:
                rb[k] = b[k] * sk * zf
                rk[k] = kk[k] * sk * zf
                a1 += rb[k]
                a2 += rk[k]
                if b[k] > bmax:
                    bmax = b[k]
            else:
                rb[k] = 0.0
                rk[k] = 0.0
        a3 = nu * e
        a0 = a1 + a2 + a3
        if a0 <= 0.0:
            break
        if stot >= LEAP_MIN and z >= LEAP_MIN:
            tau = dt - t
            if bmax > 0.0:
                tau = min(tau, LEAP_EPS / (bmax * zf))
            if a2 + a3 > 0.0:
                tau = min(tau, LEAP_EPS * zf / (a2 + a3))
            if a1 + a3 > 0.0:
                tau = min(tau, LEAP_EPS * max(e, LEAP_MIN) / (a1 + a3))
            zrem = z
            nbt = 0
            nkt = 0
            for k in range(K):
                x = _poisson(st, rb[k] * tau)
                nb_[k] = x if x < S[c, k] else S[c, k]
                y = _poisson(st, rk[k] * tau)
                if y > zrem:
                    y = zrem
                nk_[k] = y
                zrem -= y
                nbt += nb_[k]
                nkt += y
            for k in range(K):
                S[c, k] -= nb_[k]
            for k in range(K):                       # learning: the winners move up (after this leap's bites)
                u = up[k]
                if u != k and nk_[k] > 0:
                    m = nk_[k] if nk_[k] < S[c, k] else S[c, k]
                    S[c, k] -= m
                    S[c, u] += m
            nc = _binom(st, e, 1.0 - math.exp(-nu * tau))
            e += nbt - nc
            z += nc - nkt
            r += nkt
            bites += nbt
            kills += nkt
            t += tau
            if t >= dt - 1e-12:
                break
            continue
        t += -math.log(1.0 - _uniform(st)) / a0
        if t > dt:
            break
        u = _uniform(st) * a0
        if u < a1:
            k = 0
            acc = rb[0]
            while acc < u and k < K - 1:
                k += 1
                acc += rb[k]
            while k > 0 and rb[k] == 0.0:            # float round-off at the edge
                k -= 1
            S[c, k] -= 1
            e += 1
            bites += 1
        elif u < a1 + a2:
            u -= a1
            k = 0
            acc = rk[0]
            while acc < u and k < K - 1:
                k += 1
                acc += rk[k]
            while k > 0 and rk[k] == 0.0:
                k -= 1
            z -= 1
            r += 1
            kills += 1
            if up[k] != k:
                S[c, k] -= 1
                S[c, up[k]] += 1
        else:
            e -= 1
            z += 1
    return e, z, r, bites, kills


# ------------------------------------------------------------------ the army ring

@nb.njit(cache=True)
def _build_ring(S, t_reach, nbr, ring_m, ring_w, n_sold, ks, depth, queue):
    """Marks the quarantine zone and the ring (depth = BFS distance on the land graph from the reached cells),
    evacuates the ring's civilians one cell outward and stations the soldiers. Returns the number of ring cells."""
    n = len(t_reach)
    K = S.shape[1]
    for c in range(n):
        depth[c] = -1
    qh = 0
    qt = 0
    for c in range(n):
        if t_reach[c] < np.inf:
            depth[c] = 0
            queue[qt] = c
            qt += 1
    if qt == 0:
        return 0
    dmax = ring_m + ring_w
    while qh < qt:
        c = queue[qh]
        qh += 1
        if depth[c] >= dmax + 1:
            continue
        for d in range(4):
            j = nbr[c, d]
            if j >= 0 and depth[j] == -1:
                depth[j] = depth[c] + 1
                queue[qt] = j
                qt += 1
    nring = 0
    for c in range(n):
        if ring_m < depth[c] <= dmax:
            nring += 1
    if nring == 0:
        return 0
    per = n_sold // nring
    extra = n_sold - per * nring
    q = 0
    for c in range(n):
        if ring_m < depth[c] <= dmax:
            out = -1                                 # evacuate to the first neighbour outside the ring (if any)
            for d in range(4):
                j = nbr[c, d]
                if j >= 0 and (depth[j] == -1 or depth[j] > dmax):
                    out = j
                    break
            if out >= 0:
                for k in range(K):
                    if k != ks and S[c, k] > 0:
                        S[out, k] += S[c, k]
                        S[c, k] = 0
            S[c, ks] += per + (1 if q < extra else 0)
            q += 1
    return nring


# ------------------------------------------------------------------ the whole country

@nb.njit(cache=True)
def us_run2(st, S, E, Z, R, t_reach, b, kk, up, nu, mu, nbr, dt, t_end, snap_t,
            ring_T, ring_m, ring_w, n_sold, ks, watch, slow=5, rebal_h=0.0):
    """One outbreak from the state given (S, E, Z, R, t_reach are modified in place), from t = 0 to t_end hours.
    mu: (n, 4) step rate to each neighbour (0 = closed). snap_t: hours where humans and zombies per cell are saved.
    ring_T < 0: no army; ks = class of the soldiers (-1 = none). watch: cells whose first-contact time is reported.
    Returns (hourly [humans (soldiers not counted), exposed, zombies, destroyed, bites, kills] + people per class,
             snapshots of humans per cell (soldiers not counted), snapshots of zombies per cell,
             first contact of the watched cells, ring cells, first time a zombie entered a never-reached cell outside
             the ring (hours, inf if none), outbreak alive at the end,
             ring stats [people inside the quarantine zone at ring time, people outside the ring at ring time,
                         people bitten outside the ring after it formed, outbreak alive outside at the end (0/1),
                         zombies + exposed outside at the end])."""
    n = S.shape[0]
    K = S.shape[1]
    act = np.empty(n, np.int32)
    isact = np.zeros(n, np.uint8)
    touched = np.empty(n, np.int32)
    dZ = np.zeros(n, np.int32)
    depth = np.full(n, -1, np.int32)
    queue = np.empty(n, np.int32)
    rb = np.zeros(K)
    rk = np.zeros(K)
    nb_ = np.zeros(K, np.int64)
    nk_ = np.zeros(K, np.int64)
    prev = np.zeros(K, np.int64)
    na = 0
    for c in range(n):
        if E[c] > 0 or Z[c] > 0:
            isact[c] = 1
            act[na] = c
            na += 1
            if t_reach[c] == np.inf:
                t_reach[c] = 0.0
    q = np.zeros(n)
    for c in range(n):
        for d in range(4):
            if nbr[c, d] >= 0:
                q[c] += mu[c, d]
    pl = 1.0 - np.exp(-q * dt)
    ps = 1.0 - np.exp(-q * dt * slow)
    nh = int(t_end) + 1
    tot = np.zeros((nh, 6 + K), np.int64)
    cls = np.zeros(K, np.int64)
    for c in range(n):
        for k in range(K):
            cls[k] += S[c, k]
    eT = np.int64(E.sum())
    zT = np.int64(Z.sum())
    rT = np.int64(R.sum())
    bT = np.int64(0)
    kT = np.int64(0)
    tot[0, 0], tot[0, 1], tot[0, 2], tot[0, 3] = _humans(cls, ks), eT, zT, rT
    for k in range(K):
        tot[0, 6 + k] = cls[k]
    ns = len(snap_t)
    snapS = np.zeros((ns, n), np.int32)
    snapZ = np.zeros((ns, n), np.int32)
    si = 0
    nring = 0
    breach = np.inf
    ring_on = False
    outside = np.zeros(n, np.uint8)
    rstat = np.zeros(5)
    rcells = np.zeros(0, np.int32)
    next_rebal = np.inf
    nsteps = int(round(t_end / dt))
    alive = True
    h = 0
    for step in range(nsteps):
        t1 = (step + 1) * dt
        if ring_T >= 0.0 and ks >= 0 and not ring_on and t1 > ring_T:
            nring = _build_ring(S, t_reach, nbr, ring_m, ring_w, n_sold, ks, depth, queue)
            ring_on = nring > 0
            cls[:] = 0
            for c in range(n):
                for k in range(K):
                    cls[k] += S[c, k]
            if ring_on:
                for c in range(n):
                    s_ = 0
                    for k in range(K):
                        if k != ks:
                            s_ += S[c, k]
                    if depth[c] == -1 or depth[c] > ring_m + ring_w:
                        outside[c] = 1
                        rstat[1] += s_
                    elif depth[c] <= ring_m:
                        rstat[0] += s_
                rcells = np.empty(nring, np.int32)
                i = 0
                for c in range(n):
                    if ring_m < depth[c] <= ring_m + ring_w:
                        rcells[i] = c
                        i += 1
                if rebal_h > 0.0:
                    next_rebal = t1 + rebal_h
        if ring_on and t1 >= next_rebal - 1e-9:
            # the soldiers still standing are pooled and spread evenly along the ring again
            pool = 0
            for i in range(nring):
                pool += S[rcells[i], ks]
            per = pool // nring
            ext = pool - per * nring
            for i in range(nring):
                S[rcells[i], ks] = per + (1 if i < ext else 0)
            next_rebal += rebal_h
        # reactions inside each active cell
        for a in range(na):
            c = act[a]
            if E[c] == 0 and Z[c] == 0:
                continue
            e0, z0, r0 = E[c], Z[c], R[c]
            for k in range(K):
                prev[k] = S[c, k]
            e, z, r, bites, kills = _cell2(st, S, c, e0, z0, r0, b, kk, up, nu, dt, rb, rk, nb_, nk_)
            if outside[c]:
                rstat[2] += bites
            for k in range(K):
                cls[k] += S[c, k] - prev[k]
            eT += e - e0
            zT += z - z0
            rT += r - r0
            bT += bites
            kT += kills
            E[c], Z[c], R[c] = e, z, r
        # zombies step to the neighbouring land cells
        nt = 0
        for a in range(na):
            c = act[a]
            zc = Z[c]
            if zc == 0 or q[c] <= 0.0:
                continue
            empty = E[c] == 0
            if empty:
                for k in range(K):
                    if S[c, k] > 0:
                        empty = False
                        break
            if empty:
                if step % slow != 0:
                    continue
                go = _binom(st, zc, ps[c])
            else:
                go = _binom(st, zc, pl[c])
            if go == 0:
                continue
            dZ[c] -= go
            wleft = q[c]
            for d in range(4):
                j = nbr[c, d]
                if j < 0 or mu[c, d] <= 0.0:
                    continue
                if wleft <= mu[c, d] * (1.0 + 1e-12):
                    g = go
                else:
                    g = _binom(st, go, mu[c, d] / wleft)
                wleft -= mu[c, d]
                go -= g
                if g > 0 and ring_on and ring_m < depth[c] <= ring_m + ring_w and (depth[j] == -1 or depth[j] > ring_m + ring_w):
                    # Out through the ring: the zombie has to cross the soldiers' square, which takes 1/q[c] hours (the
                    # model's own time to leave a cell). It gets through only if nobody there destroys it on the way;
                    # the ones stopped stay in the square and fight under the usual rates. Without this, a step between
                    # well-mixed cells crossed a line of thousands of soldiers in an instant (review r1, 29/10).
                    kc = 0.0
                    for k in range(K):
                        kc += kk[k] * S[c, k]
                    stay = g - _binom(st, g, math.exp(-kc / q[c]))
                    dZ[c] += stay
                    g -= stay
                if g > 0:
                    if dZ[j] == 0 and isact[j] == 0:
                        touched[nt] = j
                        nt += 1
                    dZ[j] += g
                if go == 0:
                    break
        for a in range(na):
            c = act[a]
            if dZ[c] != 0:
                Z[c] += dZ[c]
                dZ[c] = 0
        for qq in range(nt):
            j = touched[qq]
            Z[j] += dZ[j]
            dZ[j] = 0
            if isact[j] == 0:
                isact[j] = 1
                act[na] = j
                na += 1
                if t_reach[j] == np.inf:
                    t_reach[j] = t1
                    if ring_on and breach == np.inf and (depth[j] == -1 or depth[j] > ring_m + ring_w):
                        breach = t1
        # drop cells with nobody left to act
        w = 0
        for a in range(na):
            c = act[a]
            if E[c] > 0 or Z[c] > 0:
                act[w] = c
                w += 1
            else:
                isact[c] = 0
        na = w
        h = int(t1 + 1e-9)
        if abs(t1 - h) < 1e-9 and h < nh:
            tot[h, 0], tot[h, 1], tot[h, 2], tot[h, 3], tot[h, 4], tot[h, 5] = _humans(cls, ks), eT, zT, rT, bT, kT
            for k in range(K):
                tot[h, 6 + k] = cls[k]
        while si < ns and t1 + 1e-9 >= snap_t[si]:
            _snap(S, Z, snapS, snapZ, si, ks)
            si += 1
        if na == 0:
            alive = False
            for hh in range(h + 1, nh):
                tot[hh, 0], tot[hh, 1], tot[hh, 2], tot[hh, 3], tot[hh, 4], tot[hh, 5] = _humans(cls, ks), eT, zT, rT, bT, kT
                for k in range(K):
                    tot[hh, 6 + k] = cls[k]
            while si < ns:
                _snap(S, Z, snapS, snapZ, si, ks)
                si += 1
            break
    if ring_on:
        for c in range(n):
            if outside[c] and (E[c] > 0 or Z[c] > 0):
                rstat[3] = 1.0
                rstat[4] += E[c] + Z[c]
    wt = np.empty(len(watch), np.float32)
    for i in range(len(watch)):
        wt[i] = t_reach[watch[i]]
    return tot, snapS, snapZ, wt, nring, breach, alive, rstat


@nb.njit(cache=True)
def _humans(cls, ks):
    """Humans in the population (civilians and police); the soldiers brought in are counted apart."""
    s = cls.sum()
    if ks >= 0:
        s -= cls[ks]
    return s


@nb.njit(cache=True)
def _snap(S, Z, snapS, snapZ, si, ks):
    n, K = S.shape
    for c in range(n):
        s_ = 0
        for k in range(K):
            if k != ks:
                s_ += S[c, k]
        snapS[si, c] = s_
        snapZ[si, c] = Z[c]


@nb.njit(cache=True)
def _infect_one(st, S, E, c):
    """Patient zero: one person of cell c, drawn in proportion to the classes there."""
    K = S.shape[1]
    tot = 0
    for k in range(K):
        tot += S[c, k]
    u = _randint(st, tot)
    k = 0
    acc = S[c, 0]
    while acc <= u:
        k += 1
        acc += S[c, k]
    S[c, k] -= 1
    E[c] += 1


@nb.njit(cache=True)
def one(S0, seed_cell, b, kk, up, nu, mu, nbr, dt, t_end, seed, snap_t, ring_T, ring_m, ring_w, n_sold, ks, watch,
        rebal_h=0.0):
    """One run with snapshots (for the maps). Returns the outputs of us_run2 plus the final first-contact times."""
    n, K = S0.shape
    st = _seed(seed)
    S = S0.copy()
    E = np.zeros(n, np.int32)
    Z = np.zeros(n, np.int32)
    R = np.zeros(n, np.int32)
    tr = np.full(n, np.inf, np.float32)
    _infect_one(st, S, E, seed_cell)
    tot, sS, sZ, wt, nring, br, al, rs = us_run2(st, S, E, Z, R, tr, b, kk, up, nu, mu, nbr, dt, t_end, snap_t,
                                                 ring_T, ring_m, ring_w, n_sold, ks, watch, 5, rebal_h)
    return tot, sS, sZ, wt, nring, br, al, tr, rs


@nb.njit(parallel=True, cache=True)
def many(S0, seed_cell, b, kk, up, nu, mu, nbr, dt, t_end, seeds, ring_T, ring_m, ring_w, n_sold, ks, watch,
         rebal_h=0.0):
    """Independent runs in parallel, each from S0 with one person of seed_cell bitten at t = 0.
    Run r uses seeds[r], ring_T[r] and n_sold[r]. No snapshots."""
    nr = len(seeds)
    n, K = S0.shape
    nh = int(t_end) + 1
    tots = np.zeros((nr, nh, 6 + K), np.int64)
    wts = np.zeros((nr, len(watch)), np.float32)
    nrings = np.zeros(nr, np.int64)
    breaches = np.zeros(nr)
    alives = np.zeros(nr, np.bool_)
    rstats = np.zeros((nr, 5))
    no_snap = np.zeros(0)
    for r in nb.prange(nr):
        st = _seed(seeds[r])
        S = S0.copy()
        E = np.zeros(n, np.int32)
        Z = np.zeros(n, np.int32)
        R = np.zeros(n, np.int32)
        tr = np.full(n, np.inf, np.float32)
        _infect_one(st, S, E, seed_cell)
        tot, _, _, wt, nring, br, al, rs = us_run2(st, S, E, Z, R, tr, b, kk, up, nu, mu, nbr, dt, t_end, no_snap,
                                                   ring_T[r], ring_m, ring_w, n_sold[r], ks, watch, 5, rebal_h)
        tots[r] = tot
        wts[r] = wt
        nrings[r] = nring
        breaches[r] = br
        alives[r] = al
        rstats[r] = rs
    return tots, wts, nrings, breaches, alives, rstats


# ------------------------------------------------------------------ fully exact version (small grids)

@nb.njit(cache=True)
def _prop(S, E, Z, c, b, kk, nu, q):
    K = S.shape[1]
    s = 0.0
    for k in range(K):
        s += (b[k] + kk[k]) * S[c, k]
    return s * Z[c] + nu * E[c] + q[c] * Z[c]


@nb.njit(cache=True)
def exact_grid2(S0, nbr, seed_cell, b, kk, up, nu, mu, t_end, seed):
    """Fully exact (Gillespie) version of us_run2 for small grids, classes and learning included, no army.
    Returns humans per class per cell at t_end and whether the outbreak is alive."""
    st = _seed(seed)
    n, K = S0.shape
    S = S0.astype(np.int64).copy()
    E = np.zeros(n, np.int64)
    Z = np.zeros(n, np.int64)
    S32 = S0.copy()
    E32 = np.zeros(n, np.int32)
    _infect_one(st, S32, E32, seed_cell)
    for k in range(K):
        S[seed_cell, k] = S32[seed_cell, k]
    E[seed_cell] = 1
    q = np.zeros(n)
    for c in range(n):
        for d in range(4):
            if nbr[c, d] >= 0:
                q[c] += mu[c, d]
    a = np.zeros(n)
    for c in range(n):
        a[c] = _prop(S, E, Z, c, b, kk, nu, q)
    A = a.sum()
    t = 0.0
    since = 0
    while A > 1e-12:
        t += -math.log(1.0 - _uniform(st)) / A
        if t > t_end:
            break
        u = _uniform(st) * A
        c = 0
        acc = a[0]
        while acc < u and c < n - 1:
            c += 1
            acc += a[c]
        v = _uniform(st) * a[c]
        done = False
        for k in range(K):
            x = b[k] * S[c, k] * Z[c]
            if v < x:
                S[c, k] -= 1
                E[c] += 1
                done = True
                break
            v -= x
        if not done:
            for k in range(K):
                y = kk[k] * S[c, k] * Z[c]
                if v < y:
                    Z[c] -= 1
                    if up[k] != k:
                        S[c, k] -= 1
                        S[c, up[k]] += 1
                    done = True
                    break
                v -= y
        if not done:
            if v < nu * E[c]:
                E[c] -= 1
                Z[c] += 1
            else:
                v -= nu * E[c]
                j = -1
                for d in range(4):
                    jj = nbr[c, d]
                    if jj >= 0 and mu[c, d] > 0.0:
                        j = jj
                        x = mu[c, d] * Z[c]
                        if v < x:
                            break
                        v -= x
                Z[c] -= 1
                Z[j] += 1
                oa = a[j]
                a[j] = _prop(S, E, Z, j, b, kk, nu, q)
                A += a[j] - oa
        oa = a[c]
        a[c] = _prop(S, E, Z, c, b, kk, nu, q)
        A += a[c] - oa
        since += 1
        if since > 100000:
            A = a.sum()
            since = 0
    return S, (E.sum() + Z.sum()) > 0


# ------------------------------------------------------------------ classes of people

def ladder(levels=range(-8, 11), step=0.05):
    """Civilians who win a share p of their fights: p = 4/9 + step * L. Same meetings per pair as the paper
    (bite + kill = GAMMA), so L = 0 is exactly the paper's beta and kappa."""
    p = np.array([P_MOVIE + step * L for L in levels])
    assert p.min() > 0 and p.max() < 1
    return list(levels), p, GAMMA * (1.0 - p), GAMMA * p


def armed(m):
    """Police and soldiers with guns: bitten like anyone when a zombie gets close (beta), but they destroy zombies
    m times as fast as an average civilian (kappa = m * 0.8 beta), from a distance too, so they also run into more
    zombies than a civilian does. Raising kappa is the paper's own way of writing a better fighter."""
    return BETA, m * ALPHA * BETA


def close_combat(m):
    """The same fighter without the reach of a gun: the paper's meetings per pair (GAMMA), destroying m times as many
    zombies per bite as a civilian (win share m*0.8 / (1 + m*0.8)), like the civilians of ladder()."""
    a = m * ALPHA
    p = a / (1.0 + a)
    return GAMMA * (1.0 - p), GAMMA * p
