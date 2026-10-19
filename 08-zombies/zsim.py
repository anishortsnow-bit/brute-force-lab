"""Zombie outbreak models from Alemi, Bierbaum, Myers & Sethna, "You Can Run, You Can Hide: The Epidemiology
and Statistical Mechanics of Zombies", Phys. Rev. E 92, 052801 (2015), arXiv:1503.01104.

Three models, as in the paper:
1. wellmixed():  S + Z -> Z + Z (rate beta*S*Z), S + Z -> S + R (rate kappa*S*Z); exact stochastic simulation.
2. lattice():    one person per site on a square lattice, bites and kills only between nearest neighbours.
3. us_runs():    the whole lower 48 on 3 km cells. Each cell is well mixed and adds an exposed state:
                 S + Z -> E + Z (beta*S*Z), E -> Z (nu*E), S + Z -> S + R (kappa*S*Z),
                 and every zombie steps to each land neighbour at rate mu (4 neighbours, rate per neighbour).
                 Inside a cell the reactions are simulated event by event (exact). Steps between cells are
                 applied every dt hours with exact binomial draws (operator splitting); exact_grid() is the fully
                 exact version used to check that the splitting does not change the results.

Paper values (Table I): beta = 3.6e-3 /hr/person, alpha = kappa/beta = 0.8, mu = 0.0914 /hr, latency 2 /hr.
"""
import math

import numba as nb
import numpy as np

BETA = 3.6e-3
ALPHA = 0.8
MU = 0.0914
NU = 2.0


# ---------------------------------------------------------------- 1. well mixed

@nb.njit(cache=True)
def wellmixed(s0, z0, alpha, nruns, seed):
    """Final number of zombies and of humans in each run (time does not matter for the end state)."""
    np.random.seed(seed)
    zf = np.zeros(nruns, np.int64)
    sf = np.zeros(nruns, np.int64)
    pb = 1.0 / (1.0 + alpha)
    for r in range(nruns):
        s, z = s0, z0
        while s > 0 and z > 0:
            if np.random.random() < pb:
                s -= 1
                z += 1
            else:
                z -= 1
        zf[r] = z
        sf[r] = s
    return zf, sf


# ---------------------------------------------------------------- 2. lattice

@nb.njit(cache=True)
def lattice(L, alpha, seed, keep_state):
    """One zombie in the middle of an L x L lattice of humans (no wrap-around).
    Returns (number of people bitten, reached the edge?, final state if keep_state: 0 human, 1 zombie, 2 dead zombie)."""
    np.random.seed(seed)
    st = np.zeros(L * L, np.uint8)
    c = (L // 2) * L + L // 2
    st[c] = 1
    cap = 4 * L * L + 16
    bz = np.empty(cap, np.int32)
    bs = np.empty(cap, np.int32)
    nb_ = 0
    bitten = 1
    edge = False
    pb = 1.0 / (1.0 + alpha)
    # bonds of the first zombie
    for d in range(4):
        x, y = c % L, c // L
        if d == 0:
            x += 1
        elif d == 1:
            x -= 1
        elif d == 2:
            y += 1
        else:
            y -= 1
        if 0 <= x < L and 0 <= y < L:
            bz[nb_] = c
            bs[nb_] = y * L + x
            nb_ += 1
    while nb_ > 0:
        k = np.random.randint(nb_)
        z, s = bz[k], bs[k]
        if st[z] != 1 or st[s] != 0:      # stale bond: drop it, no event
            nb_ -= 1
            bz[k], bs[k] = bz[nb_], bs[nb_]
            continue
        if np.random.random() < pb:      # bite
            st[s] = 1
            bitten += 1
            x0, y0 = s % L, s // L
            if x0 == 0 or y0 == 0 or x0 == L - 1 or y0 == L - 1:
                edge = True
            for d in range(4):
                x, y = x0, y0
                if d == 0:
                    x += 1
                elif d == 1:
                    x -= 1
                elif d == 2:
                    y += 1
                else:
                    y -= 1
                if 0 <= x < L and 0 <= y < L and st[y * L + x] == 0:
                    bz[nb_] = s
                    bs[nb_] = y * L + x
                    nb_ += 1
        else:                            # kill
            st[z] = 2
    if not keep_state:
        st = st[:1]
    return bitten, edge, st


@nb.njit(parallel=True, cache=True)
def lattice_many(L, alpha, n, seed):
    bitten = np.zeros(n, np.int64)
    edge = np.zeros(n, np.bool_)
    for r in nb.prange(n):
        b, e, _ = lattice(L, alpha, seed + r, False)
        bitten[r] = b
        edge[r] = e
    return bitten, edge


# ---------------------------------------------------------------- 3. the United States

@nb.njit(cache=True)
def _binom(n, p):
    """Binomial draw; above a variance of 30 the normal approximation (indistinguishable at that size, much faster)."""
    if n <= 0 or p <= 0.0:
        return 0
    if p >= 1.0:
        return n
    m = n * p
    v = m * (1.0 - p)
    if v > 30.0:
        k = int(math.floor(m + math.sqrt(v) * np.random.standard_normal() + 0.5))
        return 0 if k < 0 else (n if k > n else k)
    return np.random.binomial(n, p)


@nb.njit(cache=True)
def _poisson(lam):
    if lam <= 0.0:
        return 0
    if lam > 30.0:
        k = int(math.floor(lam + math.sqrt(lam) * np.random.standard_normal() + 0.5))
        return 0 if k < 0 else k
    return np.random.poisson(lam)


LEAP_MIN = 200       # tau-leaping only while humans and zombies both number at least this many in the cell
LEAP_EPS = 0.03      # each leap changes humans, zombies and exposed by at most ~3% on average


@nb.njit(cache=True)
def _cell(s, e, z, r, beta, kappa, nu, dt):
    """Reactions inside one cell for dt hours. Event by event (exact), except while humans and zombies are both
    numerous (big city cells), where it leaps ahead in short steps (tau-leaping). Returns (s, e, z, r, bites)."""
    t = 0.0
    bites = 0
    while True:
        if s == 0:
            if e > 0:                    # nobody left to bite: each exposed person turns independently
                k = _binom(e, 1.0 - math.exp(-nu * (dt - t)))
                e -= k
                z += k
            break
        sz = float(s) * float(z)
        a1 = beta * sz
        a2 = kappa * sz
        a3 = nu * e
        a0 = a1 + a2 + a3
        if a0 <= 0.0:
            break
        if s >= LEAP_MIN and z >= LEAP_MIN:
            tau = min(dt - t, LEAP_EPS * s / a1, LEAP_EPS * z / (a2 + a3), LEAP_EPS * max(e, LEAP_MIN) / (a1 + a3))
            nb_ = min(_poisson(a1 * tau), s)
            nk = min(_poisson(a2 * tau), z)
            nc = _binom(e, 1.0 - math.exp(-nu * tau))
            s -= nb_
            e += nb_ - nc
            z += nc - nk
            r += nk
            bites += nb_
            t += tau
            if t >= dt - 1e-12:
                break
            continue
        t += -math.log(1.0 - np.random.random()) / a0
        if t > dt:
            break
        u = np.random.random() * a0
        if u < a1:
            s -= 1
            e += 1
            bites += 1
        elif u < a1 + a2:
            z -= 1
            r += 1
        else:
            e -= 1
            z += 1
    return s, e, z, r, bites


@nb.njit(cache=True)
def us_run(pop, nbr, seeds, beta, kappa, nu, mu, dt, t_end, snap_t, seed, s_mult, block, slow=5):
    """One outbreak on the land cells.
    pop: people per land cell; nbr: (n, 4) land neighbours or -1; seeds: cell index of each initially exposed person.
    snap_t: times (hours) where the number of humans of every cell is saved.
    s_mult: per-cell multiplier on beta and kappa (1.0 = paper); block: per-cell flag, zombies cannot enter.
    slow: zombies in cells with nobody left to bite or turn (no humans, no exposed) step every `slow` steps,
          with the probability for that longer interval (same average flow; checked against exact_grid).
    Returns (hourly totals [S, E, Z, R], snapshots of S, first-contact time per cell, time the cell lost half
    its people, outbreak still alive at the end)."""
    np.random.seed(seed)
    n = len(pop)
    S = pop.astype(np.int32).copy()
    E = np.zeros(n, np.int32)
    Z = np.zeros(n, np.int32)
    Rr = np.zeros(n, np.int32)
    dZ = np.zeros(n, np.int32)
    act = np.empty(n, np.int32)
    isact = np.zeros(n, np.uint8)
    touched = np.empty(n, np.int32)
    t_reach = np.full(n, np.inf, np.float32)
    t_half = np.full(n, np.inf, np.float32)
    na = 0
    for c in seeds:
        if S[c] > 0:
            S[c] -= 1
            E[c] += 1
        if isact[c] == 0:
            isact[c] = 1
            act[na] = c
            na += 1
            t_reach[c] = 0.0
    nh = int(t_end) + 1
    tot = np.zeros((nh, 4), np.int64)
    sT = np.int64(S.sum())
    eT = np.int64(E.sum())
    zT = np.int64(0)
    rT = np.int64(0)
    tot[0, 0], tot[0, 1] = sT, eT
    snaps = np.zeros((len(snap_t), n), np.int32)
    si = 0
    nsteps = int(round(t_end / dt))
    pleave = np.zeros(5)
    pslow = np.zeros(5)
    for k in range(5):
        pleave[k] = 1.0 - math.exp(-k * mu * dt)
        pslow[k] = 1.0 - math.exp(-k * mu * dt * slow)
    kopen = np.zeros(n, np.uint8)
    for c in range(n):
        for d in range(4):
            j = nbr[c, d]
            if j >= 0 and block[j] == 0:
                kopen[c] += 1
    alive = True
    for step in range(nsteps):
        t1 = (step + 1) * dt
        # reactions inside each active cell
        for a in range(na):
            c = act[a]
            if E[c] == 0 and (Z[c] == 0 or S[c] == 0):
                continue
            s0, e0, z0 = S[c], E[c], Z[c]
            m = s_mult[c]
            s, e, z, r, bites = _cell(s0, e0, z0, Rr[c], beta * m, kappa * m, nu, dt)
            sT += s - s0
            eT += e - e0
            zT += z - z0
            rT += r - Rr[c]
            S[c], E[c], Z[c], Rr[c] = s, e, z, r
            if t_half[c] == np.inf and 2 * s < pop[c]:
                t_half[c] = t1
        # zombies step to neighbouring land cells
        nt = 0
        for a in range(na):
            c = act[a]
            zc = Z[c]
            if zc == 0:
                continue
            kk = kopen[c]
            if kk == 0:
                continue
            if S[c] == 0 and E[c] == 0:
                if step % slow != 0:
                    continue
                go = _binom(zc, pslow[kk])
            else:
                go = _binom(zc, pleave[kk])
            if go == 0:
                continue
            dZ[c] -= go
            left = kk
            for d in range(4):
                j = nbr[c, d]
                if j < 0 or block[j] != 0:
                    continue
                g = go if left == 1 else _binom(go, 1.0 / left)
                go -= g
                left -= 1
                if g > 0:
                    if dZ[j] == 0 and isact[j] == 0:
                        touched[nt] = j
                        nt += 1
                    dZ[j] += g
        for a in range(na):
            c = act[a]
            if dZ[c] != 0:
                Z[c] += dZ[c]
                dZ[c] = 0
        for q in range(nt):
            j = touched[q]
            Z[j] += dZ[j]
            dZ[j] = 0
            if isact[j] == 0:
                isact[j] = 1
                act[na] = j
                na += 1
                if t_reach[j] == np.inf:
                    t_reach[j] = t1
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
            tot[h, 0], tot[h, 1], tot[h, 2], tot[h, 3] = sT, eT, zT, rT
        while si < len(snap_t) and t1 + 1e-9 >= snap_t[si]:
            snaps[si] = S
            si += 1
        if na == 0:
            alive = False
            for hh in range(h + 1, nh):
                tot[hh, 0], tot[hh, 1], tot[hh, 2], tot[hh, 3] = sT, eT, zT, rT
            while si < len(snap_t):
                snaps[si] = S
                si += 1
            break
    return tot, snaps, t_reach, t_half, alive


@nb.njit(parallel=True, cache=True)
def us_many(pop, nbr, seed_cells, beta, kappa, nu, mu, dt, t_end, snap_t, seed0, s_mult, block, slow=5):
    """One single-person outbreak per entry of seed_cells, in parallel (run r uses seed seed0 + r)."""
    nr = len(seed_cells)
    n = len(pop)
    snaps = np.zeros((nr, len(snap_t), n), np.int32)
    alive_at = np.zeros((nr, len(snap_t)), np.bool_)
    totals = np.zeros((nr, int(t_end) + 1, 4), np.int64)
    for r in nb.prange(nr):
        sc = np.empty(1, np.int64)
        sc[0] = seed_cells[r]
        tot, sn, _, _, _ = us_run(pop, nbr, sc, beta, kappa, nu, mu, dt, t_end, snap_t, seed0 + r, s_mult, block, slow)
        snaps[r] = sn
        totals[r] = tot
        for q in range(len(snap_t)):
            h = int(snap_t[q])
            alive_at[r, q] = tot[h, 1] + tot[h, 2] > 0
    return snaps, alive_at, totals


@nb.njit(cache=True)
def exact_grid(pop, nbr, seeds, beta, kappa, nu, mu, t_end, seed):
    """Fully exact (Gillespie) version of us_run for small grids: humans left per cell at t_end, and alive flag."""
    np.random.seed(seed)
    n = len(pop)
    S = pop.astype(np.int64).copy()
    E = np.zeros(n, np.int64)
    Z = np.zeros(n, np.int64)
    for c in seeds:
        S[c] -= 1
        E[c] += 1
    deg = np.zeros(n, np.int64)
    for c in range(n):
        for d in range(4):
            if nbr[c, d] >= 0:
                deg[c] += 1
    a = np.zeros(n)
    for c in range(n):
        a[c] = (beta + kappa) * S[c] * Z[c] + nu * E[c] + mu * deg[c] * Z[c]
    A = a.sum()
    t = 0.0
    since = 0
    while A > 1e-12:
        t += -math.log(1.0 - np.random.random()) / A
        if t > t_end:
            break
        u = np.random.random() * A
        c = 0
        acc = a[0]
        while acc < u and c < n - 1:
            c += 1
            acc += a[c]
        v = np.random.random() * a[c]
        bz = beta * S[c] * Z[c]
        kz = kappa * S[c] * Z[c]
        ez = nu * E[c]
        if v < bz:
            S[c] -= 1
            E[c] += 1
        elif v < bz + kz:
            Z[c] -= 1
        elif v < bz + kz + ez:
            E[c] -= 1
            Z[c] += 1
        else:
            k = np.random.randint(deg[c])
            j = -1
            for d in range(4):
                if nbr[c, d] >= 0:
                    if k == 0:
                        j = nbr[c, d]
                        break
                    k -= 1
            Z[c] -= 1
            Z[j] += 1
            oa = a[j]
            a[j] = (beta + kappa) * S[j] * Z[j] + nu * E[j] + mu * deg[j] * Z[j]
            A += a[j] - oa
        oa = a[c]
        a[c] = (beta + kappa) * S[c] * Z[c] + nu * E[c] + mu * deg[c] * Z[c]
        A += a[c] - oa
        since += 1
        if since > 100000:              # avoid drift of the running sum
            A = a.sum()
            since = 0
    return S, (E.sum() + Z.sum()) > 0


def land_index(land):
    """Index of the land cells and their 4 land neighbours (-1 = sea or border)."""
    H, W = land.shape
    idx = -np.ones((H, W), np.int64)
    ys, xs = np.nonzero(land)
    idx[ys, xs] = np.arange(len(ys))
    nbr = -np.ones((len(ys), 4), np.int32)
    for d, (dy, dx) in enumerate(((0, 1), (0, -1), (1, 0), (-1, 0))):
        y2, x2 = ys + dy, xs + dx
        ok = (y2 >= 0) & (y2 < H) & (x2 >= 0) & (x2 < W)
        j = np.full(len(ys), -1, np.int64)
        j[ok] = idx[y2[ok], x2[ok]]
        nbr[:, d] = j
    return idx, ys, xs, nbr
