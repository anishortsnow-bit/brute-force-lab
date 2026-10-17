"""Tests for tvl.py. Run: python test_tvl.py"""
import math
import time

import numpy as np

import tvl


def test_events_match_brute_force():
    """Every recorded event is an event point within 1 patch of the person (on the torus), and none is missed."""
    P = tvl.params()
    h = tvl.history(7)
    L = P[tvl.L_]
    nlucky = int(P[tvl.NE_] * P[tvl.PL_])
    for s in range(1, int(P[tvl.STEPS_]) + 1):
        dx = np.abs(h['ax'][:, None] - h['h_ex'][s][None, :])
        dx = np.minimum(dx, L - dx)
        dy = np.abs(h['ay'][:, None] - h['h_ey'][s][None, :])
        dy = np.minimum(dy, L - dy)
        near = dx * dx + dy * dy <= 1.0
        lucky = near[:, :nlucky].any(axis=1)
        unlucky = near[:, nlucky:].any(axis=1)
        code = h['h_ev'][s]
        assert np.array_equal((code & 1) > 0, lucky), s
        assert np.array_equal((code & 4) > 0, unlucky), s
    assert np.array_equal(h['nl'], ((h['h_ev'] & 1) > 0).sum(axis=0))
    assert np.array_equal(h['nu'], ((h['h_ev'] & 4) > 0).sum(axis=0))
    assert np.array_equal(h['used'], ((h['h_ev'] & 2) > 0).sum(axis=0))


def test_event_points_move_two_patches():
    P = tvl.params()
    h = tvl.history(11)
    L = P[tvl.L_]
    dx = np.abs(np.diff(h['h_ex'], axis=0))
    dx = np.minimum(dx, L - dx)
    dy = np.abs(np.diff(h['h_ey'], axis=0))
    dy = np.minimum(dy, L - dy)
    assert np.allclose(np.sqrt(dx * dx + dy * dy), 2.0, atol=1e-9)


def test_capital_is_start_times_powers_of_two():
    h = tvl.history(3)
    assert np.allclose(h['cap'], 10.0 * 2.0 ** (h['used'] - h['nu']))
    assert np.allclose(h['h_cap'][-1], h['cap'])
    assert h['used'].max() <= h['nl'].max()
    assert (h['used'] <= h['nl']).all()


def test_same_seed_same_world():
    a = tvl.history(5)
    b = tvl.history(5)
    c = tvl.history(6)
    assert np.array_equal(a['cap'], b['cap']) and np.array_equal(a['tal'], b['tal'])
    assert not np.array_equal(a['tal'], c['tal'])
    d = tvl.history(5, life_seed=99)
    assert np.array_equal(a['tal'], d['tal']) and np.array_equal(a['ax'], d['ax'])
    assert not np.array_equal(a['cap'], d['cap'])


def test_talent_distribution():
    P = tvl.params()
    N = 200000
    tal = np.empty(N)
    ax = np.empty(N)
    ay = np.empty(N)
    seeded_population(1, N, P, tal, ax, ay)
    assert abs(tal.mean() - 0.6) < 0.001 and abs(tal.std() - 0.1) < 0.001
    assert tal.min() >= 0 and tal.max() <= 1
    assert ax.min() >= 0 and ax.max() < 201


def seeded_population(seed, N, P, tal, ax, ay):
    tvl._seed(seed)
    tvl.population(N, P, tal, ax, ay)


def test_independent_luck_closed_form():
    """With no map and one talent for everybody, the mean final capital has a closed form."""
    q, T = 0.02, 0.6
    P = tvl.params(indep=q, sT=1e-9, mT=T)
    out = tvl.batch(123, 24, 400, 1000, P)
    S = out[-1].sum(axis=0)
    n = S[29]
    mean_cap = S[30] / n
    exact = 10.0 * (1 + q * T) ** 80 * (1 - q / 2) ** 80
    assert abs(mean_cap / exact - 1) < 0.01, (mean_cap, exact)
    assert abs(S[13] / n - 80 * q) < 0.005 and abs(S[14] / n - 80 * q) < 0.005
    # a Poisson-like number of events: nobody gets none with probability (1 - q) ** 160
    assert abs(S[17] / n - (1 - q) ** 160) < 0.002


def test_batch_matches_single_worlds():
    """The batch statistics are the same numbers a plain loop over history() gives."""
    P = tvl.params()
    out = tvl.batch(500, 1, 20, 1000, P)
    S = out[-1].sum(axis=0)
    tot = 0.0
    nlsum = 0
    best_t = 0.0
    for r in range(20):
        h = tvl.history(500 + r)
        tot += h['cap'].sum()
        nlsum += h['nl'].sum()
        # the richest person, when there is no tie for first place
        top = np.sort(h['cap'])[-2:]
        if top[1] > top[0]:
            best_t += h['tal'][np.argmax(h['cap'])]
        else:
            best_t = float('nan')
    assert math.isclose(S[30], tot, rel_tol=1e-12)
    assert S[13] == nlsum
    if not math.isnan(best_t):
        assert math.isclose(S[20], best_t, rel_tol=1e-12)


def _world(seed, P, N=1000):
    tal = np.empty(N)
    ax = np.empty(N)
    ay = np.empty(N)
    cap = np.empty(N)
    nl = np.empty(N, np.int32)
    nu = np.empty(N, np.int32)
    used = np.empty(N, np.int32)
    rich = np.empty(N, np.uint8)
    tvl.world(seed, N, P, tal, ax, ay, cap, nl, nu, used, rich)
    return tal, cap, nl, nu, used, rich


def test_world_is_the_batch_world():
    """world(seed) replays exactly the world that batch() counted, and marks 10 people as rich."""
    h = tvl.history(41)
    tal, cap, nl, nu, used, rich = _world(41, tvl.params())
    assert np.array_equal(h['cap'], cap) and np.array_equal(h['tal'], tal) and np.array_equal(h['nl'], nl)
    assert rich.sum() == 10 and cap[rich == 1].min() >= cap[rich == 0].max()


def test_gentler_events_same_rich_people():
    """Events of x1.5 and /1.5 instead of x2 and /2 change how much the rich have, not who they are."""
    for seed in range(100, 150):
        a = _world(seed, tvl.params())
        b = _world(seed, tvl.params(up=1.5, down=1.5))
        assert np.array_equal(a[5], b[5])
        assert not np.array_equal(a[1], b[1])


def test_lives_keep_the_people():
    N = 1000
    P = tvl.params()
    rc = np.zeros(N, np.int64)
    caps = np.empty((3, N))
    richk = np.empty((3, 10), np.int64)
    tal = tvl.lives(77, 3_000_000_000, 20, N, P, rc, caps, richk, 3)
    assert np.array_equal(tal, tvl.history(77)['tal'])
    assert rc.sum() == 20 * 10
    assert not np.array_equal(caps[0], caps[1])
    for k in range(3):
        assert caps[k][richk[k]].min() >= np.sort(caps[k])[-10]


if __name__ == '__main__':
    t0 = time.time()
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name, f'{time.time() - t0:.1f}s')
    P = tvl.params()
    t0 = time.time()
    tvl.batch(1, 24, 200, 1000, P)
    dt = time.time() - t0
    print(f'speed: {24 * 200 / dt:,.0f} worlds per second ({24 * 200 * 1000 / dt:,.0f} careers per second)')
