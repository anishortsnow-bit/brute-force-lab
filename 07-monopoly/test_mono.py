"""Tests for mono.py. Run: python test_mono.py"""
import time

import numpy as np

import mono

NONE = np.full(40, -1, np.int64)


def test_board():
    """The printed numbers of the standard US board."""
    assert int(mono.PRICE.sum()) == 5690                       # 22 streets, 4 railroads, 2 utilities
    assert int((mono.KIND == 1).sum()) == 22 and int((mono.KIND == 2).sum()) == 4 and int((mono.KIND == 3).sum()) == 2
    assert [int(x) for x in mono.GSIZE] == [2, 3, 3, 3, 3, 3, 3, 2, 4, 2]
    assert mono.NAMES[24] == 'Illinois Avenue' and mono.NAMES[39] == 'Boardwalk' and mono.NAMES[10].startswith('Jail')
    assert int(mono.RENT[39, 5]) == 2000 and int(mono.RENT[1, 0]) == 2
    for s in range(40):
        if mono.KIND[s] == 1:
            assert all(mono.RENT[s, k] < mono.RENT[s, k + 1] for k in range(5)), s


def test_rent():
    owner = NONE.copy()
    houses = np.zeros(40, np.int64)
    mort = np.zeros(40, np.bool_)
    owner[39] = 0
    assert mono._rent(39, owner, houses, mort, 7) == 50          # Boardwalk alone
    owner[37] = 0
    assert mono._rent(39, owner, houses, mort, 7) == 100         # complete group, no houses: double
    houses[39] = 5
    houses[37] = 5
    assert mono._rent(39, owner, houses, mort, 7) == 2000        # hotel
    for k, s in enumerate((5, 15, 25, 35)):
        owner[s] = 1
        assert mono._rent(5, owner, houses, mort, 7) == 25 * 2 ** k
    owner[12] = 2
    assert mono._rent(12, owner, houses, mort, 9) == 36          # one utility: 4 x dice
    owner[28] = 2
    assert mono._rent(12, owner, houses, mort, 9) == 90          # both: 10 x dice
    mort[12] = True
    assert mono._rent(12, owner, houses, mort, 9) == 0           # mortgaged: no rent


def test_landing_matches_published_table():
    """Long-run share of rolls that end on each square, against I. Stewart, Scientific American, 1996."""
    c = mono.movement(7, 40_000_000, 0).astype(float)
    p = 100 * c / c.sum()
    for sq, v in {24: 3.18, 0: 3.11, 25: 3.06, 5: 2.99, 15: 2.91, 35: 2.44, 28: 2.81, 12: 2.62, 36: 0.87}.items():
        assert abs(p[sq] - v) <= 0.05, (mono.NAMES[sq], p[sq], v)
    assert int(np.argmax(p[:40])) in (10, 24)                    # Jail (visiting included) or Illinois on top
    assert p[30] == 0                                            # nobody ends a roll on Go To Jail


def test_invariants_hold_in_every_turn():
    """Houses and hotels add up to the box, buildings stay even, nobody owns a mortgaged lot with houses, no negative cash."""
    base = mono.policy()
    mixes = [([base] * 4, mono.options(check=1)),
             ([mono.policy(build_reserve=0), mono.policy(max_level=3), mono.policy(jail=2), mono.policy(trade=0)], mono.options(check=1)),
             ([base, mono.policy(random=0.5), mono.policy(buy_mask=255), mono.policy(bid=1.5)], mono.options(check=1, jackpot=2)),
             ([base] * 6, mono.options(check=1, no_auction=1)),
             ([base, base], mono.options(check=1, trades=0, start_cash=1000))]
    for POL, opt in mixes:
        out = mono.batch(11, 24, 50, len(POL), np.array(POL), NONE, opt, 1)
        assert int(out[6].sum()) == 0


def test_same_seed_same_game():
    POL = np.array([mono.policy()] * 4)
    a = np.zeros(8, np.int64)
    b = np.zeros(8, np.int64)
    rec = np.zeros((1, 1), np.int64)
    diff = 0
    for seed in range(50):
        mono.play(seed, 4, POL, NONE, mono.options(), a, rec)
        mono.play(seed, 4, POL, NONE, mono.options(), b, rec)
        assert np.array_equal(a, b)
        mono.play(seed + 1000, 4, POL, NONE, mono.options(), b, rec)
        diff += not np.array_equal(a, b)
    assert diff >= 45


def test_identical_players_win_equally():
    """With the seats rotating, four copies of the same player each win the same share, within the noise."""
    n = 240 * 500
    W, SEAT, U, *_ = [x.sum(axis=0) for x in mono.batch(3, 240, 500, 4, np.array([mono.policy()] * 4), NONE, mono.options(), 1)]
    assert W.sum() + U == n
    share = W / n
    se = np.sqrt(share * (1 - share) / n)
    assert np.all(np.abs(share - share.mean()) < 4 * se), share
    assert SEAT[0] > SEAT[3]                                     # the first seat does better than the last


def test_head_start_helps():
    """A player who starts with all four railroads wins far more than a fair share."""
    so = NONE.copy()
    so[[5, 15, 25, 35]] = 0
    W, _, U, *_ = [x.sum(axis=0) for x in mono.batch(5, 24, 500, 4, np.array([mono.policy()] * 4), so, mono.options(), 1)]
    assert W[0] / (24 * 500) > 0.6


if __name__ == '__main__':
    t0 = time.time()
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'):
            t1 = time.time()
            fn()
            print(f'ok  {name}  ({time.time() - t1:.1f}s)')
    print(f'all tests passed in {time.time() - t0:.0f}s')
