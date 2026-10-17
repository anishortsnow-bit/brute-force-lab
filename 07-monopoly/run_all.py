"""Every number in the Monopoly video comes from this file.

1. Where the tokens land, checked against the published long-run table (I. Stewart, Scientific American, 1996).
2. Full games between identical players: how long they last, who wins, what the first complete color group is worth.
3. One thing changed at a time: number of players, trading, house rules, a head start, and the usual strategy tips.

A game still running after 1,000 rounds counts as unfinished: nobody wins it.

Run: python run_all.py            (python run_all.py --quick for a small run)
"""
import json
import os
import sys
import time

import numpy as np

import mono

SEED = 2026
HERE = os.path.dirname(os.path.abspath(__file__))
QUICK = '--quick' in sys.argv
K = 0.02 if QUICK else 1.0
NONE = np.full(40, -1, np.int64)
LOG = open(os.path.join(HERE, 'run_quick.log' if QUICK else 'run_all.log'), 'w', encoding='utf-8')
TOTAL = [0]


def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + '\n')
    LOG.flush()


def games(name, POL, opt, n, so=NONE, seed=SEED):
    """n games; the policies rotate through the seats. Returns the summary."""
    npl = len(POL)
    n = max(240 * npl, int(n * K))
    per = n // 240
    t0 = time.time()
    W, SEAT, U, R, F, WG, ERR, NOM = [a.sum(axis=0) for a in mono.batch(seed, 240, per, npl, np.array(POL), so, opt, 1)]
    n = 240 * per
    TOTAL[0] += n
    assert ERR == 0
    fin = n - U
    cum = np.cumsum(R) / max(1, fin)
    out = dict(name=name, games=int(n), players=npl, seconds=round(time.time() - t0, 1),
               win=[float(x) for x in W / n], win_se=[float(np.sqrt(max(x, 1) / n * (1 - x / n) / n)) for x in W],
               seat_win_finished=[float(x) for x in SEAT / max(1, fin)], unfinished=float(U / n), no_monopoly=float(NOM / n),
               rounds_median=int(np.searchsorted(cum, 0.5)), rounds_p10=int(np.searchsorted(cum, 0.1)), rounds_p90=int(np.searchsorted(cum, 0.9)),
               rounds_mean=float((R * np.arange(len(R))).sum() / max(1, fin)), rounds_hist=[int(x) for x in R[:400]],
               first_group_share=[float(x) for x in F[:, 0] / max(1, F[:, 0].sum())],
               first_group_owner_wins=[float(x) for x in F[:, 1] / np.maximum(1, F[:, 2])],
               first_group_games=[int(x) for x in F[:, 0]])
    log(f"[{name}] {n:,} games, {out['seconds']}s | win {np.round(W / n, 4)} | unfinished {U / n:.2%} | rounds median {out['rounds_median']}")
    return out


def main():
    t00 = time.time()
    R = dict(seed=SEED)
    base = mono.policy()

    # ---- 1. where the tokens land
    nrolls = int(400_000_000 * K)
    mv = {}
    for stay in (0, 1):
        parts = [mono.movement(SEED + 1000 * stay + k, nrolls // 8, stay) for k in range(8)]
        c = np.sum(parts, axis=0).astype(float)
        mv['stay' if stay else 'leave'] = [float(x) for x in c / c.sum()]
    R['movement'] = dict(rolls=nrolls, names=mono.NAMES + ['In Jail'], **mv)
    p = mv['leave']
    stewart = {24: 3.18, 0: 3.11, 25: 3.06, 5: 2.99, 15: 2.91, 35: 2.44, 28: 2.81, 12: 2.62, 36: 0.87}
    R['stewart'] = []
    for sq, v in stewart.items():
        ok = abs(100 * p[sq] - v) <= 0.04
        log(f"  check: {mono.NAMES[sq]} (square {sq}): published {v}%, mine {100 * p[sq]:.3f}% {'OK' if ok else 'DIFFERENT'}")
        R['stewart'].append(dict(square=sq, name=mono.NAMES[sq], published=v, mine=100 * p[sq], ok=bool(ok)))
        if not QUICK:
            assert ok, mono.NAMES[sq]
    order = np.argsort(-np.array(p[:40]))
    log('  most landed:', [(mono.NAMES[i], round(100 * p[i], 2)) for i in order[:6]], '| in jail', round(100 * p[40], 2))

    # ---- 2. the baseline: four identical players, the swap rule
    R['base4'] = games('4 players', [base] * 4, mono.options(), 4_000_000)

    # ---- 3. one thing at a time
    R['players'] = {str(n): games(f'{n} players', [base] * n, mono.options(), 1_000_000) for n in (2, 3, 5, 6)}
    R['no_trades'] = games('no trades', [base] * 4, mono.options(trades=0), 1_000_000)
    R['jackpot'] = games('Free Parking jackpot', [base] * 4, mono.options(jackpot=1), 1_000_000)
    R['no_auction'] = games('no auctions', [base] * 4, mono.options(no_auction=1), 1_000_000)
    R['house_both'] = games('jackpot and no auctions', [base] * 4, mono.options(jackpot=1, no_auction=1), 1_000_000)
    R['jackpot500'] = games('Free Parking jackpot seeded with $500', [base] * 4, mono.options(jackpot=2), 1_000_000)
    R['jackpot500_no_auction'] = games('seeded jackpot and no auctions', [base] * 4, mono.options(jackpot=2, no_auction=1), 1_000_000)
    R['long_cap'] = games('4 players, cap of 10,000 rounds', [base] * 4, mono.options(max_rounds=10000), 200_000)
    R['cash'] = {str(c): games(f'start with {c}', [base] * 4, mono.options(start_cash=c), 500_000) for c in (1000, 2000, 3000)}

    # set against set: two players, each owns one complete group before the first roll
    R['duel'] = {}
    for ga in range(8):
        for gb in range(ga + 1, 8):
            so = NONE.copy()
            for k in range(mono.GSIZE[ga]):
                so[mono.GMEM[ga, k]] = 0
            for k in range(mono.GSIZE[gb]):
                so[mono.GMEM[gb, k]] = 1
            g = games(f'duel: {mono.GROUP_NAMES[ga]} vs {mono.GROUP_NAMES[gb]}', [base, base], mono.options(trades=0), 200_000, so)
            R['duel'][f'{ga}-{gb}'] = dict(a=mono.GROUP_NAMES[ga], b=mono.GROUP_NAMES[gb], win_a=g['win'][0], win_b=g['win'][1],
                                           unfinished=g['unfinished'], rounds_median=g['rounds_median'], games=g['games'])

    # a head start: one player owns a whole group before the first roll
    R['head_start'] = {}
    for g in range(10):
        so = NONE.copy()
        for k in range(mono.GSIZE[g]):
            so[mono.GMEM[g, k]] = 0
        R['head_start'][mono.GROUP_NAMES[g]] = games(f'head start: {mono.GROUP_NAMES[g]}', [base] * 4, mono.options(), 500_000, so)

    # the tips: player 0 changes one habit, the other three stay the same
    tips = dict(
        build_reserve_0=dict(build_reserve=0), build_reserve_300=dict(build_reserve=300), build_reserve_600=dict(build_reserve=600),
        three_houses=dict(max_level=3), four_houses=dict(max_level=4),
        jail_stay=dict(jail=1), jail_late=dict(jail=2),
        buy_reserve_200=dict(buy_reserve=200), buy_reserve_500=dict(buy_reserve=500),
        skip_utilities=dict(buy_mask=1023 - 512), skip_railroads=dict(buy_mask=1023 - 256), skip_brown=dict(buy_mask=1023 - 1),
        skip_green=dict(buy_mask=1023 - 64), skip_dark_blue=dict(buy_mask=1023 - 128), skip_rail_and_util=dict(buy_mask=255),
        only_streets_side_1_2=dict(buy_mask=0b1100001111), bid_half=dict(bid=0.5), bid_150=dict(bid=1.5), bid_set_1=dict(bid_set=1.0),
        bid_set_4=dict(bid_set=4.0), never_trades=dict(trade=0))
    R['tips'] = {}
    for name, kw in tips.items():
        R['tips'][name] = games(f'tip: {name}', [mono.policy(**kw), base, base, base], mono.options(), 1_000_000)

    # the best habits together, and against coin flips
    best_kw = {}
    for fam in (('build_reserve_0', 'build_reserve_300', 'build_reserve_600'), ('three_houses', 'four_houses'), ('jail_stay', 'jail_late'),
                ('buy_reserve_200', 'buy_reserve_500'), ('bid_half', 'bid_150'), ('bid_set_1', 'bid_set_4')):
        top = max(fam, key=lambda k: R['tips'][k]['win'][0])
        if R['tips'][top]['win'][0] > R['base4']['win'][0] + 3 * R['tips'][top]['win_se'][0]:
            best_kw.update(tips[top])
    R['best_kw'] = best_kw
    best = mono.policy(**best_kw)
    R['best_vs_base4'] = games('best habits vs three baseline players', [best, base, base, base], mono.options(), 2_000_000)
    coin = mono.policy(random=0.5)
    R['base_vs_coin2'] = games('baseline vs coin flips, 2 players', [base, coin], mono.options(), 1_000_000)
    R['best_vs_coin2'] = games('best habits vs coin flips, 2 players', [best, coin], mono.options(), 1_000_000)
    R['best_vs_base2'] = games('best habits vs baseline, 2 players', [best, base], mono.options(), 1_000_000)
    R['base_vs_coin4'] = games('baseline vs three coin-flip players', [base, coin, coin, coin], mono.options(), 1_000_000)

    R['total_games'] = int(TOTAL[0])
    log(f'total games: {TOTAL[0]:,} in {time.time() - t00:.0f}s; best habits: {best_kw}')
    json.dump(R, open(os.path.join(HERE, 'resultados_quick.json' if QUICK else 'resultados.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
