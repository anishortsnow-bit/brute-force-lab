"""Every number in the video "I Simulated How Much Luck You Need to Get Rich" comes from this file.

1. The original Talent vs Luck model, one million worlds of 1,000 people, checked against the published results.
2. One rule changed at a time (100,000 worlds each, the same people and seeds as the first 100,000 worlds above).
3. The replay: the same people live their careers twice.

"Rich" always means the richest 1% of a world: 10 people out of 1,000, ties broken at random.
"Luck" is counted before looking at who got rich: lucky events received minus unlucky events received.

Run: python run_all.py           (about ten minutes on 24 threads)
"""
import json
import os
import sys
import time

import numpy as np

import tvl

SEED = 2026
N = 1000
HERE = os.path.dirname(os.path.abspath(__file__))
QUICK = '--quick' in sys.argv
BASE = (25, 400) if QUICK else (250, 4000)          # chunks x worlds per chunk
VAR = (25, 400) if QUICK else (100, 1000)
REP_BASE = (25, 400) if QUICK else (200, 1000)
REP_VAR = (25, 200) if QUICK else (100, 500)
SWEEP = (25, 80) if QUICK else (50, 600)
SWEEP_REP = (25, 40) if QUICK else (50, 300)

LOG = open(os.path.join(HERE, 'run_all.log' if not QUICK else 'run_quick.log'), 'w', encoding='utf-8')


def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + '\n')
    LOG.flush()


def duel(H_grp, a, b):
    """P(a person from talent group a ends richer than one from group b), a tie counting as half."""
    ha, hb = H_grp[a] / H_grp[a].sum(), H_grp[b] / H_grp[b].sum()
    below = np.cumsum(hb) - hb
    return float((ha * below).sum() + 0.5 * (ha * hb).sum())


def summ(name, P, shape, seed=SEED):
    t0 = time.time()
    out = tvl.batch(seed, shape[0], shape[1], N, P)
    H_tl, H_dl, H_lu, H_c, H_grp, H_best, H_rtp, H_rlp, H_rtl = [a.sum(axis=0) for a in out[:-1]]
    Sc = out[-1]
    S = Sc.sum(axis=0)
    S[15], S[16], S[25] = Sc[:, 15].max(), Sc[:, 16].max(), Sc[:, 25].max()
    w, n, nr = S[0], S[29], S[36]
    best_mean = S[20] / w
    tp = H_rtp / H_rtp.sum()
    lp = H_rlp / H_rlp.sum()
    pc = np.arange(100) + 0.5
    dl = H_dl.sum(axis=1)                               # talent decile x rich
    nlk = H_dl.sum(axis=0)                              # net luck x rich
    tl = H_tl.sum(axis=1)                               # talent bin x rich
    net_pop = nlk.sum(axis=1) / nlk.sum()
    res = dict(
        name=name, worlds=int(w), careers=int(n), seconds=round(time.time() - t0, 1), params=[float(x) for x in P],
        share_top1=S[1] / w, share_top10=S[2] / w, share_top20=S[3] / w, share_bottom50=S[28] / w, gini=S[4] / w,
        below_start=S[5] / n, at_start=S[6] / n, above_start=S[7] / n, mean_capital=S[30] / n,
        lucky_mean=S[13] / n, unlucky_mean=S[14] / n, used_mean=S[35] / n, lucky_max=int(S[15]), unlucky_max=int(S[16]),
        no_events=S[17] / n, one_kind=S[18] / n,
        best_talent_mean=best_mean, best_talent_sd=float((S[21] / w - best_mean ** 2) ** 0.5), best_talent_gt08=S[19] / w,
        best_capital_mean=S[26] / w, best_capital_max=S[25],
        t07_per_world=S[8] / w, t07_above_start=S[9] / S[8],
        most_talented_capital_mean=S[10] / w, near_mean_capital_mean=S[11] / S[12],
        most_talented_rich=S[23] / w, most_talented_richest=S[24] / w, rich_threshold_log2_mean=S[22] / w,
        rich_talent_mean=S[27] / nr, rich_below_median_talent=S[31] / nr, rich_netluck_le0=S[32] / nr,
        rich_lucky_mean=S[33] / nr, rich_unlucky_mean=S[34] / nr,
        rich_talent_pct_mean=float((tp * pc).sum()), rich_talent_pct_median=int(np.searchsorted(tp.cumsum(), 0.5)),
        rich_top10_talent=float(tp[90:].sum()), rich_top1_talent=float(tp[99:].sum()),
        rich_luck_pct_mean=float((lp * pc).sum()), rich_luck_pct_median=int(np.searchsorted(lp.cumsum(), 0.5)),
        rich_luckiest10=float(lp[90:].sum()), rich_luckiest5=float(lp[95:].sum()), rich_luckiest1=float(lp[99:].sum()),
        rich_talent_pct_hist=[int(x) for x in H_rtp], rich_luck_pct_hist=[int(x) for x in H_rlp],
        rich_talent_x_luck_deciles=H_rtl.tolist(),
        p_rich_by_talent_decile=[float(x) for x in dl[:, 1] / dl.sum(axis=1)],
        talent_bins=dict(step=0.02, people=[int(x) for x in tl.sum(axis=1)], rich=[int(x) for x in tl[:, 1]]),
        best_talent_hist=[int(x) for x in H_best],
        net_luck=dict(offset=tvl.LKO, people=[int(x) for x in nlk.sum(axis=1)], rich=[int(x) for x in nlk[:, 1]]),
        decile_x_netluck=dict(people=H_dl.sum(axis=2).tolist(), rich=H_dl[:, :, 1].tolist()),
        lucky_dist=[int(x) for x in H_lu.sum(axis=(1, 2))], unlucky_dist=[int(x) for x in H_lu.sum(axis=(0, 2))],
        rich_lucky_dist=[int(x) for x in H_lu[:, :, 1].sum(axis=1)], rich_unlucky_dist=[int(x) for x in H_lu[:, :, 1].sum(axis=0)],
        capital_hist={str(i): int(v) for i, v in enumerate(H_c) if v},          # bin i -> log2(capital/start) = i * 0.05 - 20
        duel_top1_bottom1=duel(H_grp, 4, 0), duel_top10_bottom10=duel(H_grp, 3, 1),
        duel_top1_middle=duel(H_grp, 4, 2), duel_top10_middle=duel(H_grp, 3, 2),
    )
    # net luck needed: the smallest net luck with any rich person, and how rare it is
    idx = np.nonzero(nlk[:, 1])[0]
    res['rich_min_netluck'] = int(idx[0] - tvl.LKO)
    res['share_people_with_min_netluck_or_more'] = float(net_pop[idx[0]:].sum())
    for k in (3, 4, 5, 6):
        top = H_dl[9, k + tvl.LKO:, :].sum(axis=0)
        mid = H_dl[4:6, k + tvl.LKO:, :].sum(axis=(0, 1))
        res[f'p_rich_top_decile_netluck_ge{k}'] = float(top[1] / top.sum())
        res[f'p_rich_middle_netluck_ge{k}'] = float(mid[1] / mid.sum())
        res[f'share_netluck_ge{k}'] = float(net_pop[k + tvl.LKO:].sum())
    low = H_dl[9, :3 + tvl.LKO + 1, :].sum(axis=0)
    res['p_rich_top_decile_netluck_le3'] = float(low[1] / low.sum())
    log(f"[{name}] {int(w):,} worlds, {res['seconds']}s | top1% holds {res['share_top1']:.3f}, top20% {res['share_top20']:.3f}, "
        f"gini {res['gini']:.3f} | richest talent {best_mean:.4f}±{res['best_talent_sd']:.4f} | rich: talent pct "
        f"{res['rich_talent_pct_median']}, luck pct {res['rich_luck_pct_median']}, luckiest5 {res['rich_luckiest5']:.3f} | "
        f"P(rich|top decile) {res['p_rich_by_talent_decile'][9]:.4f} | duel 1%v1% {res['duel_top1_bottom1']:.4f}")
    return res


def replay(name, P, shape, seed=SEED):
    t0 = time.time()
    H_ov, H_pos, H_dec, S = [a.sum(axis=0) for a in tvl.replay_pairs(seed, shape[0], shape[1], N, P)]
    w, nr = S[0], S[1]
    pos = H_pos / H_pos.sum()
    res = dict(
        name=name, worlds=int(w), careers=int(2 * w * N), seconds=round(time.time() - t0, 1),
        rich_again=float((H_ov * np.arange(len(H_ov))).sum() / nr), overlap_hist=[int(x) for x in H_ov],
        worlds_with_nobody_again=float(H_ov[0] / w),
        land_top10=float(pos[90:].sum()), land_top_half=float(pos[50:].sum()), land_bottom_half=float(pos[:50].sum()),
        land_median_pct=int(np.searchsorted(pos.cumsum(), 0.5)), land_hist=[int(x) for x in H_pos],
        below_start=S[2] / nr, not_above_start=S[3] / nr,
        again_by_talent_decile=[float(H_dec[d, 1] / max(1, H_dec[d].sum())) for d in range(10)],
    )
    log(f"[replay {name}] {int(w):,} worlds twice, {res['seconds']}s | rich again {res['rich_again']:.4f} | land top10% "
        f"{res['land_top10']:.3f}, bottom half {res['land_bottom_half']:.3f}, not above start {res['not_above_start']:.3f}")
    return res


def main():
    t00 = time.time()
    R = dict(seed=SEED, people_per_world=N)
    base = summ('base', tvl.params(), BASE)
    R['base'] = base

    # ---- the published results (Pluchino, Biondo & Rapisarda 2018)
    checks = [
        ('talent of the richest person, mean (10,000 runs in the paper)', base['best_talent_mean'], 0.667, 0.006),
        ('talent of the richest person, standard deviation', base['best_talent_sd'], 0.09, 0.006),
        ('people with talent > 0.7 per world', base['t07_per_world'], 160, 3),
        ('of those, share ending above the starting capital', base['t07_above_start'], 0.32, 0.01),
        ('share of people with no event at all', base['no_events'], 0.16, 0.01),
        ('share of capital held by the richest 20% ("80-20")', base['share_top20'], 0.80, 0.05),
    ]
    R['paper_checks'] = []
    for what, mine, paper, tol in checks:
        ok = abs(mine - paper) <= tol
        log(f"  check: {what}: paper {paper}, mine {mine:.4f} {'OK' if ok else 'DIFFERENT'}")
        R['paper_checks'].append(dict(what=what, paper=paper, mine=mine, ok=bool(ok)))
        if not QUICK:
            assert ok, what

    # ---- one rule at a time
    q = (base['lucky_mean'] + base['unlucky_mean']) / 2 / 80      # same number of events per person, without the map
    R['indep_q'] = q
    variants = dict(
        indep=tvl.params(indep=q),                 # luck with no streaks
        mild=tvl.params(up=1.5, down=1.5),         # gentler events
        opp=tvl.params(opp=1),                     # talent attracts opportunities
        pay=tvl.params(pay=1.0),                   # a paycheck in proportion to talent
        shield=tvl.params(shield=1),               # talent softens bad luck
        start=tvl.params(start=1.0),               # unequal start
        all=tvl.params(opp=1, pay=1.0, shield=1),  # every advantage for talent at once
    )
    R['base100k'] = summ('base100k', tvl.params(), VAR)
    for name, P in variants.items():
        R[name] = summ(name, P, VAR)

    # ---- the replay
    R['replay'] = dict(base=replay('base', tvl.params(), REP_BASE))
    for name in ('indep', 'opp', 'pay', 'shield', 'all'):
        R['replay'][name] = replay(name, variants[name], REP_VAR)

    # ---- how gentle must luck be before talent takes over? The paycheck world, with events from x2 down to x1 (no effect)
    R['sweep'] = []
    sweep_careers = 0
    for f in (2.0, 1.7, 1.5, 1.3, 1.2, 1.1, 1.05, 1.0):
        P = tvl.params(pay=1.0, up=f, down=f)
        a = summ(f'pay, events x{f}', P, SWEEP)
        b = replay(f'pay, events x{f}', P, SWEEP_REP)
        sweep_careers += a['careers'] + b['careers']
        R['sweep'].append(dict(
            factor=f, rich_again=b['rich_again'], land_top10=b['land_top10'], share_top1=a['share_top1'],
            rich_top10_talent=a['rich_top10_talent'], rich_top1_talent=a['rich_top1_talent'],
            rich_talent_pct_median=a['rich_talent_pct_median'], rich_luck_pct_median=a['rich_luck_pct_median'],
            rich_luckiest5=a['rich_luckiest5'], duel_top1_bottom1=a['duel_top1_bottom1'],
            most_talented_rich=a['most_talented_rich'], p_rich_top_decile=a['p_rich_by_talent_decile'][9]))

    careers = sum(v['careers'] for k, v in R.items() if isinstance(v, dict) and 'careers' in v)
    careers += sum(v['careers'] for v in R['replay'].values()) + sweep_careers
    R['total_careers'] = int(careers)
    log(f'total careers simulated: {careers:,} in {time.time() - t00:.0f}s')
    json.dump(R, open(os.path.join(HERE, 'resultados.json' if not QUICK else 'resultados_quick.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
