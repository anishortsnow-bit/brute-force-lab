"""The one world shown on screen, and its replays. Writes samples.json.

The world is one of the first 1,000 worlds of run_all.py (same seed, so it is one of the worlds that were counted).
It is the first world, in seed order, that looks like the typical case rather than an extreme one:
  - one person is clearly the richest (no tie for first place);
  - that person's talent is within 0.03 of the average talent of the richest person over all worlds (0.67);
  - the richest 1% hold between 30% and 40% of all capital (the average is 35%);
  - the most talented person ends with no more than the starting capital (true in most worlds).
Then the same 1,000 people live 10,000 more careers.

Run: python samples.py
"""
import json
import os

import numpy as np

import tvl

SEED = 2026
N = 1000
LIVES = 10000
KEEP = 11
HERE = os.path.dirname(os.path.abspath(__file__))
P = tvl.params()


def play(seed):
    tal = np.empty(N)
    ax = np.empty(N)
    ay = np.empty(N)
    cap = np.empty(N)
    nl = np.empty(N, np.int32)
    nu = np.empty(N, np.int32)
    used = np.empty(N, np.int32)
    rich = np.empty(N, np.uint8)
    tvl.world(seed, N, P, tal, ax, ay, cap, nl, nu, used, rich)
    return tal, ax, ay, cap, nl, nu, used, rich


def main():
    chosen = None
    stats = dict(tested=0, most_talented_not_above_start=0, most_talented_rich=0)
    for seed in range(SEED, SEED + 1000):
        tal, ax, ay, cap, nl, nu, used, rich = play(seed)
        srt = np.sort(cap)
        imt = int(np.argmax(tal))
        stats['tested'] += 1
        stats['most_talented_not_above_start'] += int(cap[imt] <= 10)
        stats['most_talented_rich'] += int(rich[imt])
        ok = (srt[-1] > srt[-2] and abs(tal[int(np.argmax(cap))] - 0.67) <= 0.03
              and 0.30 <= srt[-10:].sum() / cap.sum() <= 0.40 and cap[imt] <= 10)
        if ok and chosen is None:
            chosen = seed
    print('chosen world', chosen, stats)

    h = tvl.history(chosen)
    tal, ax, ay, cap, nl, nu, used, rich = play(chosen)
    assert np.array_equal(h['cap'], cap) and np.array_equal(h['tal'], tal)
    k = np.round(np.log2(h['h_cap'] / 10.0)).astype(int)            # capital = 10 * 2 ** k, exactly
    assert np.allclose(10.0 * 2.0 ** k, h['h_cap'])
    ibest = int(np.argmax(cap))
    imt = int(np.argmax(tal))
    events = [[int(s), int(i), int(h['h_ev'][s, i])] for s, i in zip(*np.nonzero(h['h_ev']))]
    order = np.argsort(-cap, kind='stable')

    rich_count = np.zeros(N, np.int64)
    caps = np.empty((KEEP, N))
    richk = np.empty((KEEP, N // 100), np.int64)
    tal2 = tvl.lives(chosen, 3_000_000_000, LIVES, N, P, rich_count, caps, richk, KEEP)
    assert np.array_equal(tal2, tal)
    rich1 = [int(i) for i in np.nonzero(rich)[0]]
    again = [int(len(set(rich1) & set(int(x) for x in richk[j]))) for j in range(KEEP)]

    out = dict(
        seed=chosen, selection=stats, lives=LIVES,
        tal=[round(float(x), 4) for x in tal], ax=[round(float(x), 2) for x in ax], ay=[round(float(x), 2) for x in ay],
        k_final=[int(x) for x in k[-1]], nl=[int(x) for x in nl], nu=[int(x) for x in nu], used=[int(x) for x in used],
        rich=rich1, richest=ibest, most_talented=imt,
        k_hist_richest=[int(x) for x in k[:, ibest]], k_hist_most_talented=[int(x) for x in k[:, imt]],
        ev_richest=[int(x) for x in h['h_ev'][:, ibest]], ev_most_talented=[int(x) for x in h['h_ev'][:, imt]],
        events=events,
        ex=np.round(h['h_ex'], 1).tolist(), ey=np.round(h['h_ey'], 1).tolist(),
        share_top1=float(np.sort(cap)[-10:].sum() / cap.sum()), share_top20=float(np.sort(cap)[-200:].sum() / cap.sum()),
        below_start=int((cap < 10).sum()), at_start=int((cap == 10).sum()), above_start=int((cap > 10).sum()),
        no_events=int(((nl == 0) & (nu == 0)).sum()),
        top10=[dict(person=int(i), talent=round(float(tal[i]), 3), talent_rank=int((tal > tal[i]).sum()) + 1,
                    capital=float(cap[i]), lucky=int(nl[i]), used=int(used[i]), unlucky=int(nu[i])) for i in order[:10]],
        most_talented_info=dict(talent=round(float(tal[imt]), 3), capital=float(cap[imt]), lucky=int(nl[imt]),
                                used=int(used[imt]), unlucky=int(nu[imt]), wealth_rank=int((cap > cap[imt]).sum()) + 1),
        replay=dict(
            rich_sets=[[int(x) for x in richk[j]] for j in range(KEEP)],
            k_final=[[int(x) for x in np.round(np.log2(caps[j] / 10.0))] for j in range(KEEP)],
            again=again,
            rich_count=[int(x) for x in rich_count],
            rich_count_most_talented=int(rich_count[imt]), rich_count_richest=int(rich_count[ibest]),
            rich_count_first_rich=[int(rich_count[i]) for i in rich1],
            never_rich=int((rich_count == 0).sum()), max_rich_count=int(rich_count.max()),
            talent_of_max=round(float(tal[int(np.argmax(rich_count))]), 3)),
    )
    json.dump(out, open(os.path.join(HERE, 'samples.json'), 'w'))
    b = out['top10'][0]
    m = out['most_talented_info']
    print(f"richest: talent {b['talent']} (rank {b['talent_rank']}), capital {b['capital']:.0f}, lucky {b['lucky']} used {b['used']} unlucky {b['unlucky']}")
    print(f"most talented: talent {m['talent']}, capital {m['capital']}, lucky {m['lucky']} used {m['used']} unlucky {m['unlucky']}, wealth rank {m['wealth_rank']}")
    print(f"top1 share {out['share_top1']:.3f} top20 {out['share_top20']:.3f} below {out['below_start']} at {out['at_start']} above {out['above_start']} no events {out['no_events']}")
    print('top10:', [(d['talent'], d['talent_rank'], int(d['capital']), d['lucky'], d['unlucky']) for d in out['top10']])
    print('replay: rich again in lives 2..12:', again, '| in 10,000 lives: most talented rich', out['replay']['rich_count_most_talented'],
          'the richest', out['replay']['rich_count_richest'], 'first rich', out['replay']['rich_count_first_rich'],
          'max', out['replay']['max_rich_count'], 'talent of max', out['replay']['talent_of_max'], 'never', out['replay']['never_rich'])


if __name__ == '__main__':
    main()
