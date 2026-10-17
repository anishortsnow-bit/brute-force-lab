"""Prints the numbers of resultados.json and samples.json in the order the video uses them. Run: python report.py"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, 'resultados.json')))
S = json.load(open(os.path.join(HERE, 'samples.json')))
B = R['base']


def pc(x, d=1):
    return f'{100 * x:.{d}f}%'


print(f"TOTAL careers: {R['total_careers']:,}   base: {B['worlds']:,} worlds = {B['careers']:,} careers")
print('\n== checks against the paper')
for c in R['paper_checks']:
    print(f"  {c['what']}: paper {c['paper']}  mine {c['mine']:.4f}  {'OK' if c['ok'] else 'DIFFERENT'}")

print('\n== inequality (base)')
print(f"  richest 1% hold {pc(B['share_top1'])}, richest 10% {pc(B['share_top10'])}, richest 20% {pc(B['share_top20'])}, "
      f"poorer half {pc(B['share_bottom50'], 2)}, Gini {B['gini']:.3f}")
print(f"  end below start {pc(B['below_start'])}, exactly at start {pc(B['at_start'])}, above {pc(B['above_start'])}; mean capital {B['mean_capital']:.1f}")
print(f"  richest person of a world: mean capital {B['best_capital_mean']:.0f}, max {B['best_capital_max']:.0f}; "
      f"threshold for the richest 1%: 10 x 2^{B['rich_threshold_log2_mean']:.2f} on average")
ch = {int(k): v for k, v in B['capital_hist'].items()}
tot = sum(ch.values())
ks = sorted(ch)
print('  final capital, by doublings k (capital = 10 * 2^k):', {round(k * 0.05 - 20): pc(ch[k] / tot, 2) for k in ks if ch[k] / tot > 0.0005})

print('\n== who is rich (base)')
print(f"  talent of the rich: mean {B['rich_talent_mean']:.3f}; percentile mean {B['rich_talent_pct_mean']:.1f}, median {B['rich_talent_pct_median']}; "
      f"below-median talent {pc(B['rich_below_median_talent'])}; from the top 10% of talent {pc(B['rich_top10_talent'])}; top 1% {pc(B['rich_top1_talent'])}")
print(f"  the single richest person: talent {B['best_talent_mean']:.3f} +- {B['best_talent_sd']:.3f}; has talent > 0.8 in {pc(B['best_talent_gt08'])} of worlds")
print(f"  the most talented person of a world: rich in {pc(B['most_talented_rich'], 2)} of worlds, the richest in {pc(B['most_talented_richest'], 2)} (a random person: 1% and 0.1%)")
print('  P(rich | talent decile):', [pc(x, 2) for x in B['p_rich_by_talent_decile']],
      f"  top/bottom = {B['p_rich_by_talent_decile'][9] / B['p_rich_by_talent_decile'][0]:.1f}x")

print('\n== luck, counted before looking at the outcome (base)')
ld = np.array(B['lucky_dist'], float)
ud = np.array(B['unlucky_dist'], float)
print(f"  events per person: lucky {B['lucky_mean']:.3f}, unlucky {B['unlucky_mean']:.3f} (lucky ones used {B['used_mean']:.3f}); "
      f"max lucky {B['lucky_max']}, max unlucky {B['unlucky_max']}; no event at all {pc(B['no_events'])}; only one kind {pc(B['one_kind'])}")
print('  P(lucky events = k), k = 0..8:', [pc(x) for x in (ld / ld.sum())[:9]], ' P(k >= 5):', pc((ld / ld.sum())[5:].sum(), 2))
print(f"  the rich: {B['rich_lucky_mean']:.2f} lucky and {B['rich_unlucky_mean']:.2f} unlucky events on average; with net luck <= 0: {pc(B['rich_netluck_le0'], 3)}")
print(f"  luck percentile of the rich: mean {B['rich_luck_pct_mean']:.1f}, median {B['rich_luck_pct_median']}; in the luckiest 10%: {pc(B['rich_luckiest10'], 2)}, "
      f"5%: {pc(B['rich_luckiest5'], 2)}, 1%: {pc(B['rich_luckiest1'], 2)}")
nl = B['net_luck']
people = np.array(nl['people'], float)
rich = np.array(nl['rich'], float)
off = nl['offset']
print(f"  smallest net luck of any rich person: {B['rich_min_netluck']}; share of people with at least that: {pc(B['share_people_with_min_netluck_or_more'], 2)}")
print('  net luck n: share of people | P(rich | n)')
for n in range(-2, 11):
    print(f"    {n:+d}: {pc(people[n + off] / people.sum(), 2)} | {pc(rich[n + off] / max(1, people[n + off]), 2)}")
for k in (3, 4, 5, 6):
    print(f"  net luck >= {k}: {pc(B[f'share_netluck_ge{k}'], 2)} of people; P(rich): top talent decile {pc(B[f'p_rich_top_decile_netluck_ge{k}'])}, middle deciles {pc(B[f'p_rich_middle_netluck_ge{k}'])}")
print(f"  top talent decile with net luck <= 3: P(rich) = {pc(B['p_rich_top_decile_netluck_le3'], 3)}")
dn = B['decile_x_netluck']
pe = np.array(dn['people'], float)
ri = np.array(dn['rich'], float)
cols = [(-15, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 15)]
print('  P(rich | talent decile (rows 1..10), net luck <=3, 4, 5, 6, 7, 8+):')
for d in range(10):
    print('    ', d + 1, [pc(ri[d, a + off:b + off + 1].sum() / max(1, pe[d, a + off:b + off + 1].sum())) for a, b in cols])

print('\n== duels: who ends richer (a tie counts as half)')
print(f"  top 1% of talent vs bottom 1%: {pc(B['duel_top1_bottom1'], 2)}; top 10% vs bottom 10%: {pc(B['duel_top10_bottom10'], 2)}; "
      f"top 1% vs middle: {pc(B['duel_top1_middle'], 2)}; top 10% vs middle: {pc(B['duel_top10_middle'], 2)}")

print('\n== replay: the same people, new luck')
rp = R['replay']['base']
print(f"  {rp['worlds']:,} worlds, twice: rich again {pc(rp['rich_again'], 2)} (chance alone: 1%); nobody rich again in {pc(rp['worlds_with_nobody_again'])} of worlds")
print(f"  where the first life's rich land: richest 10% {pc(rp['land_top10'])}, top half {pc(rp['land_top_half'])}, bottom half {pc(rp['land_bottom_half'])}, "
      f"median percentile {rp['land_median_pct']}; below start {pc(rp['below_start'])}, not above start {pc(rp['not_above_start'])}")
print('  rich again, by talent decile of the rich person:', [pc(x, 2) for x in rp['again_by_talent_decile']])

print('\n== one rule at a time (100,000 worlds each; replay 50,000 worlds twice)')
print(f"  {'':10s} top1%  top20%  gini | rich: talent pct, luck pct, luckiest5% | P(rich|top dec) | most talented rich | duel 1v1 | rich again")
for name in ('base100k', 'indep', 'mild', 'opp', 'pay', 'shield', 'start', 'all'):
    v = R[name]
    rr = R['replay'].get('base' if name == 'base100k' else name)
    print(f"  {name:10s} {pc(v['share_top1'])}  {pc(v['share_top20'])}  {v['gini']:.3f} | {v['rich_talent_pct_median']:3d} {v['rich_luck_pct_median']:3d} {pc(v['rich_luckiest5'])} | "
          f"{pc(v['p_rich_by_talent_decile'][9], 2)} | {pc(v['most_talented_rich'], 2)} | {pc(v['duel_top1_bottom1'])} | {pc(rr['rich_again'], 2) if rr else '-'}"
          f" | above start {pc(v['above_start'])}")
vi = R['indep']
li = np.array(vi['lucky_dist'], float)
print(f"  streaks: P(lucky events >= 5) with the map {pc((ld / ld.sum())[5:].sum(), 2)}, without {pc((li / li.sum())[5:].sum(), 2)}; "
      f"no events {pc(B['no_events'])} vs {pc(vi['no_events'])}; rich lucky events {B['rich_lucky_mean']:.2f} vs {vi['rich_lucky_mean']:.2f}")

print('\n== sweep: paycheck world, size of an event')
for s in R['sweep']:
    print(f"  x{s['factor']}: rich again {pc(s['rich_again'], 2)}, rich from top 10% talent {pc(s['rich_top10_talent'])}, top 1% talent {pc(s['rich_top1_talent'])}, "
          f"talent pct {s['rich_talent_pct_median']}, luck pct {s['rich_luck_pct_median']}, duel {pc(s['duel_top1_bottom1'])}, top1% hold {pc(s['share_top1'])}, "
          f"most talented rich {pc(s['most_talented_rich'])}")

print('\n== the world on screen (seed %d)' % S['seed'])
print('  selection stats over the first 1,000 worlds:', S['selection'])
b = S['top10'][0]
m = S['most_talented_info']
print(f"  richest: talent {b['talent']} (rank {b['talent_rank']}), capital {b['capital']:.0f} = {b['capital'] / 10:.0f}x, lucky {b['lucky']} (used {b['used']}), unlucky {b['unlucky']}")
print(f"  most talented: talent {m['talent']}, capital {m['capital']}, lucky {m['lucky']} (used {m['used']}), unlucky {m['unlucky']}, wealth rank {m['wealth_rank']}")
print(f"  top 1% hold {pc(S['share_top1'])}, top 20% {pc(S['share_top20'])}; below start {S['below_start']}, at {S['at_start']}, above {S['above_start']}; no events {S['no_events']}")
rpw = S['replay']
rc = np.array(rpw['rich_count'])
tal = np.array(S['tal'])
print(f"  replays: rich again in the next {len(rpw['again'])} lives: {rpw['again']}")
print(f"  in {S['lives']:,} lives: most talented rich {rpw['rich_count_most_talented']}, first life's richest {rpw['rich_count_richest']}, "
      f"max {rc.max()} (talent {tal[rc.argmax()]:.3f}), min {rc.min()} (talent {tal[rc.argmin()]:.3f}), never rich {int((rc == 0).sum())}, "
      f"median person {int(np.median(rc))}")
