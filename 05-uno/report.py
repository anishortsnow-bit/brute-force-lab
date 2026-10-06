"""Every number the video quotes, straight from resultados.json and draws.json."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, 'resultados.json')))
DR = json.load(open(os.path.join(HERE, 'draws.json'))) if os.path.exists(os.path.join(HERE, 'draws.json')) else {}


def g(k, f='p'):
    return R[k][f] if k in R else None


def pc(x, d=2):
    return '-' if x is None else f'{100 * x:.{d}f}%'


def luck(p, n=2):
    return None if p is None else 1 - (p - 1 / n) / (1 - 1 / n)


total = sum(v['n'] * (v.get('games_mean', 1)) for v in R.values() if isinstance(v, dict) and 'n' in v)
print(f'total games: {int(total):,}')
print('tic-tac-toe: best v random', pc(R['ttt']['p']), 'luck', pc(1 - R['ttt']['skill']))
print('Sidajaya forward (random tie)', pc(g('sida_forward_randomtie')), 'range',
      pc(min(g(f'sida_forward_tie{k}') for k in range(5))), pc(max(g(f'sida_forward_tie{k}') for k in range(5))),
      ' reverse', pc(g('sida_reverse')))
print('random v random', pc(g('random_random')), 'first player', pc(R['random_random']['p_first']))
print()
print('== two players, official rules (win rate, luck = 2(1 - p))')
for k in ('casual_random', 'expert_random', 'search_random', 'cheat_random', 'expert_casual', 'search_casual',
          'search_expert', 'cheat_expert', 'cheat_search', 'search_nodraw_random'):
    if k in R:
        d = R[k]
        print(f'  {k:22s} {pc(d["p"])} +- {pc(d["se"])}  luck {pc(luck(d["p"]))}  n={d["n"]:,}  first {pc(d["p_first"])}'
              f'  second {pc(d["p_second"])}')
print('K curve:', ', '.join(f'{K}: {pc(g(f"k{K}_search_random"))}' for K in (8, 32, 128, 512, 8192)),
      f'2048: {pc(g("search_random"))}')
d = R['search_random']
print(f'search turns: nothing {pc(d["forced"] / d["turns"])}, one {pc(d["one"] / d["turns"])}, choice '
      f'{pc(d["multi"] / d["turns"])}; deliberate draws per game {d["voluntary"] / d["n"]:.2f}; decisions with options '
      f'per game {d["sdec"] / d["n"]:.1f}; kept the hand-made pick {pc(d["sagree"] / d["sdec"])}')
if 'expert_expert' in R:
    e = R['expert_expert']
    print(f'expert v expert: first player {pc(e["p_first"])}; turns nothing {pc(e["forced"] / e["turns"])}, one '
          f'{pc(e["one"] / e["turns"])}, choice {pc(e["multi"] / e["turns"])}')
    print('  win rate by wild cards dealt (mine - yours):',
          ', '.join(f'{k}: {pc(v[0] / v[1], 1)} ({v[1]:,})' for k, v in sorted(e['by_wild_diff'].items(), key=lambda x: int(x[0]))))
    print('  by wild cards held:', ', '.join(f'{k}: {pc(v[0] / v[1], 1)}' for k, v in sorted(e['by_wilds'].items(), key=lambda x: int(x[0]))))
    print('  by Wild Draw Fours held:', ', '.join(f'{k}: {pc(v[0] / v[1], 1)}' for k, v in sorted(e['by_wd4'].items(), key=lambda x: int(x[0]))))
print()
print('== matches to 500')
for k in ('m_casual_random', 'm_expert_random', 'm_expert_casual', 'm_search_random', 'm_search_casual'):
    if k in R:
        d = R[k]
        print(f'  {k:18s} match {pc(d["p"])} +- {pc(d["se"])}  game {pc(d["game_p"])}  games/match {d["games_mean"]:.2f}')
print('== players (luck = 1 - (p - 1/n)/(1 - 1/n))')
for n in (3, 4, 5, 6, 8, 10):
    row = []
    for who in ('search', 'expert', 'casual', 'random'):
        k = f'p{n}_{who}_random' if who != 'random' else f'p{n}_random'
        if k in R:
            row.append(f'{who} {pc(R[k]["p"])} (luck {pc(luck(R[k]["p"], n), 1)})')
    print(f'  {n} players: ' + '; '.join(row))
print('== house rules and Crazy Eights (luck)')
for k in ('stack', 'drawuntil', 'stack_drawuntil', 'c8'):
    row = []
    for who in ('search', 'expert', 'casual'):
        kk = f'{k}_{who}_random'
        if kk in R:
            row.append(f'{who} {pc(R[kk]["p"])} luck {pc(luck(R[kk]["p"]), 1)}')
    print(f'  {k}: ' + '; '.join(row))
print('== ablation (expert with one habit off, vs random; full expert', pc(g('expert_random')), ')')
for f in ('holdwild', 'color', 'attack', 'chain', 'void', 'points'):
    k = f'abl_{f}_random'
    if k in R:
        print(f'  without {f:9s} {pc(R[k]["p"])}  ({100 * (R[k]["p"] - g("expert_random")):+.2f} points)')
if DR:
    wo = sum(v for k, v in DR['by_alternative'].items() if k.startswith('only wild'))
    pl = DR['wilds'].get('wild played, nothing else', 0)
    print(f'deliberate draws ({DR["games"]} games): {DR["deliberate_draws"]}, by alternative {DR["by_alternative"]}')
    print(f'  wild-only turns: drew {wo}, played the wild {pl} -> drew {pc(wo / (wo + pl))}')
    print('  after drawing a playable card:', DR['after_draw'])
