"""Prints every number used in the video from resultados.json, as Markdown tables."""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, 'resultados.json')))
NAMES = ('beginner', 'intermediate', 'expert')
SIZE = {'beginner': '9 × 9 · 10', 'intermediate': '16 × 16 · 40', 'expert': '30 × 16 · 99'}
RULE = {'safe': 'classic (first click never a mine)', 'zero': 'modern (first click opens an area)'}


def p(x, d=1):
    return f'{100 * x:.{d}f}%'


def m(name, rule, strat='smart'):
    return R['main'][f'{name}/{rule}/{strat}']


print(f"total games: {R['total_games']:,}   seed {R['seed']}")
print('\n## Literature check (Tu et al. 2017, first click safe, in a corner, lowest-probability guessing)')
TU = {'8x8/10': (0.7844, 0.3606), '16x16/40': (0.7430, 0.3449), '30x16/99': (0.3556, 0.05097)}
print('| board | won, paper | won, mine | no guess, paper | no guess, mine |\n|---|---|---|---|---|')
for k, (w, n) in TU.items():
    s = R['check'][k]
    print(f"| {k} | {p(w, 2)} | {p(s['win'], 3)} | {p(n, 3)} | {p(s['no_guess'], 3)} |")

print('\n## First click (best cell by win rate, perfect logic + smart guessing)')
for name in NAMES:
    for rule in ('safe', 'zero'):
        h = R['heat'][f'{name}/{rule}']
        H = len(h['win']); W = len(h['win'][0])
        b = h['best']; bn = h['best_no_guess']
        print(f"{name:12s} {rule:4s} best ({b['r']},{b['c']}) win {p(b['win'], 2)}  no-guess {p(b['no_guess'], 2)} | best no-guess ({bn['r']},{bn['c']}) {p(bn['no_guess'], 2)}"
              f" | centre win {p(h['win'][H // 2][W // 2], 2)} no-guess {p(h['no_guess'][H // 2][W // 2], 2)} | corner win {p(h['win'][0][0], 2)}")

print('\n## Main runs: share of games with at least one forced guess')
print('| board | classic | modern |\n|---|---|---|')
for name in NAMES:
    print(f"| {name} {SIZE[name]} | {p(1 - m(name, 'safe')['no_guess'])} | {p(1 - m(name, 'zero')['no_guess'])} |")
print('\n## Main runs: details')
for name in NAMES:
    for rule in ('safe', 'zero'):
        s = m(name, rule)
        print(f"{name:12s} {rule:4s} games {s['games']:,} first {s['first']}  forced {p(1 - s['no_guess'], 3)}  no-guess {p(s['no_guess'], 3)}"
              f"  basic-only {p(s['basic_only'], 3)}  pattern-only {p(s['pattern_only'], 3)}")
        print(f"   stuck: pattern {p(s['stuck_pattern'])} full {p(s['stuck_full_logic'])} forced {p(s['stuck_forced'])}  counter-needed {p(s['safe_needed_counter'])}"
              f"  guesses/game {s['guesses_per_game']:.3f} per won {s['guesses_per_won_game']:.3f}  immediate {p(s['first_guess_immediate'])}")
        fh = np.array(s['frac_hist'], float); fh /= fh.sum()
        ph = np.array(s['pmin_hist'], float); ph /= ph.sum()
        print(f"   first tenth {p(fh[:5].sum())} last tenth {p(fh[-5:].sum())}  coin {p(s['coin_flips'])} two-square {p(s['coin_flips_endgame_2cells'])}"
              f"  risk 5-25%: {p(ph[5:25].sum())}  <5%: {p(ph[:5].sum())}  lost on coin {s['lost_on_coin']:,}")
        for st in ('smart', 'safest', 'frontier', 'random'):
            q = m(name, rule, st)
            mx = max(i for i, v in enumerate(q['guess_hist_won']) if v)
            print(f"      {st:8s} win {p(q['win'], 3)} ± {p(q['win_se'], 3)}  mean risk clicked {p(q['mean_p_chosen'])}  guesses/game {q['guesses_per_game']:.3f}  max guesses in a won game {mx}")

print('\n## First click can be a mine ("any" rule)')
for name in NAMES:
    s = R['any'][name]
    print(f"{name:12s} boom on click 1 {p(s['boom_first_click'], 2)}  win {p(s['win'], 2)} (predicted {p(s['predicted_win'], 2)})  no-guess {p(s['no_guess'], 2)} (predicted {p(s['predicted_no_guess'], 2)})")

print('\n## Density sweep (no-guess share)')
for k, pts in R.get('sweep', {}).items():
    print(k, ' '.join(f"{100 * q['density']:.1f}%:{100 * q['no_guess']:.1f}" for q in pts))
if 'counter_fix' in R:
    print('\n## Mine counter needed (fixed solver)')
    for k, s in R['counter_fix'].items():
        print(k, p(s['safe_needed_counter']))
