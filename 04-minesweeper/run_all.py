"""Experiment 04: How often does Minesweeper force you to guess?

1. Checks the engine against published numbers (Tu, Li, Chen, Zu & Gu, AAAI-17 workshop, Table 2:
   first click safe and in a corner, lowest-probability guessing, and "give up when forced to guess").
2. First click: every cell of the board (one quarter, by symmetry), under the classic rule (first click is
   never a mine) and the modern rule (first click always opens an area).
3. Main runs on Beginner, Intermediate and Expert, from the best first cell for each rule, with four ways to
   guess, on the same boards (same seed). The "any" rule (first click can be a mine) follows exactly from the
   classic one and is checked by simulation.
4. Mine density sweep on the three board sizes.
Writes resultados.json and appends to run_all.log. Rerunning skips the parts already in resultados.json.
"""
import json, os, sys, time
import numpy as np
import runner as R

HERE = os.path.dirname(os.path.abspath(__file__))
OUTF = os.path.join(HERE, 'resultados.json')
SEED = 2026
BOARDS = {'beginner': (9, 9, 10), 'intermediate': (16, 16, 40), 'expert': (30, 16, 99)}
STRATS = ('smart', 'safest', 'frontier', 'random')
N_CHECK = 10_000_000
N_HEAT = {'beginner': 400_000, 'intermediate': 300_000, 'expert': 200_000}
N_MAIN = {'beginner': 25_000_000, 'intermediate': 25_000_000, 'expert': 25_000_000}
N_SWEEP = 1_000_000

RES = json.load(open(OUTF)) if os.path.exists(OUTF) else {}
LOGF = open(os.path.join(HERE, 'run_all.log'), 'a', encoding='utf-8')


def log(s):
    print(s, flush=True)
    LOGF.write(s + '\n'); LOGF.flush()


def save():
    games = 0
    for part in ('check', 'heat', 'main', 'any', 'sweep', 'counter_fix'):
        games += _count(RES.get(part, {}))
    RES['total_games'] = games
    RES['seed'] = SEED
    json.dump(RES, open(OUTF + '.tmp', 'w'), indent=0)
    os.replace(OUTF + '.tmp', OUTF)


def _count(d):
    if isinstance(d, dict):
        if 'games' in d and 'win' in d:
            return d['games']
        return sum(_count(v) for k, v in d.items() if k not in ('best', 'best_no_guess'))
    if isinstance(d, list):
        return sum(_count(v) for v in d)
    return 0


log(f'===== run_all {time.strftime("%Y-%m-%d %H:%M")}')

# ------------------------------------------------------------------ 1. literature check
TU = {'8x8/10': (8, 8, 10, 0.7844, 0.3606), '16x16/40': (16, 16, 40, 0.7430, 0.3449), '30x16/99': (30, 16, 99, 0.3556, 0.05097)}
RES.setdefault('check', {})
for key, (W, H, M, p_win, p_ng) in TU.items():
    if key not in RES['check']:
        RES['check'][key] = R.run(W, H, M, 'safe', 0, 'safest', N_CHECK, seed=SEED + 1, log=log)
        save()
    s = RES['check'][key]
    log(f'  Tu et al. {key}: win {100 * p_win:.2f}% (mine {100 * s["win"]:.3f}%), no guess {100 * p_ng:.3f}% (mine {100 * s["no_guess"]:.3f}%)')
    assert abs(s['no_guess'] - p_ng) < 0.0012, key          # strategy-free number: must match closely
    assert abs(s['win'] - p_win) < 0.003, key               # depends on tie-breaks: close is enough
    assert s['capped_games'] == 0

# ------------------------------------------------------------------ 2. first click, every cell
RES.setdefault('heat', {})
for name, (W, H, M) in BOARDS.items():
    for rule in ('safe', 'zero'):
        key = f'{name}/{rule}'
        if key in RES['heat']:
            continue
        qh, qw = (H + 1) // 2, (W + 1) // 2
        grid_w = np.zeros((H, W)); grid_ng = np.zeros((H, W)); grid_g = np.zeros((H, W))
        cells = []
        for r in range(qh):
            for c in range(qw):
                s = R.run(W, H, M, rule, R.cell(W, H, r, c), 'smart', N_HEAT[name], seed=SEED + 2)
                cells.append({'r': r, 'c': c, 'win': s['win'], 'no_guess': s['no_guess'], 'guesses': s['guesses_per_game'],
                              'games': s['games']})
                for rr in (r, H - 1 - r):
                    for cc in (c, W - 1 - c):
                        grid_w[rr, cc] = s['win']; grid_ng[rr, cc] = s['no_guess']; grid_g[rr, cc] = s['guesses_per_game']
        best = max(cells, key=lambda d: d['win'])
        best_ng = max(cells, key=lambda d: d['no_guess'])
        RES['heat'][key] = {'cells': cells, 'win': grid_w.tolist(), 'no_guess': grid_ng.tolist(), 'guesses': grid_g.tolist(),
                            'best': best, 'best_no_guess': best_ng}
        log(f'heat {key}: best win at ({best["r"]},{best["c"]}) {100 * best["win"]:.2f}%, best no-guess at '
            f'({best_ng["r"]},{best_ng["c"]}) {100 * best_ng["no_guess"]:.2f}%, centre {100 * grid_w[H // 2, W // 2]:.2f}%')
        save()

# ------------------------------------------------------------------ 3. main runs
RES.setdefault('main', {})
for name, (W, H, M) in BOARDS.items():
    for rule in ('safe', 'zero'):
        b = RES['heat'][f'{name}/{rule}']['best']
        first = R.cell(W, H, b['r'], b['c'])
        for strat in STRATS:
            key = f'{name}/{rule}/{strat}'
            if key in RES['main']:
                continue
            RES['main'][key] = R.run(W, H, M, rule, first, strat, N_MAIN[name], seed=SEED, detail=True, log=log)
            save()
        # same boards, so the no-guess share cannot depend on how you guess
        ng = [RES['main'][f'{name}/{rule}/{s}']['no_guess'] for s in STRATS]
        assert max(ng) - min(ng) < 1e-12, ng

# the "any" rule: first click can be a mine. Given the first cell is safe, the other mines are uniform: the classic rule.
RES.setdefault('any', {})
for name, (W, H, M) in BOARDS.items():
    if name in RES['any']:
        continue
    b = RES['heat'][f'{name}/safe']['best']
    first = R.cell(W, H, b['r'], b['c'])
    s = R.run(W, H, M, 'any', first, 'smart', 4_000_000, seed=SEED + 3, log=log)
    m = RES['main'][f'{name}/safe/smart']
    pred_w = (1 - M / (W * H)) * m['win']; pred_ng = (1 - M / (W * H)) * m['no_guess']
    assert abs(s['win'] - pred_w) < 5 * s['win_se'] + 1e-4 and abs(s['no_guess'] - pred_ng) < 5 * s['no_guess_se'] + 1e-4
    assert abs(s['boom_first_click'] - M / (W * H)) < 0.002
    s['predicted_win'] = pred_w; s['predicted_no_guess'] = pred_ng
    RES['any'][name] = s
    save()

# ------------------------------------------------------------------ 4. mine density
SWEEP = {'9x9': (9, 9, list(range(4, 25))), '16x16': (16, 16, sorted(set(range(10, 78, 3)) | {40})),
         '30x16': (30, 16, sorted(set(range(20, 146, 5)) | {99}))}
RES.setdefault('sweep', {})
for key, (W, H, Ms) in SWEEP.items():
    for rule in ('safe', 'zero'):
        k2 = f'{key}/{rule}'
        if k2 in RES['sweep']:
            continue
        name = {'9x9': 'beginner', '16x16': 'intermediate', '30x16': 'expert'}[key]
        b = RES['heat'][f'{name}/{rule}']['best']
        first = R.cell(W, H, b['r'], b['c'])
        pts = []
        for M in Ms:
            s = R.run(W, H, M, rule, first, 'smart', N_SWEEP, seed=SEED + 4)
            pts.append({'M': M, 'density': M / (W * H), 'win': s['win'], 'no_guess': s['no_guess'], 'guesses': s['guesses_per_game'],
                        'games': s['games']})
        RES['sweep'][k2] = pts
        log(f'sweep {k2}: ' + ', '.join(f'{p["M"]}:{100 * p["no_guess"]:.1f}' for p in pts))
        save()

# ------------------------------------------------------------------ 5. how often a safe square needs the mine counter
# The first version of the solver pruned arrangements with more mines than are left before checking whether a cell was
# safe from the numbers alone, so it undercounted this one statistic (nothing else changes: same deductions, same games).
# It is recomputed here with the fixed solver (test_solver.py checks it against brute force).
RES.setdefault('counter_fix', {})
for name, (W, H, M) in BOARDS.items():
    for rule in ('safe', 'zero'):
        key = f'{name}/{rule}/smart'
        if key in RES['counter_fix']:
            continue
        b = RES['heat'][f'{name}/{rule}']['best']
        s = R.run(W, H, M, rule, R.cell(W, H, b['r'], b['c']), 'smart', 2_000_000, seed=SEED + 5, log=log)
        RES['counter_fix'][key] = s
        RES['main'][key]['safe_needed_counter'] = s['safe_needed_counter']
        RES['main'][key]['safe_needed_counter_source'] = 'counter_fix (2,000,000 games, fixed solver)'
        save()

save()
log(f'total games: {RES["total_games"]:,}')
