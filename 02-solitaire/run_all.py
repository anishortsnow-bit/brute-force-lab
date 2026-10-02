"""Experiment 02: Klondike, turn one, unlimited passes. Writes resultados.json and the arrays to ./output."""
import json, os, time
import numpy as np
import klondike as K

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output'); os.makedirs(OUT, exist_ok=True)
HERE = os.path.dirname(os.path.abspath(__file__))
R = {}
N = 100_000_000; CH = 10_000_000
for st, name in ((2, 'careful'), (1, 'greedy'), (0, 'random')):
    t = time.time(); wins = 0; hist = np.zeros(53, np.int64)
    for s0 in range(0, N, CH):
        w, f, _ = K.play_many(s0, CH, st, 24)
        wins += int(w.sum()); hist += np.bincount(f.astype(np.int64), minlength=53)
        if s0 == 0:
            np.save(os.path.join(OUT, f'play_{name}_won_first10M.npy'), w)
    R[name] = {'games': N, 'wins': wins, 'win_rate': wins / N, 'foundation_hist': hist.tolist(), 'seconds': round(time.time() - t)}
    print(name, R[name]['win_rate'], R[name]['seconds'], flush=True)
    json.dump(R, open(os.path.join(HERE, 'resultados.json'), 'w'), indent=1)

t = time.time(); zero = 0; mn = 10**9; hist = np.zeros(80, np.int64)
for s0 in range(0, N, CH):
    lg = K.legal_at_start(s0, CH); zero += int((lg == 0).sum()); hist += np.bincount(lg.astype(np.int64), minlength=80)[:80]
R['legal_at_start'] = {'deals': N, 'no_move': zero, 'hist': hist.tolist()}
print('no-move deals', zero, flush=True)
json.dump(R, open(os.path.join(HERE, 'resultados.json'), 'w'), indent=1)

ND = 20_000
t = time.time(); r, nd = K.solve_many(0, ND, 5_000_000, 24, 23)
p1 = {'won': int((r == 1).sum()), 'lost': int((r == 0).sum()), 'unknown': int((r == -1).sum()), 'seconds': round(time.time() - t)}
print('pass1', p1, flush=True)
ids = np.nonzero(r == -1)[0].astype(np.int64)
t = time.time(); r2, n2 = K.solve_ids(ids, 20_000_000, 12, 25)
r[ids] = r2; nd[ids] = n2
p2 = {'retried': int(ids.size), 'won': int((r2 == 1).sum()), 'lost': int((r2 == 0).sum()), 'unknown': int((r2 == -1).sum()), 'seconds': round(time.time() - t)}
print('pass2', p2, flush=True)
np.save(os.path.join(OUT, 'solver_res_20k.npy'), r); np.save(os.path.join(OUT, 'solver_nodes_20k.npy'), nd)
cw = np.load(os.path.join(OUT, 'play_careful_won_first10M.npy'))[:ND]
R['solver'] = {'deals': ND, 'pass1_5M': p1, 'pass2_20M': p2,
               'won': int((r == 1).sum()), 'lost': int((r == 0).sum()), 'unknown': int((r == -1).sum()),
               'careful_won_among_solver_won': int((cw & (r == 1)).sum()), 'careful_won_total_in_sample': int(cw.sum())}
json.dump(R, open(os.path.join(HERE, 'resultados.json'), 'w'), indent=1)
print('done', R['solver'], flush=True)
