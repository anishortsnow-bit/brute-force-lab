"""The one game shown on screen. Writes samples.json.

It is one of the baseline games of run_all.py (same seed and seats, so it is one of the games that were counted):
the first one, in seed order, that looks typical: it ends, in 45 to 60 rounds (the median is 53), and a color group
was completed along the way.

Run: python samples.py
"""
import json
import os

import numpy as np

import mono

SEED = 2026
HERE = os.path.dirname(os.path.abspath(__file__))
NONE = np.full(40, -1, np.int64)
POL = np.array([mono.policy() for _ in range(4)])
opt = mono.options()
res = np.zeros(8, np.int64)
rec = np.zeros((400, 97), np.int64)
chosen = None
for n in range(0, 4000, 4):                       # multiples of 4: the seats are not rotated in the batch
    rec[:] = 0
    mono.play(SEED + n, 4, POL, NONE, opt, res, rec)
    if res[0] >= 0 and 45 <= res[1] <= 60 and res[2] >= 0 and res[5] <= 400:
        chosen = n
        break
turns = int(res[5])
out = dict(seed=SEED + chosen, game_index=chosen, winner=int(res[0]), rounds=int(res[1]), turns=turns,
           first_group=mono.GROUP_NAMES[int(res[2])], first_group_owner=int(res[3]), first_group_round=int(res[4]),
           player=[int(x) for x in rec[:turns, 0]], owner=rec[:turns, 1:41].tolist(), level=rec[:turns, 41:81].tolist(),
           pos=rec[:turns, 81:85].tolist(), cash=rec[:turns, 89:93].tolist())
json.dump(out, open(os.path.join(HERE, 'samples.json'), 'w'))
print({k: v for k, v in out.items() if not isinstance(v, list)})
last = rec[turns - 1]
print('final cash:', last[89:93].tolist(), '| lots of the winner:', int((last[1:41] == res[0]).sum()), '| buildings:', int((last[41:81] % 10).sum()))
