"""Checks every number spoken in the video (narration blocks b01 to b33) against resultados2.json, extra_v2.json
(zombies alive on ring day and at the end, 29/10) and prep.json / prep.npz / the showcase maps (in ZOMBIE2_OUT, as in
run2.py) for the terrain and the police. Run after any change to the analysis: python conferir_numeros.py (exit code 1
if anything is off)."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
AN = HERE
OUT = os.getenv('ZOMBIE2_OUT', os.path.join(HERE, 'out'))
d = json.load(open(os.path.join(AN, 'resultados2.json'), encoding='utf-8'))
X = json.load(open(os.path.join(AN, 'extra_v2.json'), encoding='utf-8'))
prep = json.load(open(os.path.join(OUT, 'prep.json'), encoding='utf-8'))
PZ = np.load(os.path.join(OUT, 'prep.npz'))
CITY = list(PZ['city_names'])
A, B0, C = d['A_base'], d['B0_base_same_seeds'], d['C']
TOTAL = d['meta']['total_people']
rows = []


def ok(label, got, want, tol=0.5):
    rows.append((label, got, want, abs(got - want) <= tol))


def pct(x):
    return 100 * x


def held(v):
    return round(v['held_share'] * v['took_off'])


def city(name):
    return int(PZ['city_idx'][CITY.index(name)])


# b01: more than half bitten after four months (no army, same as video 08)
ok('b01 humans day 120, no army (%)', pct(A['humans_day120']), 44.3, 0.1)
ok('b01 more than half bitten', float(A['humans_day120'] < 0.5), 1, 0)
# b04: four wins out of nine
ok('b04 win share x9', 9 * d['B1_unequal']['classes']['win_share'][1], 4, 0.01)
# b07: one officer per 415 people
ok('b07 people per officer', prep['people_per_officer'], 415, 0.5)
# b08: police x10
P10 = d['B4_police_x10']
ok('b08 police x10 humans day 120 (%)', pct(P10['humans_day120']), 47, 0.5)
ok('b08 no police humans day 120 (%)', pct(B0['humans_day120']), 44, 0.5)
ok('b08 take-off about the same (x10 vs none, of 240)', P10['took_off'] - B0['took_off'], 0, 5)
# b09: police x100, all 240 runs die out; tipping point near 100
P100 = d['B4_police_x100']
ok('b09 police x100 runs', P100['runs'], 240, 0)
ok('b09 police x100 alive at end', P100.get('alive_at_end', 0), 0, 0)
ok('b09 tipping point (x)', d['meta']['police_threshold_x'], 100, 6)
# b10: close combat, 99 of 100, 46%
PC = d['B4_police_x100_close_combat']
ok('b10 close-combat win share (%)', pct(PC['classes']['win_share'][1]), 99, 0.5)
ok('b10 close-combat humans day 120 (%)', pct(PC['humans_day120']), 46, 0.5)
# b12: about 340 soldiers per 3 km on day 8; crossing a square takes about three hours on foot
K8 = C['x10_day8_soldiers60000']
ok('b12 soldiers per ring cell, day 8', d['meta']['soldiers'] / K8['ring_cells_median'], 340, 10)
ok('b12 ring margin (km)', d['meta']['ring_margin_km'], 15, 0)
ok('b12 hours to cross a square (1 / step rate)', 1 / PZ['mu_flat'][city('New York, NY')].sum(), 3, 0.3)
ok('b12 crossing rule used in the run', float('ring_crossing' in d['meta']), 1, 0)
E = d['E']
EB10, EB100 = E['breach_x10_day8_soldiers60000'], E['breach_x100_day8_soldiers60000']
# b13: x10, 8 of 9; broke about a day later at one square with almost the whole army standing; army bitten; country same
ok('b13 x10 win share x9', 9 * (10 * 0.8) / (1 + 10 * 0.8), 8, 0.01)
ok('b13 x10 day 8 held', held(K8), 0, 0)
ok('b13 x10 day 8 runs', K8['took_off'], 48, 0)
ok('b13 x10: breach, days after the ring (median), about one', EB10['breach_day_median'] - 8, 1, 0.25)
ok('b13 x10: every run breached within 1.2 days', EB10['breach_day_max'] - 8, 1, 0.2)
ok('b13 x10: army standing at the breach (%, median), almost all', pct(EB10['standing_at_breach_median']), 97, 1)
ok('b13 x10: army standing at the breach (%, lowest)', pct(EB10['standing_at_breach_min']), 94, 1)
ok('b13 x10 day 8 soldiers bitten by day 120 (%)', pct(K8['soldiers_lost_share']), 100, 0.5)
ok('b13 x10 day 8 humans day 120 vs none (pts)', pct(K8['humans_day120'] - A['humans_day120']), 0, 0.5)
# b14: one soldier per eight zombies destroyed; 60,000 x 8 = about half a million; zombies alive on day 8
ok('b14 kills per soldier lost at x10', 10 * 0.8, 8, 0)
ok('b14 zombies 60,000 soldiers can take out (millions)', 60000 * 10 * 0.8 / 1e6, 0.5, 0.05)
Z8 = X['x100_day8_300k']['zombies_ringday_median']
ok('b14 zombies alive on day 8, before the ring (millions, median of 48), almost five', Z8 / 1e6, 4.85, 0.1)
# b15: x100, 80 of 81, ~16 million in month one, broke in all 48 at one square with 95% standing, head start mostly gone
H8 = C['x100_day8_soldiers60000']
ok('b15 x100 win share x81', 81 * (100 * 0.8) / (1 + 100 * 0.8), 80, 0.01)
ok('b15 x100 day 8 saved day 28 (millions)', H8['saved_vs_no_army_day28'] / 1e6, 16, 0.5)
ok('b15 x100 day 8 held', held(H8), 0, 0)
ok('b15 x100: army standing at the breach (%, median)', pct(EB100['standing_at_breach_median']), 95, 0.5)
ok('b15 head start mostly gone (day 120 / day 28)', H8['saved_vs_no_army_day120'] / H8['saved_vs_no_army_day28'], 0, 0.5)
# b16: moving army: 60k x100 holds 48 of 48, 91% human; x10 breaks within about a day, army wiped out
M100, M10 = E['moving_x100_day8_soldiers60000'], E['moving_x10_day8_soldiers60000']
ok('b16 moving 60k x100 held', held(M100), 48, 0)
ok('b16 moving 60k x100 humans day 120 (%)', pct(M100['humans_day120']), 91, 0.6)
ok('b16 moving 60k x10 held', held(M10), 0, 0)
ok('b16 moving 60k x10: breach, days after the ring (median)', M10['breach_day_median'] - 8, 1, 0.25)
ok('b16 moving 60k x10 soldiers bitten (%)', pct(M10['soldiers_lost_share']), 100, 0.5)
ok('b16 moving rule used', float('moving_army' in d['meta']), 1, 0)
# b17: standing still: 300k x100 -> 48 of 48; 1.3M x10 -> 4; 3M x10 -> 48; moving 1.3M x10 -> 48
S300 = C['x100_day8_soldiers300000']
M3 = C['x10_day8_soldiers3000000']
ok('b17 300k x100 held', held(S300), 48, 0)
ok('b17 1.3M x10 held', held(C['x10_day8_soldiers1300000']), 4, 0)
ok('b17 3M x10 held', held(M3), 48, 0)
ok('b17 moving 1.3M x10 held', held(E['moving_x10_day8_soldiers1300000']), 48, 0)
# b18: 300k x100 did the same job as 3M x10; moving needs less (60k x100 and 1.3M x10 hold only when moving)
ok('b18 same firepower (soldiers x deadliness)', 300000 * 100 / (3000000 * 10), 1, 0)
ok('b18 humans day 120, 300k x100 vs 3M x10 (pts)', pct(S300['humans_day120'] - M3['humans_day120']), 0, 0.3)
ok('b18 soldiers lost, 300k x100 vs 3M x10 (pts)', pct(S300['soldiers_lost_share'] - M3['soldiers_lost_share']), 0, 1.5)
# b21-b23: unequal fighters
U = d['B1_unequal']
ws = U['classes']['win_share']
ok('b21 weakest third wins (%), about 1 in 4', pct(ws[0]), 25, 1)
ok('b21 strongest third wins (%), almost 2 in 3', pct(ws[2]), 64.5, 2.2)
ok('b22 bitten weak / strong', U['bitten_weak_over_strong'], 1, 0.01)
ok('b22 strong: zombies destroyed before bitten', ws[2] / (1 - ws[2]), 1.8, 0.05)
ok('b22 weakest: zombies destroyed before bitten', ws[0] / (1 - ws[0]), 0.33, 0.02)
ok('b23 destroyed per bite', U['destroyed_per_bite'], 0.98, 0.005)
ok('b23 humans day 120 (%)', pct(U['humans_day120']), 50, 0.5)
# b24-b26: learning
L = d['B2_learning']
ok('b24 top win share (%)', pct(L['classes']['win_share'][-1]), 94, 0.5)
ok('b24 step (points)', pct(L['classes']['win_share'][1] - L['classes']['win_share'][0]), 5, 0.01)
ok('b25 veterans, at least one kill (millions)', L['veterans_1plus'] / 1e6, 4.6, 0.05)
ok('b25 maxed out (thousands)', L['veterans_top'] / 1e3, 790, 15)
ok('b25 learning alone humans day 120 (%)', pct(L['humans_day120']), 51, 0.5)
UL = d['B3_unequal_learning']
ok('b26 humans day 120 (%), two in three', pct(UL['humans_day120']), 66.7, 0.5)
ok('b26 never loses half', UL['half_lost_share'], 0, 0)
ok('b26 Philadelphia first zombie (day)', UL['first_contact']['Philadelphia, PA']['median_day_of_reached'], 14, 0.5)
ok('b26 Philadelphia, no army (day)', A['first_contact']['Philadelphia, PA']['median_day_of_reached'], 9, 0.5)
ok('b26 St. Louis reached share', UL['first_contact']['St. Louis, MO']['reached_share'], 0, 0)
# b27-b28: terrain
T1, T3 = d['B5_terrain'], d['B5_terrain_slopes_x3']
ok('b27 Pittsburgh, terrain (weeks)', T1['first_contact']['Pittsburgh, PA']['median_day_of_reached'] / 7, 5, 0.25)
ok('b27 Pittsburgh, slopes x3 (weeks)', T3['first_contact']['Pittsburgh, PA']['median_day_of_reached'] / 7, 5, 0.25)
ok('b27 terrain humans day 120 (%)', pct(T1['humans_day120']), 44, 0.5)
ok('b27 slopes x3 humans day 120 (%)', pct(T3['humans_day120']), 44, 0.7)
# b28: nowhere near the Rockies (showcase maps, no army and slopes x3), and gentle slopes where the outbreak went
den = PZ['xs'][city('Denver, CO')]
for f in ('run_base.npz', 'run_B5_terrain_slopes_x3.npz'):
    reached = np.isfinite(np.load(os.path.join(OUT, f))['t_reach'])
    ok(f'b28 {f}: farthest west reached in 4 months, km east of Denver (900+)', (PZ['xs'][reached].min() - den) * 3, 950, 50)
    ok(f'b28 {f}: slopes x3, walking speed where it went, mean (%)', pct(PZ['walk3'][reached].mean()), 97.5, 0.5)
    ok(f'b28 {f}: real slopes, 95% of the squares it went through at 99%+ speed', pct(np.percentile(PZ['walk'][reached], 5)), 99, 0.5)
# b29: everything together: 96%, held in all 44, outbreak dead inside, about 1 soldier in 100; no army 70%
D = d['D_all']
DN = d['D_no_army']
ok('b29 humans day 120 (%)', pct(D['humans_day120']), 96, 0.5)
ok('b29 runs that took off', D['took_off'], 44, 0)
ok('b29 held', held(D), 44, 0)
ok('b29 outbreak dead at the end (share of runs alive)', D['alive_at_end'], 0, 0)
D8a, D8b = X['day8_all'], X['day8_base']
ok('b29 bitten by day 8, everything together (millions, median)', D8a['bites_median'] / 1e6, 11.5, 0.25)
ok('b29 bitten by day 8: everything together / last time, less than half', D8a['bites_median'] / D8b['bites_median'], 0.47, 0.03)
ok('b29 zombies destroyed per bite by day 8, almost one', D8a['destroyed_per_bite'], 1, 0.01)
ok('b29 same people, no army: humans day 120 (%)', pct(DN['humans_day120']), 70, 0.5)
ok('b29 same seeds with and without the army', DN['took_off'], D['took_off'], 0)
ok('b29 zombies alive on day 8, everything together (thousands, median), about five', X['D_all']['zombies_ringday_median'] / 1e3, 5, 0.5)

bad = [r for r in rows if not r[3]]
for label, got, want, good in rows:
    print('OK  ' if good else 'FAIL', label, round(float(got), 4), 'vs', want)
print(f'{len(rows) - len(bad)} of {len(rows)} numbers match')
sys.exit(1 if bad else 0)
