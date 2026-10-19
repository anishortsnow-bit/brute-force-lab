# Experiment 08: A Zombie Outbreak on 329 Million Americans

Code for the Brute Force Lab Halloween special *I Simulated a Zombie Outbreak on 329 Million Americans*.

This one is not a game. It is the zombie model of A. A. Alemi, M. Bierbaum, C. R. Myers and J. P. Sethna (Cornell), "You Can Run, You Can Hide: The Epidemiology and Statistical Mechanics of Zombies", *Physical Review E* 92, 052801 (2015), [arXiv:1503.01104](https://arxiv.org/abs/1503.01104), rebuilt from the paper, checked against its published numbers, run on the 2020 Census, and then run with one rule changed at a time.

**Everything here describes a model whose numbers come from zombie movies. None of it is a forecast of anything.**

## The model

Four states: human (S), exposed (E, bitten and about to turn), zombie (Z) and destroyed zombie (R).

| rate | value | what it means |
|---|---|---|
| bite, `beta` | 3.6e-3 per hour, per human-zombie pair in the same square | S + Z -> E + Z |
| kill, `kappa = alpha * beta` | `alpha` = 0.8 | S + Z -> S + R. The authors estimated 0.8 from *Night of the Living Dead* and *Shaun of the Dead* |
| turn, `nu` | 2 per hour (about 30 minutes) | E -> Z |
| step, `mu` | 0.0914 per hour, to each land neighbour | a zombie walking at 1 foot per second across 3 km squares |

All four values are the paper's Table I. (The text of the paper mentions a 30-minute half-life for turning, which would be 1.39 per hour; this code uses the table.)

- Inside a square, everyone can meet everyone (the square is well mixed).
- Between squares, only zombies move, to the 4 neighbouring land squares. Nobody flees or travels, as in the paper.
- Zombies do not enter the sea or leave the country, so islands with no land link are never reached.

## The people

The 2020 Census for the lower 48 states and DC: 329,260,619 people in 238,193 block groups.

- Each person is placed at a random spot inside their own block group (sub-cells of 375 m) and counted on squares of exactly 3 x 3 km in an equal-area projection (USA Contiguous Albers, spherical earth). The paper did the same with the 2010 census blocks.
- As in the paper, six iterations of binary closing add one person to empty land squares between occupied ones (55,240 people here).
- 974 x 1,539 squares, 873,134 of them land. The busiest square, in lower Manhattan, holds 299,457 people (the paper: 299,616).

The census files are not in this repository. Download them into `census/`:

- `CenPop2020_Mean_BG.txt`, population and center of each block group: https://www2.census.gov/geo/docs/reference/cenpop2020/blkgrp/CenPop2020_Mean_BG.txt
- `cb_2020_us_bg_500k.zip`, block group shapes: https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_bg_500k.zip

## How it runs

- Inside each square, the reactions are simulated event by event (exact Gillespie). When a square holds more than 200 humans and 200 zombies at once, they advance in short leaps (tau-leaping, at most 3% change per leap).
- Zombies step between squares every 6 minutes with exact binomial draws; where nobody is left to bite, every 30 minutes with the probability for that interval.
- `test_split.py` checks this shortcut against the fully exact simulation on real patches of the grid. Every comparison is within the noise. In lower Manhattan, over 4,000 runs, 19.95% of outbreaks are alive after 12 hours both ways.

## Checked against the paper

| | paper | here |
|---|---|---|
| 200 people, 1 zombie, `alpha` 0.6: zombies win | 40% | 39.96% (200,000 runs) |
| same room: zombies left when they win | 81 (formula) | 83 on average |
| one person per square: tipping point of `alpha` | 0.4373 | 0.4367 (L = 256, 512, 1024; 3,000 runs per point) |
| one random patient zero: outbreak still going at 7 / 28 days | 21.0% / 20.8% | 19.3% / 19.25% (4,000 runs) |
| highest risk after one month | northeastern Pennsylvania | Lehigh, Northampton, Carbon, Monroe, Pike (PA) in the top 12 of 984 counties |
| highest risk after one month, West | the San Joaquin Valley near Bakersfield | in California: Tulare, Kings, Kern, Fresno; Los Angeles 8th |
| one in a million infected: remote land after four months | parts of Montana and Nevada untouched | parts of Montana (54,000 to 721,000 people) and Nevada (5,000 to 22,000) never reached, in all 4 runs |

`run_all.py` stops if a check fails.

Where this version is slower than the paper: the first week. With one in a million infected, the paper says most people are zombies after a week; here 45% are. Possible reasons: 2020 block groups instead of 2010 blocks (more people spread out in the countryside), the turning rate from the table, and the equal-area grid.

## Results in the video

- **Patient zero** is destroyed before taking off in about 4 of 5 outbreaks. Each zombie bites `beta / kappa` = 1.25 people on average before it is destroyed, and 44% are destroyed before their first bite.
- **One zombie in lower Manhattan** (seed 2531, the first seed tried): 21,248 bitten and 6 zombies standing after 12 hours; the busiest square half gone at hour 16; 14.8 million bitten by the end of day 3. First contact: Allentown 7.6 days, Philadelphia 8.7, Boston 19.4, Washington 20.4, Chicago 78, St. Louis 100. After 120 days, 56% bitten and no square reached west of Iowa and Missouri.
- **One in a million at the start** (8 runs of 28 days, 4 of them to 120 days): 97.9% still human after one day, 54.8% after a week, half gone in 8.7 days, 13.7% still human after a month, 0.7% after four months (0.1% to 1.5%).
- **Riskiest counties after one month** (984 counties with 50,000+ people, 4,000 outbreaks from one random person): Lehigh PA, Northampton PA, Hunterdon NJ, Carbon PA, Warren NJ. Manhattan is 32nd, Philadelphia 35th. The differences are small: 20.2% of Lehigh's people overrun on average against 19.0% in Manhattan.
- **One rule at a time** (same starting people and seeds):
  - zombies 10 times faster: half gone in 2.9 days instead of 8.7; 0.08% still human after a month;
  - everyone hides (meets half as many people: `beta` and `kappa` both halved): half gone in 7.6 days, no help;
  - `alpha` 0.9: 52% and 60% still human after two weeks (41% with the movie value);
  - `alpha` 1.0: the outbreak dies out; 1.8% of patient zeros take off.

## Files

| file | what it is |
|---|---|
| `zsim.py` | the three models: one room, one person per square, the whole country (Python + Numba) |
| `grid.py` | the population grid from the census files; writes `out/grid.npz` and `out/grid.json` |
| `cities.py` | the reference cities (downtown coordinates) |
| `run_all.py` | every number in the video; writes `resultados.json`, `run_all.log` and the maps in `out/` |
| `extra_hide.py` | hiding and fast zombies with 8 starting draws (section C2 of `resultados.json`) |
| `test_split.py` | the stepped simulation against the fully exact one |
| `resultados.json`, `run_all.log`, `extra_hide.log` | the results and logs of the full run |

```
pip install numpy numba scipy
python grid.py
python run_all.py          # about 50 minutes on 16 threads; python run_all.py --quick for a small run
python extra_hide.py       # about 15 minutes
python test_split.py --dts 0.1
python test_split.py --city
```

The folders can be changed with the environment variables `ZOMBIE_CENSUS` and `ZOMBIE_OUT`. Seed 2031 (2020 for the placement of people).
