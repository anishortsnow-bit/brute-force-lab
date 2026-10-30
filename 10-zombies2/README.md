# Experiment 10: Can the US Army Stop One Zombie in Manhattan?

Code for the Brute Force Lab video *Can the US Army Stop One Zombie in Manhattan?*, part 2 of [experiment 08](../08-zombies).

Same base as before: the zombie model of A. A. Alemi, M. Bierbaum, C. R. Myers and J. P. Sethna (Cornell), "You Can Run, You Can Hide: The Epidemiology and Statistical Mechanics of Zombies", *Physical Review E* 92, 052801 (2015), [arXiv:1503.01104](https://arxiv.org/abs/1503.01104), on the 2020 Census grid of experiment 08. The paper has no police, no army, no guns, no differences between people, no learning and no terrain, so all of that is our own extrapolation. The choices are listed below, and the video marks them on screen as "OUR CHOICE".

**Everything here describes a model whose numbers come from zombie movies. None of it is a forecast of anything.**

## The model

The four states, the paper's Table I rates and the 873,134 squares of 3 x 3 km are those of experiment 08, and every run starts the same way: one person bitten in the busiest square of lower Manhattan. An outbreak "takes off" if it is still going after 3 days. With one class of people, flat ground and no soldiers, `zsim2.py` is the old model. Random numbers now come from xoshiro256**, one stream per run, so a seed always gives the same run.

Fighters. A class that wins a share `p` of its fights has bite rate `(1 - p) * beta * (1 + alpha)` and kill rate `p * beta * (1 + alpha)`, the same meetings per pair as the paper, so `p` = 4/9 is exactly the paper. The ladder is `p = 4/9 + 0.05 * L`. In the unequal runs each civilian is weak, average or strong with probability 1/3 and wins 24.4%, 44.4% or 64.4%.

Learning (our choice). Each zombie destroyed moves a person one step up the ladder, up to 94.4%. Officers and soldiers do not move.

Police (our choice). 787,565 state and local officers (BJS 2018) for 327,167,434 people, one for every 415.4, placed by a binomial draw: 792,989 on the grid. An officer is bitten like anyone when a zombie gets close, but destroys zombies `m` times as fast as a civilian (`kappa = m * 0.8 * beta`), for guns that reach from a distance. `m` is 10 or 100. The `close_combat` variant has a civilian's meetings and wins `0.8m / (1 + 0.8m)` of the fights. The tipping point is at about 104.8x.

The ring (our choice). On the ring day, every square already reached plus 5 more is the quarantine zone, and the next square around it is the ring, counted in steps on land so it follows the coast. That is 15 km past the farthest zombie, 1 square wide. Civilians on the ring move one square out. On day 8 the ring is about 175 squares, so 60,000 soldiers is about 343 per square. Soldiers come from outside, are not counted as humans, and turn when bitten like anyone. They fight at 10x or 100x; the 10x is borrowed from Witkowski & Blais (2013), who use it for the ending of *Shaun of the Dead*.

The crossing rule (our choice, added after review r1). A zombie leaving the ring has to cross the soldiers' square first. It gets through with probability `exp(-kill rate / step rate)`; otherwise it stays and fights.

The moving army (our choice, section E only, added after review r2). Every 24 hours the soldiers still standing are pooled and spread evenly along the ring again (`rebal_h`). This is the best case for mobility. Everywhere else soldiers stay where they were put.

Terrain. NOAA ETOPO1 and Tobler's hiking function. Time adds up, so the code averages pace, not speed: over 72 walking directions for each 1' sample, then over the samples of a square. A second version triples every slope, as an upper bound for what 1.85 km samples smooth away.

"Held" means no zombie and no bitten person alive outside the ring on day 120. Each army run uses the same seed as a run without the army and is identical until the ring forms, so "saved" is a paired difference.

## Checked

`test2.py`, results in `test2.json`, about 4 minutes:

| check | expected | here |
|---|---|---|
| 200 people, 1 zombie, `alpha` 0.6: zombies win | 40% (the paper) | 39.99% (200,000 runs) |
| one class, three pieces of the real grid: new engine against the engine of experiment 08 | same within noise | \|z\| ≤ 1.63 |
| same pieces: new engine against the fully exact simulation | same within noise | \|z\| ≤ 1.14 |
| three classes with identical rates against one class | identical | \|z\| ≤ 0.01 |
| unequal thirds, learning, police at 10x and terrain together, against the fully exact simulation | same within noise | \|z\| ≤ 0.64 |

Every run of `run2.py` also checks that humans + bitten + zombies + destroyed zombies add up to the population, and stops if not. `conferir_numeros.py` checks all 83 numbers spoken in the video against the files here: 83 of 83.

## Results in the video

b01 to b33 are the blocks of the narration.

| what | key | result |
|---|---|---|
| The first video again (b01) | `A_base` | 44.3% human after 120 days, Philadelphia on day 8.7 |
| Police at 10x (b08) | `B4_police_x10` | 47.3%, against 44.4% with none |
| Police at 100x (b09) | `B4_police_x100` | outbreak dead in all 240 runs (2 went past 3 days first) |
| Police at 100x, no reach (b10) | `B4_police_x100_close_combat` | wins 98.8% of fights, ends at 46.2% |
| 60,000 standing, 10x (b13) | `C`, `E.breach_*` | held 0 of 48, broke on day 8.95 to 9.15 with 97% standing, all bitten by day 120, 44.5% |
| 60,000 standing, 100x (b15) | `C`, `E.breach_*` | held 0 of 48, broke on day 10.1 to 17.6 with 95% standing; +15.8 million on day 28, +4.3 million on day 120 |
| 60,000 moving, 100x (b16) | `E.moving_*` | held 48 of 48, 91.5% human, 41% of soldiers lost |
| 60,000 moving, 10x (b16) | `E.moving_*` | held 0 of 48, all soldiers lost |
| Bigger armies, standing (b17) | `C` | 300,000 at 100x held 48; 1.3 million at 10x held 4; 3 million at 10x held 48 |
| 1.3 million moving, 10x (b17) | `E` | held 48 of 48 |
| Unequal fighters (b21 to b23) | `B1_unequal` | 50.2%; 0.978 zombies destroyed per bite instead of 0.800 |
| Learning (b24, b25) | `B2_learning` | 51.2%; 4.62 million with a win, 795,000 at the top |
| Unequal and learning (b26) | `B3_unequal_learning` | 66.3%, Philadelphia on day 14.1, St. Louis in none of 49 runs |
| Terrain (b27, b28) | `B5_*` | 44.3%, and 44.6% with slopes x3; Pittsburgh on day 34.2 and 35.3 (34.1 flat) |
| Everything at once (b29) | `D_all`, `extra_v2.json` | 96.0%, held 44 of 44, 1.0% of soldiers lost; 69.6% without the army |

The army's firepower in b14: 60,000 x 8 = 480,000 zombies, against 4.89 million alive on day 8. In b29 the soldiers face 4,770 instead.

Also in `resultados2.json` but not in the video: the ring at 100x on days 1, 2, 3, 5 and 14, and 300,000 soldiers at 10x.

## Limits

- At 10x the crossing rule stops being airtight below about 20 soldiers in a square (2 at 100x). 24 soldiers at 10x still let 15% through.
- The moving "48 of 48" needs both the crossing rule and the daily redistribution. Redistributing every 72 hours, the ring broke in 4 of 6 test runs; every 168 hours, in 6 of 6. Without the crossing rule it broke on day 9 in 6 of 6.
- At 10x moving does not help, even every 0.1 h: breaks on days 10.7 to 12.9, army gone by day 30.
- The ring margin (6, 15 or 30 km) changes the day it breaks, not the outcome.
- The moving 60,000 at 100x hold because zombies reach the line slowly, not because the firepower is enough. By arithmetic (4.8 million against about 5.6 million) the ring would fall near day 450. A 360-day test still held, with 10% of the army left.
- Police at 100x sits right at the tipping point. About 2 outbreaks in 240 should take off, with an 11% chance of none, so "all 240" is compatible with luck.
- "91% human" is everyone outside the zone. The 3.4 million inside are lost, and 3.3 to 3.7 million zombies are still in there on day 120. Only everything at once sees the outbreak die inside.
- A soldier ten times deadlier being worth ten soldiers is built into the model, since soldiers are bitten equally often at any strength. 300,000 at 100x against 3 million at 10x cannot disagree.
- The moving army also covers ring squares no zombie ever reaches.
- Zombies walk at random and do not follow people.
- One ring square of 166 (seed 10000) has no neighbour outside the ring, so about 520 civilians stay.
- The 787,565 officers leave out federal officers. The Katrina troops were spread over four states, some aboard ships.
- In de Mendonça et al. civilians fight too, and 47 per 1,000 is the minimum military in their model. In Ek et al. "humans win" includes losing the Helsinki region, and the 0.1% leak is the same along the whole border, not one weak spot.
- ETOPO1 includes the bottom of the Great Lakes and Death Valley is flattened to sea level. Neither matters at this scale.
- In four months the outbreak never gets near the Rockies, and at 3 km even the Appalachians average out to gentle slopes, so terrain never really got tested.

## Reviews

The logic went through five rounds of review before the narration was recorded. The reports are in `reviews/`, in Portuguese, and they quote the drafts of the narration they read, so some lines are not in the final video. In them, `analises/zombies2/` is this folder and `analises/zombies/` is `../08-zombies/`. Findings are graded QUEBRA (a sentence becomes false or misleading), ENFRAQUECE (it only partly holds) or NOTA (no risk).

| round | findings | what changed |
|---|---|---|
| r1 | 2 QUEBRA, 3 ENFRAQUECE, 10 NOTA | zombies left squares full of soldiers without a fight: crossing rule added, C and D run again |
| r2 | 1 QUEBRA, 4 ENFRAQUECE, 9 NOTA | a standing army falls at one square: section E and `rebal_h` added |
| r3 | 1 ENFRAQUECE, 6 NOTA | at 10x the line falls where the front hits it: narration fixed |
| r4 | 1 ENFRAQUECE, 5 NOTA | the moving army does not need less firepower: narration fixed |
| r5 | 5 NOTA | closed |

## How to run

```
pip install numpy numba scipy
```

Tested with Python 3.12, NumPy 2.5, Numba 0.67 and SciPy 1.17. You need the grid of experiment 08 and NOAA ETOPO1 at 1 arc-minute for the lower 48, saved as `data/etopo_conus.nc`: https://coastwatch.pfeg.noaa.gov/erddap/griddap/etopo180.nc?altitude%5B(24.0):1:(50.0)%5D%5B(-125.0):1:(-66.0)%5D

```
cd ../08-zombies && python grid.py && cd ../10-zombies2
python prep.py               # under a minute
python test2.py              # about 4 minutes
python run2.py               # 7 to 8 hours on 24 threads; --quick for a small run, --resume to keep finished sections
python extra_v2.py           # about 15 minutes; delete extra_v2.json first
python conferir_numeros.py
python vitrine_extra.py      # maps only
python vitrine_anel.py       # maps only
```

Folders can be changed with `ZOMBIE_GRID`, `ZOMBIE_ETOPO` and `ZOMBIE2_OUT`. Seeds: 10000 to 10599 for take-off (A, C, E), 20000 to 20239 for scenarios that change who fights (B, D), 20261028 for the strength thirds, 20261027 for the police.

## Files

| file | what it is |
|---|---|
| `zsim2.py` | the engine, plus the fully exact version for small grids |
| `prep.py` | terrain and police on the grid |
| `run2.py` | every number in the video, sections A to E |
| `extra_v2.py` | zombies alive on ring day and at the end, bites up to day 8 |
| `test2.py` | the engine checks |
| `conferir_numeros.py` | the 83 numbers of the narration against the results |
| `vitrine_extra.py`, `vitrine_anel.py` | the maps and ring scenes; the only numbers from them are the Denver distances (918 and 942 km) and walking speeds of b28 |
| `resultados2.json`, `extra_v2.json`, `test2.json`, `prep.json` | the results |
| `reviews/` | the five reports |

## Sources

- A. A. Alemi et al. (2015), [arXiv:1503.01104](https://arxiv.org/abs/1503.01104). The base model.
- C. Witkowski and B. Blais, "Bayesian Analysis of Epidemics - Zombies, Influenza, and other Diseases" (2013), [arXiv:1311.6376](https://arxiv.org/abs/1311.6376). The 0.8 and the *Shaun of the Dead* ending.
- K. Ek et al., "Zombie Epidemic — on Modeling the Effect of Interventions" (2024), [arXiv:2410.12502](https://arxiv.org/abs/2410.12502). Helsinki quarantine: 902 of 1,000; with a 0.1% leak, zombies win 658 of 1,000.
- J. P. A. de Mendonça et al., "Modeling our survival in a zombie apocalypse", *The Mathematical Intelligencer* (2019), [arXiv:1802.10443](https://arxiv.org/abs/1802.10443). 47 soldiers per 1,000.
- P. Munz et al., "When zombies attack!" (2009). The model Witkowski & Blais extend.
- W. Tobler, NCGIA Technical Report 93-1 (1993). The hiking function.
- C. Amante and B. W. Eakins, NOAA Technical Memorandum NESDIS NGDC-24 (2009). ETOPO1.
- Bureau of Justice Statistics, Census of State and Local Law Enforcement Agencies 2018 (NCJ 302187).
- Hurricane Katrina: US Department of Defense briefing, 6 September 2005: 41,500 National Guard and 17,500 active-duty troops.
- US active duty, about 1.3 million: Defense Manpower Data Center (1.34 million in October 2025).
