# Experiment 06: How Much Luck Do You Need to Get Rich?

Code for the Brute Force Lab video *I Simulated How Much Luck You Need to Get Rich*.

This one is not a game. It is the "Talent vs Luck" model of A. Pluchino, A. E. Biondo and A. Rapisarda (*Advances in Complex Systems* 21, 1850014, 2018; [arXiv:1802.07068](https://arxiv.org/abs/1802.07068)), rebuilt from the paper, checked against its published numbers, and then run with one rule changed at a time.

**Everything here describes the rules of a model. None of it is a measurement of real careers.**

## The model

- 1,000 people at fixed random positions in a 201 x 201 world that wraps around.
- Talent is drawn from a normal distribution (mean 0.6, standard deviation 0.1, cut to the range 0 to 1) and never changes.
- Everyone starts with 10 units of capital.
- 500 event points, half lucky and half unlucky, each move 2 patches in a random direction every step.
- An event point within 1 patch of a person is an event for that person:
  - **lucky:** the capital doubles, with a probability equal to the person's talent;
  - **unlucky:** the capital is cut in half.
- 80 steps of six months: a 40-year career.

Two details the paper leaves open, and what this code does:

- Two event points of the same kind over the same person in the same step count as one event.
- Positions are continuous (not snapped to patch centers).

## Definitions used in the video

- **Rich:** the richest 1% of a world (10 people out of 1,000). Ties are broken at random.
- **Luck:** counted before looking at who got rich. It is the number of lucky events a person *received* (used or not) minus the number of unlucky events. Counting only the lucky events a person *used* would mix luck with talent.
- **The duel:** one person from the top 1% of talent against one from the bottom 1%. Who ends up richer? A tie counts as half. Duersch, Lambrecht & Oechssler (2020) measured the same thing for real games: 99.3% in tournament chess, 60.2% in two-player Crazy Eights.
- **The replay:** the same people (same talents, same positions) live their careers again with new events.

## Files

| file | what it is |
|---|---|
| `tvl.py` | the model (Python + Numba), with the rule variants as parameters |
| `test_tvl.py` | tests: events against a brute-force distance check, closed-form mean for independent luck, reproducibility, and more |
| `run_all.py` | every experiment in the video; writes `resultados.json` and `run_all.log` |
| `samples.py` | the one world shown on screen and its 10,000 replays; writes `samples.json` |
| `report.py` | prints the numbers in the order the video uses them |

```
pip install numpy numba
python test_tvl.py
python run_all.py        # about ten minutes on 24 threads; python run_all.py --quick for a small run
python samples.py
python report.py
```

Seed 2026. The full run is 3,180,000,000 simulated careers.

## Checked against the paper

One million worlds of the original model:

| | paper | this code |
|---|---|---|
| average talent of the richest person of a run | 0.667 | 0.670 |
| standard deviation of that talent | 0.09 | 0.093 |
| people with talent above 0.7, per run | about 160 | 158.6 |
| of those, share ending above the starting capital | 32% | 31.4% |
| people with no event at all | about 16% | 16.2% |
| capital held by the richest 20% | 80% (one run) | 77.2% (average) |

`run_all.py` asserts these six.

One number does not match: the paper says about 40% of people met only one kind of event; this code gives 48.1%. With 16% of people meeting no event and the two kinds independent, 48% is what the arithmetic gives (2 x 0.4 x 0.6), so the paper's figure looks approximate.

## Results: the original model

- The richest 1% hold 34.9% of all capital; the poorer half holds 7.4%. 44.5% of people end below where they started, 28.8% exactly where they started.
- The typical (median) rich person is more talented than 68% of people. 30.1% of the rich are below the median talent.
- The most talented person of a world ends up rich in 3.29% of worlds, and the richest in 0.54%.
- Chance of getting rich by talent decile: from 0.29% (bottom) to 2.04% (top), 6.9 times higher. 98% of the top decile still do not get rich.
- Events per person in 40 years: 1.54 lucky and 1.54 unlucky on average. The rich received 7.79 lucky and 0.33 unlucky.
- The typical rich person is luckier than 99% of the people in their world. 98.3% of the rich were in the luckiest 5%, and 51.5% in the luckiest 1%.
- With a net luck of 3 or less, 17 people got rich out of 915,616,812. A net luck of 4 or more happens to 8.4% of people (1 in 12). With 6 or more, the chance of getting rich is 61.6% in the top talent decile and 30.6% in the middle deciles.
- The duel: the top 1% of talent ends richer than the bottom 1% in 59.6% of pairs.
- The replay (200,000 worlds, two lives each): 1.24% of the first life's rich are rich again (chance alone: 1%). 48.9% land in the poorer half.

## Results: one rule at a time

100,000 worlds each (the same people and seeds as the first 100,000 worlds above); replays with 50,000 worlds.

| rule | richest 1% hold | duel | rich again | rich from the luckiest 5% |
|---|---|---|---|---|
| original | 34.9% | 59.7% | 1.24% | 98.3% |
| no streaks: the same number of events on average, independent | 11.2% | 64.1% | 1.22% | 93.5% |
| gentler events: x1.5 and /1.5 | 10.4% | 59.7% | same as original | 98.3% |
| talent attracts opportunity: lucky events arrive in proportion to talent | 45.4% | 67.3% | 2.08% | 99.6% |
| a paycheck every step, in proportion to talent | 23.7% | 84.9% | 1.43% | 93.4% |
| talent softens bad luck: divide by 2^(1 - talent) | 33.9% | 74.7% | 1.40% | 93.3% |
| the last three together | 34.7% | 98.1% | 2.76% | 91.5% |
| unequal start (lognormal, sigma 1; not in the video) | 40.4% | 58.1% | not run | 79.4% |

- **Streaks.** On the map, an event point that passed you is still nearby and often comes back. That is why 16.2% of people meet no event (4.5% with independent events) and 8.1% receive five or more lucky events (1.9% with independent events). Most of the inequality in the original model comes from this.
- **Gentler events** leave the list of rich people exactly unchanged (`test_gentler_events_same_rich_people`): the ranking only depends on how many events each person turned into gains and losses.
- **With all three advantages for talent,** the duel looks like chess (98.1%), and still only 2.76% of the rich repeat.

## Results: when does talent take over the top?

The paycheck world, with the size of an event turned down:

| a lucky event multiplies capital by | rich who come from the top talent decile | rich again |
|---|---|---|
| 2 | 24.2% | 1.42% |
| 1.7 | 25.7% | 1.50% |
| 1.5 | 27.9% | 1.65% |
| 1.3 | 33.5% | 2.05% |
| 1.2 | 41.1% | 2.71% |
| 1.1 | 63.1% | 6.68% |
| 1.05 | 90.1% | 25.39% |
| 1 (no luck) | 100% | 100% |

So the result is not a law of nature. It is what happens when luck multiplies what you have and talent only adds to it.

## The world shown on screen

World 2030, one of the first 1,000 worlds of the main run. It is the first one, in seed order, that looks like the typical case (the exact criteria are at the top of `samples.py`).

- The richest person: talent 0.641 (350th of 1,000), nine lucky events (eight used), no unlucky ones, 256 times the starting capital.
- The most talented person (0.869): three lucky events, three unlucky, and exactly the starting capital at the end.
- Over the next 11 lives, one of the original ten rich people made it back, once.
- Over 10,000 lives, the most talented person got rich 297 times, and every one of the 1,000 people got rich at least 7 times.

## Sources

- A. Pluchino, A. E. Biondo & A. Rapisarda, "Talent vs Luck: the role of randomness in success and failure", *Advances in Complex Systems* 21, 1850014 (2018). arXiv:1802.07068
- P. Duersch, M. Lambrecht & J. Oechssler, "Measuring skill and chance in games", *European Economic Review* 127 (2020), Table 2
