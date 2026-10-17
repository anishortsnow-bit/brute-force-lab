# Experiment 07: 53 Million Games of Monopoly

Code for the Brute Force Lab video *I Simulated 53 Million Games of Monopoly*.

A viewer asked for Monopoly to be solved. It can't be: half of the game is people making deals, and there is no rule for that. The other half is dice, and this is that half: the full game on the standard US board, played by simple players, with one thing changed at a time.

**Everything about trading here describes these players, not people.** See "What is simplified".

## The game

The official rule sheet, standard US board (2008 amounts for the two tax squares):

- dice, doubles, three doubles to Jail, the GO salary, both card decks (16 cards each, drawn in order and returned to the bottom);
- buying at the printed price, and an auction among all players when the player who landed there does not buy;
- rents: double on a complete unimproved color group, railroads by the number owned, utilities by the dice;
- even building, the 32 houses and 12 hotels of the box, buildings sold back at half price, mortgages with 10% interest;
- Jail: pay $50, use a card, or roll doubles; out on the third turn;
- bankruptcy to a player or to the Bank (the Bank auctions the lots).

A game still running after 1,000 rounds counts as **unfinished**: nobody wins it.

## The players

A player is a row of numbers (`mono.policy`), so one habit can be changed at a time. The default player:

- buys whatever it lands on if it can afford it;
- bids up to the printed price at auctions (twice the price when the lot completes a color group, its own or an opponent's);
- builds as soon as it owns a color group, up to hotels, keeping $100 in hand;
- pays to leave Jail at once;
- swaps one lot for one with another player when that swap completes a color group for both.

## What is simplified

- **Trading is that one swap rule.** No cash deals, no three-way deals, no bluffing, no ganging up on the leader. Everything that depends on trading (how long games last, how many never end, what refusing to trade costs) is a statement about these players.
- When the box runs short of houses, they are first come, first served (the rule sheet auctions them).
- Get Out of Jail Free cards are not sold between players.
- A creditor who receives mortgaged lots does not pay the 10% transfer interest.

The ranking of the color groups comes from duels played with trading switched off, so it does not depend on the swap rule.

## Files

| file | what it is |
|---|---|
| `mono.py` | the game and the players (Python + Numba) |
| `test_mono.py` | tests: the printed board numbers, rents, landing shares against the published table, the invariants after every turn, reproducibility, symmetry between identical players |
| `run_all.py` | every experiment in the video; writes `resultados.json` and `run_all.log` |
| `samples.py` | the one game shown on screen (a baseline game of `run_all.py`); writes `samples.json` |

```
pip install numpy numba
python test_mono.py
python run_all.py        # about twenty minutes on 24 threads; python run_all.py --quick for a small run
python samples.py
```

Seed 2026. The full run is 53,290,800 games and 400 million dice rolls for the landing table.

## Checked against a published table

Where the tokens land in the long run (share of all rolls that end on the square, leaving Jail at once), against I. Stewart, "Monopoly Revisited", *Scientific American*, October 1996:

| square | published | this code |
|---|---|---|
| Illinois Avenue | 3.18% | 3.19% |
| GO | 3.11% | 3.10% |
| B&O Railroad | 3.06% | 3.07% |
| Reading Railroad | 2.99% | 2.97% |
| Pennsylvania Railroad | 2.91% | 2.92% |
| Water Works | 2.81% | 2.81% |
| Electric Company | 2.62% | 2.60% |
| Short Line | 2.44% | 2.43% |
| Chance (third one) | 0.87% | 0.88% |

`run_all.py` asserts all nine within 0.04 points. The published numbers match players who pay to leave Jail at once; with players who stay as long as allowed, Illinois Avenue drops to 3.00%. The article gives 5.89% for the Jail square; this code gives 6.21% (2.27% just visiting plus 3.94% in Jail), and since the article does not say exactly what its figure counts, that square is not asserted. The most landed street after Illinois is New York Avenue (3.08%): Jail sends tokens back to square 10, and the orange group sits 6, 8 and 9 steps past it.

## Results: four identical players

4 million games.

- Each player wins 20.6% of games; 17.5% never end. With a limit of 10,000 rounds instead of 1,000, 17.3% are still running. In 11.7% of games nobody ever completes a color group.
- Games that end last 53 rounds at the median (about 200 turns), 61 on average, and 90% are over by round 90.
- Seat order, in games that end: 27.4%, 25.8%, 24.2%, 22.6%.

Games that never end, by number of players (1 million games each):

| players | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|
| never end | 0.005% | 1.4% | 17.5% | 41.2% | 59.6% |

## Results: color against color

Two players, each starting with one complete color group, everything else for sale, no trading. 200,000 games per pair. How often the row beats the column:

| | brown | light blue | pink | orange | red | yellow | green | dark blue |
|---|---|---|---|---|---|---|---|---|
| **brown** | | 13% | 17% | 9% | 15% | 16% | 25% | 25% |
| **light blue** | 85% | | 52% | 40% | 51% | 51% | 65% | 60% |
| **pink** | 82% | 48% | | 44% | 51% | 53% | 66% | 63% |
| **orange** | 91% | 60% | 56% | | 58% | 62% | 75% | 72% |
| **red** | 85% | 49% | 49% | 42% | | 54% | 67% | 67% |
| **yellow** | 84% | 49% | 47% | 38% | 46% | | 65% | 64% |
| **green** | 75% | 35% | 34% | 25% | 33% | 35% | | 49% |
| **dark blue** | 75% | 40% | 37% | 28% | 33% | 36% | 51% | |

Orange beats every other color. Light blue beats dark blue 60% of the time. Green loses to every color except brown.

In four-player games, the player who completes the first color group wins this share of the games that end:

| brown | light blue | pink | orange | red | yellow | green | dark blue |
|---|---|---|---|---|---|---|---|
| 22% | 65% | 65% | 75% | 59% | 60% | 47% | 78% |

Brown is the first group completed in 45% of games, dark blue in 12%, green in 4%.

## Results: one habit at a time

One player changes one habit; the other three stay the same. 1 million games each. An unchanged player wins 20.6%.

| habit | wins | change |
|---|---|---|
| spends everything on building (keeps $0) | 21.9% | +1.3 |
| keeps $300 before building | 17.5% | -3.1 |
| keeps $600 before building | 13.1% | -7.5 |
| stops at three houses | 16.8% | -3.9 |
| stops at four houses | 19.1% | -1.5 |
| stays in Jail as long as allowed, always | 20.2% | -0.4 |
| stays in Jail once an opponent has buildings | 21.5% | +0.9 |
| keeps $200 after buying a lot | 19.5% | -1.1 |
| keeps $500 after buying a lot | 15.6% | -5.1 |
| never buys utilities | 21.1% | +0.5 |
| never buys railroads | 20.8% | +0.2 |
| never buys the browns | 15.5% | -5.2 |
| never buys the greens | 21.2% | +0.5 |
| never buys the dark blues | 13.2% | -7.4 |
| bids up to half the printed price | 20.1% | -0.5 |
| bids up to 1.5 times the printed price | 21.4% | +0.7 |
| never trades | 6.6% | -14.1 |

- The best habits together (keep $0, stay in Jail late, bid higher): 23.7% against three unchanged players.
- A head start, one player owning a group before the first roll (500,000 games each): both utilities 31% of wins, the four railroads 86%, brown 44%, any other color group 92% to 99%.

## Results: what makes games endless

Four identical players, 1 million games each:

| rules | never end |
|---|---|
| official | 17.5% |
| taxes and fines go to Free Parking | 18.1% |
| the same, plus $500 from the Bank each time | 19.0% |
| no auctions | 16.8% |
| seeded jackpot and no auctions | 18.7% |
| no trading | 67.8% |

With these players, the house rules hardly matter, and trading matters a great deal.

Monopoly is a trademark of Hasbro. This is an independent analysis, not affiliated with or endorsed by Hasbro.
