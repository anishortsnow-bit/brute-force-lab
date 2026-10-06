# Experiment 05: How Much of UNO Is Luck?

Code for the Brute Force Lab video *How Much of UNO Is Luck?*

## The luck meter

The best player I could build plays a player who picks a random legal move. If skill decided every game, it would win them all; if luck decided every game, it would win half. Model each game as decided by skill (the better player always wins) with probability `s`, or by a coin flip otherwise. A win rate `p` then means `s = 2p - 1`, and the share of games decided by luck is `2(1 - p)`. With `n` players the coin flip is `1/n`: `s = (p - 1/n) / (1 - 1/n)`.

Two ends of the scale: tic-tac-toe, solved exactly in `ttt.py` (the best response to a random player scores 97.99%, a draw counting as half a win: 4.0% luck), and War, which has no decisions (100% luck).

## The rules

Official Mattel rules (2008 instruction sheet), two players unless noted:

- **Deck:** 108 cards.
- **Playing a card:** match the color or the number/symbol, or play a wild card and name the next color.
- **Wild Draw Four:** only allowed when you hold no card of the current color. Every player here follows that rule, so nobody needs to challenge.
- **Drawing:** if you can't play, draw one card and play it if it fits. You may also draw instead of playing.
- **Two players:** Skip and Reverse mean play again. After a Draw Two or Wild Draw Four the other player draws and play comes back.
- **Start card:**
  - Wild Draw Four: goes back into the deck.
  - Wild: the first player names the color.
  - Draw Two: the first player draws two and misses the turn.
  - Reverse: the dealer starts.
  - Skip: the first player is skipped.
- **Empty deck:** the discard pile is reshuffled into a new deck.
- **House rules from the same sheet:** Progressive UNO (stack Draw Two on Draw Two, Wild Draw Four on Wild Draw Four) and "draw until you can play".
- **Crazy Eights** runs on the same engine: 52 cards, eights are wild, draw until you can play, 7 cards each with two players.

## The players

| player | how it plays |
|---|---|
| `random` | any legal card from the hand, at random; a random color for a wild |
| `casual` ("habits") | saves wild cards until nothing else fits; names the color it holds most of |
| `expert` ("hand-made") | scores every legal card: how many cards it keeps in the color it leaves behind, chaining Skip/Reverse/Draw Two with two players, attacking when the other player has one or two cards, colors the other player showed it lacks (from their draws), card points |
| `search` | determinized Monte Carlo search (below) |
| `cheat` | the same search, but it sees the other player's real hand (the deck order stays hidden) |

**How the search works:**
- At every decision with more than one option, it deals out the cards it can't see in K random ways. Each colour of a wild card counts as its own option, and so does drawing.
- Each deal is consistent with everything it has seen: the cards not in its hand or on the discard pile go to the other hands and the deck. When a player draws, it is assumed to hold nothing playable, so its oldest cards are kept free of that color, symbol and the wild cards.
- For each deal it tries every option and finishes the game with the hand-made strategy on both sides. All options share the same deals and random seeds.
- **Two stages:** every option gets K/4 deals, then the best three (plus the hand-made player's choice) get the rest. It plays the option that won most often.
- The main runs use K = 2048.

## Files

- `engine.py`: the game (Numba).
  - Hands are kept as card counts and bit masks, so legal moves are a few bit operations.
  - The state machine: `advance()` resolves everything automatic; `apply()` takes a decision.
  - Also: the policies, the search, `run_batch` (head-to-head games, the deal shared by both seats) and `run_matches` (official scoring, first to 500 points).
- `test_engine.py`: the checks, all passing.
  - Scripted rule positions and start cards.
  - 92,000 decisions checked against an independent implementation of the legal moves, with card-count invariants.
  - The search's card dealing.
  - Symmetry.
  - Replication of Sidajaya et al. (2024).
- `run_all.py`: every run from the video. Writes `resultados.json` and `run_all.log`.
- `trace.py`: replays one game move by move (same seed, same code), with the search's estimate for every option.
- `draws.py`: replays 600 games of the main run and sorts every deliberate draw. Writes `draws.json`.
- `ttt.py`: tic-tac-toe, exact.
- `report.py`: prints every number used in the video.

## Results (two players, official rules)

**Checked against Sidajaya et al. (2024)** with their setup: 4 players, one strategic player against three random ones, Wild Draw Four allowed any time, no start-card effects. 2,500,000 games each.

| their player | paper | mine |
|---|---|---|
| plays wild cards at once | 24.7% | 24.72% |
| saves wild cards | 31.7% | 30.59% to 33.18% |

The second row depends on how their code breaks color ties (`max(set(...))`, which follows Python's hash order and changes from run to run). With random ties: 31.87%.

**Against a random player:**

| player | wins | luck on the meter | games |
|---|---|---|---|
| habits | 60.09% | 79.8% | 10,000,000 |
| hand-made | 62.36% | 75.3% | 10,000,000 |
| **search** | **74.18% ± 0.22** | **51.6%** | 40,000 |
| cheat (sees your hand) | 91.74% ± 0.28 | 16.5% | 10,000 |

**Thinking time** (futures per option → wins against random):

| 8 | 32 | 128 | 512 | 2048 | 8192 |
|---|---|---|---|---|---|
| 46.97% | 57.23% | 67.28% | 72.27% | 74.18% | 75.05% |

**Between sensible players:**
- hand-made vs habits 52.24%
- search vs habits 62.54%
- search vs hand-made 61.64%
- cheat vs search 72.10%

**Matches to 500 points** (about 17 games each with two players):

| matchup | single game | match |
|---|---|---|
| hand-made vs habits | 52.27% | 68.55% |
| habits vs random | 60.05% | 81.79% |
| search vs habits | 63.22% | 80.85% |

**The best player's turns:**
- nothing playable: 20.7%;
- one kind of card fits: 34.4%;
- a real choice: 44.9%.

**Deliberate draws:**
- When a wild card was the only card it could play, the search drew instead 41.0% of the time (352 of 858, in 600 games).
- Without the option to draw by choice it wins 72.62% ± 0.45 instead of 74.18%.

**The deal** (hand-made vs hand-made, 20,000,000 games):
- Moving first wins 51.55%.
- Wild cards dealt, mine minus yours:

  | difference | −2 | −1 | 0 | +1 | +2 | +3 |
  |---|---|---|---|---|---|---|
  | wins | 18.0% | 34.8% | 50.0% | 65.2% | 82.0% | 91.8% |

**More players** (search vs random players; luck on the meter):

| players | 2 | 3 | 4 | 6 | 10 |
|---|---|---|---|---|---|
| luck | 52% | 76% | 84% | 92% | 95% |

**House rules** (luck on the meter):
- official 51.6%
- stacking 53.2%
- draw until you can play 43.2%
- both 44.9%
- Crazy Eights 50.4%

**Which habits matter** (hand-made player with one habit switched off, against random):
- color left on the table and color called: −5.52 points
- saving wild cards: −4.84
- remembering the colors the other player lacked: −1.69
- chaining: −0.39
- attacking near the end: −0.09
- dumping high cards first: +0.25

Total: 288,676,228 games. Seed 2026. On a 24-thread CPU, `run_all.py` takes about 3 hours.

## Run it

```
pip install numba numpy
python test_engine.py
python run_all.py
python report.py
```

## Sources

- Mattel, *UNO* instruction sheet (2008), including the house rules
- P. Sidajaya, J. H. K. Low, C. C. Aw & V. Scarani, *Emergence of fluctuation relations in UNO*, arXiv:2406.09348; code at github.com/PeterSidajaya/uno-fluctuation
- P. Duersch, M. Lambrecht & J. Oechssler, *Measuring skill and chance in games*, European Economic Review 127 (2020) 103472

UNO is a trademark of Mattel. This project is not affiliated with or endorsed by Mattel.
