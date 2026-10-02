# Experiment 02: I Played 100 Million Games of Solitaire

Code for the Brute Force Lab video *I Played 100 Million Games of Solitaire*.

The game is Klondike, turning **one** card at a time with **unlimited passes** through the stock. Under these rules, every stock card can be reached at any moment, so the stock is modelled as a set of cards that can be played whenever they fit.

## Files

- `klondike.py`: the whole engine (Numba).
  - `deal(n)`: deal number `n` is reproducible, shuffled with splitmix64 seeded with `n`.
  - Three players that only see what a human sees. Face-down cards stay unknown to them.
    - `0` random: picks any sensible legal move at random.
    - `1` greedy: always takes the first move on a fixed priority list.
    - `2` careful: rules of thumb.
      - Only sends a card to the foundations when nothing still in play could need it.
      - Digs into the column with the most face-down cards.
      - Plays stock cards that free buried runs.
      - Saves empty columns.
  - A "thoughtful" solver that sees every card. It runs a depth-first search with:
    - A transposition table, so positions already seen are skipped.
    - Safe automatic moves to the foundations.
    - Pruning that never loses a win: a stock card is only put on the tableau if a visible card could then go on it. In turn-one with unlimited passes, a stock card can always wait.
  - The search has a node limit. Deals that hit it are reported as **unknown**, never as won or lost.
- `run_all.py`: the experiments from the video.
- `trajetoria.py`: records single games move by move. The animations come from these real games.

## Results (from the video)

| | games | won |
|---|---|---|
| random player | 100,000,000 | 4.87% |
| greedy player | 100,000,000 | 13.24% |
| careful player | 100,000,000 | 24.63% |

Solver, deals 0 to 19,999, every card known:

| | deals |
|---|---|
| won | 17,836 |
| proven impossible | 1,457 |
| unknown after 20,000,000 positions | 707 |

So between 89.2% and 92.7% of turn-one deals are winnable with every card known.

**Deals with no playable card at all:** 15 in 100,000,000, about 1 in 6.7 million. Every ace is face down and no card fits anywhere.

**Comparison:** C. Blake & I. P. Gent, *The Winnability of Klondike Solitaire and Many Other Patience Games* (arXiv:1906.12314). They report 81.945% ± 0.084% for turn-**three** Klondike with every card known.

## Run it

```
pip install numba numpy
python run_all.py
```

The full run takes about 40 minutes on a 24-thread CPU.
