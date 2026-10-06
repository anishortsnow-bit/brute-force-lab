# Experiment 03: The Perfect Strategy for Guess Who

Code for the Brute Force Lab video *The Perfect Strategy for Guess Who*.

Guess Who? is treated as a race between two shrinking lists of suspects. A position is `(n, m)`: the player to move still has `n` faces up, the opponent has `m`. A question splits the mover's `n` faces into `b` ("yes") and `n - b` ("no"). Any split is possible, because a question can be about any set of faces ("Is your person one of these six?").

The game is solved for three ways of ending it:

| rule | what it means | first player wins (24 vs 24) |
|---|---|---|
| `official` | A guess uses your turn. A wrong guess loses the game. | 5/9 = 55.6% |
| `free_guess` | A guess uses your turn. A wrong guess only removes that face. | 2/3 = 66.7% |
| `instant` | The moment you are down to one face, you win. | 91/144 = 63.2% |

## Files

- `exact.py`: the exact solution, with fractions.
  - `solve(rule)`: the win probability and every optimal move for all positions.
  - `evaluate(rule, a, b)`: the exact result of any two fixed strategies playing each other.
  - The published theorems as functions: `cushing` (official rules) and `nica` (instant win), with Nica's closed form `nica_value`.
- `board.py`: the board from the video. 24 original faces, one name per letter from A to X, built like the classic board: nine yes/no features with five faces each, and five hair colours with 5, 4, 5, 5 and 5 faces.
- `classic.py`: the feature list of the classic board, used only as a cross-check.
- `sim.py`: the Monte Carlo tournament (Numba). Every game has two real mystery people and two real lists of suspects, and every question is a real set of faces. It knows nothing about `exact.py`, so the two can be checked against each other.
  - `RANDOM_FEATURE`: asks a random single-feature question that still splits its suspects.
  - `BEST_FEATURE`: asks the single-feature question whose split is closest to half.
  - `TABLE`: follows a table `move[n][m]` (the splitter, the perfect players and the bold player).
- `run_all.py`: every experiment from the video. Writes `resultados.json`.

## Results (from the video)

**The exact solution agrees with the published results.**
- Official rules: the strategy of Cushing, Gipp, Levick, Rickinson & Stewart (2025) is optimal in every position checked (up to 40 vs 40).
- Instant win: the strategy and the closed-form win probability of Nica (2016) match in every position checked (up to 40 vs 40).
- From 23 vs 23 (editions where you draw a card and can rule out your own face), the first player wins 55.95% under the official rules and 65.97% with free guesses, the values in O'Neill (2021).

**Official rules, 100,000,000 games per pairing, share of games won by the player who moves first:**

| first \ second | random feature | best feature | splitter | perfect | bold |
|---|---|---|---|---|---|
| random feature | 55.50% | 50.89% | 44.62% | 44.71% | 48.52% |
| best feature | 61.09% | 56.58% | 50.53% | 50.64% | 54.07% |
| splitter | 67.49% | 64.12% | 55.55% | 55.55% | 65.28% |
| perfect | 67.31% | 64.17% | 55.56% | 55.56% | 65.97% |
| bold | 66.07% | 62.50% | 55.55% | 55.55% | 62.85% |

- Taking turns going first, the splitter beats random feature questions 61.4% of the time and the best single-feature questions 56.8%.
- Perfect against the splitter: exactly 5/9 for whoever moves first, in all four pairings. On a 24-face board the splitter gives up nothing against a perfect opponent.
- A splitter that never takes a last-chance guess wins 11.1% of its games going second, instead of 44.4%.
- Exact split in half is not optimal in 35 of the 576 positions (for example, with 6 faces against 8, asking about 2 beats asking about 3).
- If the second player starts with 16 faces instead of 24, the game is exactly 50/50.

**Instant win.** Going second against a splitter, another splitter wins 22.2% and the bold player wins 41.7%.

**Free guesses.** The perfect player guesses with three faces or fewer, and whenever the opponent has two or fewer. Against a splitter that waits until two, that is worth 11.1 points.

**Simulation against the exact values.** The largest gap over all pairings of table strategies is 0.005 percentage points.

## Run it

```
pip install numba numpy
python run_all.py
```

The 4.9 billion games take about 80 seconds on a 24-thread CPU.

## Sources

- M. Nica, *Optimal Strategy in "Guess Who?": Beyond Binary Search*, Probability in the Engineering and Informational Sciences 30 (2016). arXiv:1509.03327
- D. Cushing, S. Gipp, E. Levick, E. Rickinson & D. I. Stewart, *Optimal play in "Guess Who?"* (2025). arXiv:2508.00799
- B. O'Neill, *Optimal guessing in "Guess Who"*, PLOS ONE 16(3), 2021

Guess Who? is a trademark of Hasbro. This project is not affiliated with or endorsed by Hasbro.
