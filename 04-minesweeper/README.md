# Experiment 04: How Often Does Minesweeper Force You to Guess?

Code for the Brute Force Lab video *How Often Does Minesweeper Force You to Guess?*

A guess is **forced** when no covered square can be proven safe, using every number on the board and the total number of mines. A square is safe when it holds no mine in any arrangement of mines that agrees with everything visible. The first click is never counted as a guess.

## The players

Every game is played by the same deduction engine, in three levels, always tried in this order:

1. **Simple rules.** A number that already touches as many flags as it shows opens the rest of its neighbours. A number that needs exactly as many mines as it has covered neighbours flags them all. A mine counter at zero opens everything.
2. **Patterns.** Every two numbers that share covered squares are compared (this covers the 1-1 and 1-2-1 patterns). With `z` mines in the shared squares, each number's own squares hold `n - z`, so the possible range of `z` decides what is certain.
3. **Perfect logic.** The frontier (covered squares next to a number) is split into independent pieces. Squares that touch exactly the same numbers are grouped into boxes. Each piece is enumerated by backtracking, counting its arrangements by number of mines. The pieces are then combined with the squares that touch no number (`U` squares sharing the remaining `M - k` mines, weight `C(U, M - k)`).
   - Whether a square is safe is decided with exact booleans (does any arrangement that fits every number and the mine counter put a mine there?). No floating point is involved.
   - When nothing is safe, the same counts give the exact chance of a mine under every covered square.

When a guess is forced, one of four rules picks the square:

| rule | picks |
|---|---|
| `random` | any covered square |
| `frontier` | a random covered square next to a number |
| `safest` | the lowest exact chance of a mine (random tie-break) |
| `smart` | the lowest chance; ties go to the square with the fewest covered neighbours (the best chance to open an area) |

Three first-click rules: `any` (the first click can be a mine), `safe` (the first square is never a mine; classic Windows) and `zero` (the first square and its neighbours are never mines, so the first click always opens an area). Mines are placed uniformly among the allowed squares.

## Files

- `solver.py`: the engine (Numba): board generation, the three levels of deduction, the four guessing rules, one game, and a parallel batch runner.
- `runner.py`: runs millions of games in batches and keeps the statistics.
- `run_all.py`: every experiment from the video. Writes `resultados.json` and `run_all.log`.
- `test_solver.py`: brute-force checks on small boards (safe squares, mines, exact probabilities, the mine-counter flag, soundness of the pattern rule, the first-click rules, full games).
- `trace.py`: replays one game move by move (same seed, same code) to extract the positions shown in the video.
- `report.py`: prints every number used in the video from `resultados.json`.

## Results

**Checked against Tu et al. (2017), Table 2** (first click safe, in a corner, 10,000,000 games each):

| board | won, paper | won, mine | no guess, paper | no guess, mine |
|---|---|---|---|---|
| 8x8/10 | 78.44% | 78.497% | 36.060% | 36.075% |
| 16x16/40 | 74.30% | 74.298% | 34.490% | 34.500% |
| 30x16/99 | 35.56% | 35.562% | 5.097% | 5.096% |

**Games with at least one forced guess** (25,000,000 games each, best first click):

| board | classic (first click never a mine) | modern (first click opens an area) |
|---|---|---|
| Beginner 9 x 9/10 | 45.5% | 8.1% |
| Intermediate 16 x 16/40 | 65.5% | 27.8% |
| Expert 30 x 16/99 | 94.9% | 83.8% |

**Classic Expert, every time the simple rules run dry:** a pattern finds a move 48.2%, full logic finds a safe square 6.5%, truly forced 45.3%.

**Where and how bad (Expert):** a forced guess on the second click in 49.3% of classic games. An exact 50/50 is 8.4% (classic) and 15.4% (modern) of forced guesses; two squares and one mine: 2.9% and 5.5%.

**Games won by way of guessing (same boards):**

| | random | frontier | safest | smart |
|---|---|---|---|---|
| beginner, classic | 87.00% | 76.26% | 89.54% | 91.53% |
| beginner, modern | 96.54% | 96.13% | 97.02% | 96.99% |
| intermediate, classic | 68.81% | 59.08% | 74.29% | 77.56% |
| intermediate, modern | 85.87% | 84.85% | 88.26% | 88.25% |
| expert, classic | 22.83% | 19.80% | 35.58% | 38.27% |
| expert, modern | 37.39% | 36.38% | 50.62% | 50.91% |

A random square next to the numbers hides a mine 39.3% of the time on classic Expert; a random square anywhere, 24.0%.

**Best first click on Expert:** a corner with the classic rule (38.46% won, center 34.89%); (3,3) with the modern rule (50.94%, center 49.37%).

**Density:** with the modern first click, the share of games that need no guess falls from about 99.8% at 4% mines to about 0 by 25 to 30%, on all three board sizes (`resultados.json`, key `sweep`).

Total: 902,400,000 games. Seed 2026. On a 24-thread CPU, `run_all.py` takes about 3.5 hours.

## Run it

```
pip install numba numpy
python test_solver.py
python run_all.py
python report.py
```

## Sources

- J. Tu, T. Li, S. Chen, C. Zu & Z. Gu, *Exploring Efficient Strategies for Minesweeper*, AAAI-17 Workshops (2017)
- R. Dempsey & C. Guinn, *A Phase Transition in Minesweeper*, FUN with Algorithms (2020). arXiv:2008.04116
- R. Kaye, *Minesweeper is NP-complete*, The Mathematical Intelligencer 22 (2000)
- A. Scott, U. Stege & I. van Rooij, *Minesweeper May Not Be NP-Complete but Is Hard Nonetheless*, The Mathematical Intelligencer 33 (2011)
- D. N. Hill, JSMinesweeper (github.com/DavidNHill/JSMinesweeper)

Microsoft Minesweeper is a Microsoft product. This project is not affiliated with or endorsed by Microsoft.
