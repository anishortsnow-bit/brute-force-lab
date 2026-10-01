# Experiment 01: Can a Game of War Last Forever?

Video: https://youtu.be/4JHUfKRKYmc

| File | What it does |
|---|---|
| `war_sim.py` | Plays full 52-card games of War (Numba, parallel). Rules: war with 1 or 3 cards face down; won cards returned in a fixed or random order. Used for game lengths and the four-aces result (400 million games). |
| `war_cycles.py` | Detects infinite games with Brent's cycle detection. Runs the 800-million-game loop search on the real deck and the small-deck experiments (1 suit vs. several suits). |
| `war_chains.py` | Counts stacked wars (double, triple, ... war) in 100 million games. |

Requirements: Python 3.12, NumPy, Numba.

```python
from war_sim import simulate
rounds, wars, winner, aces = simulate(1_000_000, war_down=3, order=2, seed=1, max_rounds=100_000)
```

Comments in the code are in Portuguese.
