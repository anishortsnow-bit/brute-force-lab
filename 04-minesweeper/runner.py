"""Runs millions of games in parallel batches and keeps the statistics (no per-game data is stored)."""
import os, time
import numpy as np
import numba
import solver as S

BIN = S.binom_table()
NODE_CAP = 20_000_000
GMAX = 32                       # forced guesses kept per game (games with more are counted, not described)
_NB = {}


def geometry(W, H):
    if (W, H) not in _NB:
        _NB[(W, H)] = S.neighbors(W, H)
    return _NB[(W, H)]


def cell(W, H, r, c):
    return r * W + c


class Stats:
    """Aggregates one experiment (board, rule, first click, guessing rule)."""

    def __init__(self, W, H, M, detail):
        self.W, self.H, self.M, self.detail = W, H, M, detail
        self.n = 0
        self.wins = 0
        self.boom1 = 0
        self.noguess = 0            # finished without a single forced guess
        self.basic_only = 0         # ... and the simple rules were enough all the way
        self.pattern_only = 0       # ... and simple rules plus two-number patterns were enough
        self.pair = 0               # times the simple rules ran dry and a pattern found something
        self.capped = 0
        self.stuck = 0              # times the simple rules ran dry
        self.hidden = 0             # ... patterns did not help, and full logic still found a safe cell
        self.global_used = 0        # ... and that needed the mine counter
        self.games_hidden = 0       # games with at least one such moment before any guess
        self.guess_hist = np.zeros(64, np.int64)            # forced guesses per game
        self.guess_hist_won = np.zeros(64, np.int64)
        self.guesses = 0
        if detail:
            N = W * H
            self.p_hist = np.zeros(101, np.int64)            # chosen cell's mine chance, 1% bins (50% exact counted apart)
            self.pmin_hist = np.zeros(101, np.int64)
            self.coin = 0                                    # forced guesses where the best cell is exactly 50%
            self.coin_end2 = 0                               # ... with two covered cells and one mine left
            self.frac_hist = np.zeros(50, np.int64)          # how far into the game (share of safe cells open)
            self.first_frac_hist = np.zeros(50, np.int64)
            self.first_move = 0                              # first forced guess right after the first click
            self.cell_hist = np.zeros(N, np.int64)           # where forced guesses are clicked
            self.cell_lost = np.zeros(N, np.int64)
            self.recorded = 0
            self.lost_on_coin = 0
            self.pos_hist = np.zeros(50, np.int64)           # guess number at which games are lost
            self.p_frac = np.zeros((10, 101), np.int64)      # chosen cell's mine chance by stage of the game
            self.p_sum = 0.0

    def add(self, OUT, GP, GPMIN, GFRAC, GCELL, GNUNK, GMR, GTIE):
        n = OUT.shape[0]
        self.n += n
        win = OUT[:, S.O_WIN] == 1
        ng = OUT[:, S.O_GUESS]
        self.wins += int(win.sum())
        self.boom1 += int(OUT[:, S.O_BOOM1].sum())
        alive = OUT[:, S.O_BOOM1] == 0
        ok = alive & (ng == 0)
        self.noguess += int(ok.sum())
        self.pattern_only += int((ok & (OUT[:, S.O_HIDDEN] == 0)).sum())
        self.basic_only += int((ok & (OUT[:, S.O_HIDDEN] == 0) & (OUT[:, S.O_PAIR] == 0)).sum())
        self.pair += int(OUT[:, S.O_PAIR].sum())
        self.capped += int(OUT[:, S.O_CAPPED].sum())
        self.stuck += int(OUT[:, S.O_STUCK].sum())
        self.hidden += int(OUT[:, S.O_HIDDEN].sum())
        self.global_used += int(OUT[:, S.O_GLOBAL].sum())
        self.games_hidden += int((alive & (OUT[:, S.O_BASIC_HIDDEN1] == 1)).sum())
        self.guesses += int(ng[alive].sum())
        self.guess_hist += np.bincount(np.minimum(ng[alive], 63), minlength=64)
        self.guess_hist_won += np.bincount(np.minimum(ng[win], 63), minlength=64)
        if not self.detail:
            return
        G = GP.shape[1]
        k = np.minimum(ng, G)
        mask = np.arange(G)[None, :] < k[:, None]
        p = GP[mask]; pm = GPMIN[mask]; fr = GFRAC[mask]; ce = GCELL[mask]; nu = GNUNK[mask]; mr = GMR[mask]
        self.recorded += int(mask.sum())
        self.p_sum += float(p.astype(np.float64).sum())
        self.p_hist += np.bincount(np.clip((p * 100 + 1e-9).astype(int), 0, 100), minlength=101)
        self.pmin_hist += np.bincount(np.clip((pm * 100 + 1e-9).astype(int), 0, 100), minlength=101)
        coin = np.abs(pm - 0.5) < 1e-9
        self.coin += int(coin.sum())
        self.coin_end2 += int((coin & (nu == 2) & (mr == 1)).sum())
        self.frac_hist += np.bincount(np.clip((fr * 50).astype(int), 0, 49), minlength=50)
        fb = np.clip((fr * 10).astype(int), 0, 9); pb = np.clip((p * 100 + 1e-9).astype(int), 0, 100)
        self.p_frac += np.bincount(fb * 101 + pb, minlength=1010).reshape(10, 101)
        self.cell_hist += np.bincount(ce, minlength=self.W * self.H)
        f1 = OUT[:, S.O_FIRSTFRAC]
        has = alive & (ng > 0)
        self.first_frac_hist += np.bincount(np.clip(f1[has] * 50 // 10000, 0, 49), minlength=50)
        self.first_move += int((has & (OUT[:, S.O_IMMEDIATE] == 1)).sum())
        # the losing guess is the last one of a lost game
        lost = alive & ~win & (ng > 0) & (ng <= G)
        rows = np.nonzero(lost)[0]; last = ng[rows] - 1
        self.cell_lost += np.bincount(GCELL[rows, last], minlength=self.W * self.H)
        self.lost_on_coin += int((np.abs(GPMIN[rows, last] - 0.5) < 1e-9).sum())
        self.pos_hist += np.bincount(np.minimum(ng[rows] - 1, 49), minlength=50)

    def summary(self):
        n = self.n; a = n - self.boom1
        d = {'games': n, 'win': self.wins / n, 'win_se': (self.wins / n * (1 - self.wins / n) / n) ** 0.5,
             'boom_first_click': self.boom1 / n,
             'no_guess': self.noguess / n, 'no_guess_se': (self.noguess / n * (1 - self.noguess / n) / n) ** 0.5,
             'basic_only': self.basic_only / n, 'pattern_only': self.pattern_only / n, 'capped_games': self.capped,
             'stuck_moments': self.stuck, 'stuck_pattern': self.pair / max(self.stuck, 1),
             'stuck_full_logic': self.hidden / max(self.stuck, 1), 'stuck_forced': self.guesses / max(self.stuck, 1),
             'safe_needed_counter': self.global_used / max(self.hidden, 1),
             'games_beyond_simple_rules_before_guess': self.games_hidden / max(a, 1),
             'guesses_per_game': self.guesses / max(a, 1),
             'guesses_per_won_game': float((self.guess_hist_won * np.arange(64)).sum() / max(self.wins, 1)),
             'guess_hist': self.guess_hist.tolist(), 'guess_hist_won': self.guess_hist_won.tolist()}
        if self.detail:
            r = max(self.recorded, 1)
            d.update({'recorded_guesses': self.recorded, 'p_hist': self.p_hist.tolist(), 'pmin_hist': self.pmin_hist.tolist(),
                      'coin_flips': self.coin / r, 'coin_flips_endgame_2cells': self.coin_end2 / r,
                      'frac_hist': self.frac_hist.tolist(), 'first_frac_hist': self.first_frac_hist.tolist(),
                      'cell_hist': self.cell_hist.tolist(), 'cell_lost': self.cell_lost.tolist(),
                      'lost_on_coin': self.lost_on_coin, 'first_guess_immediate': self.first_move / max(a, 1), 'losing_guess_index_hist': self.pos_hist.tolist(),
                      'p_by_stage': self.p_frac.tolist(), 'mean_p_chosen': self.p_sum / r})
        return d


def run(W, H, M, rule, first, strategy, games, seed=2026, g0=0, detail=False, batch=None, log=None):
    NB, NBC = geometry(W, H)
    T = numba.get_num_threads()
    batch = batch or (200_000 if detail else 2_000_000)
    st = Stats(W, H, M, detail)
    G = GMAX if detail else 1
    t0 = time.time()
    done = 0
    while done < games:
        n = min(batch, games - done)
        OUT = np.zeros((n, S.NOUT), np.int64)
        GP = np.zeros((n, G), np.float32); GPM = np.zeros((n, G), np.float32); GF = np.zeros((n, G), np.float32)
        GC = np.zeros((n, G), np.int32); GN = np.zeros((n, G), np.int32); GM = np.zeros((n, G), np.int32); GT = np.zeros((n, G), np.int32)
        S.run_batch(W, H, M, S.RULES[rule] if isinstance(rule, str) else rule, first, S.STRATS[strategy] if isinstance(strategy, str) else strategy,
                    seed, g0 + done, n, T, NODE_CAP, NB, NBC, BIN, OUT, GP, GPM, GF, GC, GN, GM, GT)
        st.add(OUT, GP, GPM, GF, GC, GN, GM, GT)
        done += n
    dt = time.time() - t0
    s = st.summary()
    s.update({'W': W, 'H': H, 'M': M, 'rule': rule, 'first': int(first), 'strategy': strategy, 'seed': seed, 'seconds': round(dt, 1)})
    if log:
        log(f"{W}x{H}/{M} {rule:4s} first={first:3d} {strategy:8s} {games:>11,} games  win {100 * s['win']:.3f}%  "
            f"no-guess {100 * s['no_guess']:.3f}%  guesses/game {s['guesses_per_game']:.3f}  capped {s['capped_games']}  {dt:.0f}s")
    return s
