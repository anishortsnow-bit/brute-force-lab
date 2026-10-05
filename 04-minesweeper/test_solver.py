"""Checks the perfect-logic solver against brute force on small boards.

For random positions (some cells open, some true mines flagged) every mine arrangement of the covered cells
is listed by brute force. The solver must find exactly the cells that are safe in all of them, exactly the
cells that are mines in all of them, and (when nothing is safe) the exact mine probabilities.
Also checks the three first-click rules and that a whole game never opens a mine with logic alone.
Run: python test_solver.py
"""
import itertools, sys, time
import numpy as np
import solver as S

BIN = S.binom_table()


def brute(W, H, M, st, num):
    N = W * H
    cov = [i for i in range(N) if st[i] != 1]
    opn = [i for i in range(N) if st[i] == 1]
    NB, NBC = S.neighbors(W, H)
    total = 0
    cnt = {i: 0 for i in cov}
    for combo in itertools.combinations(cov, M):
        ms = set(combo)
        if any(i not in ms for i in cov if st[i] == 2):
            continue
        ok = True
        for c in opn:
            if sum(1 for k in range(NBC[c]) if NB[c, k] in ms) != num[c]:
                ok = False
                break
        if ok:
            total += 1
            for i in ms:
                cnt[i] += 1
    return total, cnt


def random_position(W, H, M, rng):
    N = W * H
    mine = np.zeros(N, np.int8)
    mine[rng.choice(N, M, replace=False)] = 1
    NB, NBC = S.neighbors(W, H)
    num = np.array([sum(mine[NB[i, k]] for k in range(NBC[i])) for i in range(N)], np.int8)
    st = np.zeros(N, np.int8)
    safe = [i for i in range(N) if not mine[i]]
    nopen = rng.integers(1, len(safe))
    stack = np.empty(N + 8, np.int32)
    for i in rng.choice(safe, nopen, replace=False):
        if rng.random() < 0.5:
            S.reveal(int(i), st, num, NB, NBC, stack)
        else:
            st[i] = 1
    for i in range(N):
        if mine[i] and rng.random() < 0.25:
            st[i] = 2
    return mine, num, st, NB, NBC


def check_positions(n_tests=1500, seed=1):
    rng = np.random.default_rng(seed)
    boards = [(5, 4, 4), (5, 5, 6), (6, 4, 7), (4, 4, 3), (6, 5, 5), (7, 3, 6), (5, 5, 10)]
    worst = 0.0; n_forced = 0; n_safe_found = 0; n_glob_checked = 0
    for t in range(n_tests):
        W, H, M = boards[t % len(boards)]
        N = W * H
        mine, num, st, NB, NBC = random_position(W, H, M, rng)
        if (st == 1).sum() == N - M:
            continue
        total, cnt = brute(W, H, M, st, num)
        assert total > 0
        sbuf = np.empty(N, np.int32); mbuf = np.empty(N, np.int32); prob = np.full(N, -1.0); res = np.zeros(8, np.int64)
        S.full_solve(M, st, num, NB, NBC, BIN, 10**8, sbuf, mbuf, prob, res)
        safe = set(sbuf[:res[0]].tolist()); mines = set(mbuf[:res[1]].tolist())
        want_safe = {i for i in range(N) if st[i] == 0 and cnt[i] == 0}
        want_mine = {i for i in range(N) if st[i] == 0 and cnt[i] == total}
        assert safe == want_safe, (t, safe, want_safe)
        assert mines == want_mine, (t, mines, want_mine)
        assert all(cnt[i] == total for i in range(N) if st[i] == 2)         # flags given were real mines
        # "needed the mine counter": some safe cell is not safe from the numbers alone
        front = [i for i in range(N) if st[i] == 0 and any(st[NB[i, k]] == 1 for k in range(NBC[i]))]
        if res[0] > 0 and len(front) <= 14:
            opn = [c for c in range(N) if st[c] == 1 and any(st[NB[c, k]] == 0 for k in range(NBC[c]))]
            can_mine = set()
            for bits in range(1 << len(front)):
                ms = {front[j] for j in range(len(front)) if bits >> j & 1}
                if all(sum(1 for k in range(NBC[c]) if NB[c, k] in ms or st[NB[c, k]] == 2) == num[c] for c in opn):
                    can_mine |= ms
            want_glob = any(i not in front or i in can_mine for i in safe)
            assert bool(res[2]) == want_glob, (t, res[2], want_glob)
            n_glob_checked += 1
        if res[0] == 0:
            n_forced += 1
            for i in range(N):
                if st[i] == 0:
                    worst = max(worst, abs(prob[i] - cnt[i] / total))
        else:
            n_safe_found += 1
    assert worst < 1e-9, worst
    print(f'positions: {n_tests}, with safe cells: {n_safe_found}, forced: {n_forced}, max probability error {worst:.1e}, '
          f'mine-counter flag checked {n_glob_checked}')


def check_patterns(n_tests=1500, seed=3):
    """The two-number pattern rule must be sound: everything it opens is safe, everything it flags is a mine."""
    rng = np.random.default_rng(seed)
    boards = [(5, 4, 4), (5, 5, 6), (6, 4, 7), (6, 5, 5), (7, 3, 6)]
    found = 0
    for t in range(n_tests):
        W, H, M = boards[t % len(boards)]
        N = W * H
        mine, num, st, NB, NBC = random_position(W, H, M, rng)
        if (st == 1).sum() == N - M:
            continue
        total, cnt = brute(W, H, M, st, num)
        st2 = st.copy(); c2 = np.zeros(2, np.int64); stack = np.empty(N + 8, np.int32)
        S.pair_pass(st2, num, NB, NBC, stack, c2, 1)      # one deduction: it may only use what the position shows
        for i in range(N):
            if st[i] == 0 and st2[i] == 1:      # deduced, or opened by the zero area of a deduced cell
                cascade = any(st[NB[i, k]] == 0 and st2[NB[i, k]] == 1 and num[NB[i, k]] == 0 for k in range(NBC[i]))
                assert mine[i] == 0 and (cnt[i] == 0 or cascade), (t, i); found += 1
            if st[i] == 0 and st2[i] == 2:
                assert cnt[i] == total, (t, i); found += 1
    print(f'pattern rule sound on {n_tests} positions ({found} deductions checked)')


def check_rules(n=20000):
    NB, NBC = S.neighbors(9, 9)
    rs = np.zeros(1, np.uint64)
    mine = np.zeros(81, np.int8); num = np.zeros(81, np.int8); pool = np.empty(81, np.int32)
    first = 40
    hit = np.zeros(81)
    for g in range(n):
        S.seed_game(rs, 7, g)
        S.make_board(10, S.R_ZERO, first, rs, NB, NBC, mine, num, pool)
        assert mine.sum() == 10 and mine[first] == 0 and num[first] == 0
        hit += mine
        S.seed_game(rs, 8, g)
        S.make_board(10, S.R_SAFE, first, rs, NB, NBC, mine, num, pool)
        assert mine.sum() == 10 and mine[first] == 0
    allowed = [i for i in range(81) if i != first and i not in NB[first, :NBC[first]]]
    p = hit[allowed] / n
    assert abs(p.mean() - 10 / 72) < 1e-12 and p.std() < 0.01, (p.mean(), p.std())
    print('first-click rules ok; zero rule: every allowed cell holds a mine', f'{p.min():.3f}..{p.max():.3f} of the time (expected {10 / 72:.3f})')


def check_games(n=3000):
    """Full games: logic must never open a mine, and a forced guess must really have no safe cell."""
    for (W, H, M) in ((9, 9, 10), (16, 16, 40), (30, 16, 99)):
        NB, NBC = S.neighbors(W, H); N = W * H
        rs = np.zeros(1, np.uint64)
        mine = np.zeros(N, np.int8); num = np.zeros(N, np.int8); st = np.zeros(N, np.int8)
        pool = np.empty(N, np.int32); stack = np.empty(N + 8, np.int32); prob = np.zeros(N)
        sbuf = np.empty(N, np.int32); mbuf = np.empty(N, np.int32); res = np.zeros(8, np.int64); cand = np.empty(N, np.int32)
        out = np.zeros(S.NOUT, np.int64)
        G = 64
        gp = np.zeros(G); gpmin = np.zeros(G); gfrac = np.zeros(G); gcell = np.zeros(G, np.int32)
        gnunk = np.zeros(G, np.int32); gmr = np.zeros(G, np.int32); gtie = np.zeros(G, np.int32)
        wins = 0; t0 = time.time()
        for g in range(n):
            S.seed_game(rs, 99, g)
            S.play(M, S.R_SAFE, 0, S.S_SAFEST, rs, NB, NBC, BIN, 10**7, mine, num, st, pool, stack, prob, sbuf, mbuf, res, cand,
                   out, gp, gpmin, gfrac, gcell, gnunk, gmr, gtie)
            wins += out[S.O_WIN]
            # flags placed by logic are always real mines; opened cells are never mines (except a losing guess)
            assert all(mine[i] == 1 for i in range(N) if st[i] == 2)
            if out[S.O_WIN]:
                assert (st == 1).sum() == N - M
        print(f'{W}x{H}/{M}: {n} games, logic never failed, win rate {wins / n:.3f}, {1000 * (time.time() - t0) / n:.2f} ms/game (1 thread)')


if __name__ == '__main__':
    check_rules()
    check_positions(int(sys.argv[1]) if len(sys.argv) > 1 else 1500)
    check_patterns()
    check_games()
    print('ALL TESTS PASSED')
