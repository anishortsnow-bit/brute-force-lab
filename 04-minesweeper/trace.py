"""Replays single games move by move (same seeds and same code as the big runs) and records snapshots.

Used to pick real positions for the video: a forced 50/50 at the end of an Expert game, a spot where the simple
rules are stuck but full logic finds a safe cell, and the list of every mine arrangement on a small frontier.
"""
import numpy as np
import solver as S
import runner as R


class Game:
    def __init__(self, W, H, M, rule, first, strategy, seed, game):
        self.W, self.H, self.M, self.N = W, H, M, W * H
        self.NB, self.NBC = R.geometry(W, H)
        self.rule, self.first, self.strategy = rule, first, strategy
        N = self.N
        self.rs = np.zeros(1, np.uint64)
        S.seed_game(self.rs, seed, game)
        self.mine = np.zeros(N, np.int8); self.num = np.zeros(N, np.int8); self.st = np.zeros(N, np.int8)
        self.pool = np.empty(N, np.int32); self.stack = np.empty(N + 8, np.int32); self.prob = np.zeros(N)
        self.sbuf = np.empty(N, np.int32); self.mbuf = np.empty(N, np.int32); self.res = np.zeros(8, np.int64)
        self.cand = np.empty(N, np.int32)
        S.make_board(M, rule, first, self.rs, self.NB, self.NBC, self.mine, self.num, self.pool)
        self.cnt = np.zeros(2, np.int64)
        self.events = []

    def snap(self, kind, **kw):
        e = {'kind': kind, 'st': self.st.copy(), 'open': int(self.cnt[0]), 'flags': int(self.cnt[1])}
        e.update(kw)
        self.events.append(e)

    def play(self):
        W, N, M = self.W, self.N, self.M
        S_ = N - M
        if self.mine[self.first]:
            self.snap('boom', cell=self.first)
            return False
        self.cnt[0] += S.reveal(self.first, self.st, self.num, self.NB, self.NBC, self.stack)
        self.snap('first', cell=self.first)
        while True:
            if self.cnt[0] == S_:
                self.snap('win')
                return True
            before = self.st.copy()
            S.basic_pass(M, self.st, self.num, self.NB, self.NBC, self.stack, self.cnt)
            if (before != self.st).any():
                self.snap('basic', before=before)
            if self.cnt[0] == S_:
                self.snap('win')
                return True
            before = self.st.copy()
            if S.pair_pass(self.st, self.num, self.NB, self.NBC, self.stack, self.cnt) > 0:
                self.snap('pattern', before=before)
                continue
            S.full_solve(M, self.st, self.num, self.NB, self.NBC, R.BIN, R.NODE_CAP, self.sbuf, self.mbuf, self.prob, self.res)
            mines = self.mbuf[:self.res[1]].copy()
            for q in mines:
                self.st[q] = 2; self.cnt[1] += 1
            if self.res[0] > 0:
                safe = self.sbuf[:self.res[0]].copy()
                self.snap('hidden', safe=safe, mines=mines, counter=int(self.res[2]))
                for q in safe:
                    self.cnt[0] += S.reveal(int(q), self.st, self.num, self.NB, self.NBC, self.stack)
                continue
            prob = self.prob.copy()
            g, pmin, ntie = S.choose_guess(self.strategy, self.st, self.NB, self.NBC, self.prob, self.rs, self.cand)
            self.snap('guess', cell=int(g), prob=prob, pmin=float(pmin), ntie=int(ntie), mines=mines, boom=bool(self.mine[g]))
            if self.mine[g]:
                return False
            self.cnt[0] += S.reveal(int(g), self.st, self.num, self.NB, self.NBC, self.stack)


def frontier_pieces(W, H, st, num):
    """Groups the covered cells next to numbers into independent pieces (cells, numbers)."""
    NB, NBC = R.geometry(W, H); N = W * H
    front = [i for i in range(N) if st[i] == 0 and any(st[NB[i, k]] == 1 for k in range(NBC[i]))]
    fset = set(front)
    parent = {i: i for i in front}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for c in range(N):
        if st[c] != 1:
            continue
        cov = [NB[c, k] for k in range(NBC[c]) if NB[c, k] in fset]
        for x in cov[1:]:
            parent[find(x)] = find(cov[0])
    groups = {}
    for i in front:
        groups.setdefault(find(i), []).append(i)
    out = []
    for cells in groups.values():
        cs = set(cells)
        nums = [c for c in range(N) if st[c] == 1 and any(NB[c, k] in cs for k in range(NBC[c]))]
        out.append((sorted(cells), nums))
    return out


def arrangements(W, H, st, num, cells, nums):
    """Every mine arrangement of `cells` that agrees with the numbers in `nums` (flags count as mines)."""
    NB, NBC = R.geometry(W, H)
    idx = {c: j for j, c in enumerate(cells)}
    need = {}
    for c in nums:
        need[c] = int(num[c]) - sum(1 for k in range(NBC[c]) if st[NB[c, k]] == 2)
    touch = {c: [idx[NB[c, k]] for k in range(NBC[c]) if NB[c, k] in idx] for c in nums}
    sols = []
    x = [0] * len(cells)

    def ok(upto):
        for c in nums:
            s = sum(x[j] for j in touch[c] if j < upto); rest = sum(1 for j in touch[c] if j >= upto)
            if s > need[c] or s + rest < need[c]:
                return False
        return True

    def rec(j):
        if j == len(cells):
            sols.append(tuple(x)); return
        for v in (0, 1):
            x[j] = v
            if ok(j + 1):
                rec(j + 1)
        x[j] = 0
    rec(0)
    return sols
