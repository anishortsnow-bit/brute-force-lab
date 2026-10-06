"""Replays single games step by step and records everything a scene needs: every decision (hand, top card,
options, the search's win estimate for each option, the choice) and the state after it."""
import numpy as np
import engine as E

T = E.uno_table()


class Game:
    def __init__(self, R, seats, seed, game, dealer=None, K=256, z=1.5, rpol=E.P_EXPERT, opol=E.P_EXPERT, X=None,
                 T=T):
        self.R, self.T, self.seats, self.K, self.z = R, T, np.array(seats, np.int64), K, z
        self.rpol, self.opol = rpol, opol
        self.X = E.expert_flags() if X is None else X
        self.S = np.zeros(E.SLEN, np.int64)
        self.rs = np.zeros(1, np.uint64)
        NP = R[E.R_NP]
        E.seed_game(self.rs, seed, game // NP)
        E.new_game(self.S, R, T, self.rs, (game // NP) % NP if dealer is None else dealer)
        E.seed_game(self.rs, seed + 7919, game)
        self.Dd = np.zeros(E.SLEN, np.int64)
        self.Da = np.zeros(E.SLEN, np.int64)
        self.cand = np.zeros(64, np.int64)
        self.wins = np.zeros(64, np.int64)
        self.sd = np.zeros(64, np.int64)
        self.sd2 = np.zeros(64, np.int64)
        self.outc = np.zeros(64, np.int64)
        self.buf = np.zeros(E.NTMAX, np.int64)
        self.pool = np.zeros(E.NTMAX, np.int64)
        self.rrs = np.zeros(1, np.uint64)
        self.info = np.zeros(4, np.int64)
        self.events = [self.snap('start')]

    def hand(self, p):
        S = self.S
        out = []
        for c in range(self.R[E.R_NT]):
            out += [c] * int(S[E.OFF_H + p * E.NTMAX + c])
        return out

    def snap(self, kind, **kw):
        S = self.S
        NP = int(S[E.H_NP])
        d = dict(kind=kind, cur=int(S[E.H_CUR]), phase=int(S[E.H_PHASE]), top=int(S[E.H_TOP]), color=int(S[E.H_COL]),
                 dir=int(S[E.H_DIR]), deck=int(S[E.H_DN]), discard=int(S[E.H_XN]), pend=int(S[E.H_PEND]),
                 drawn=int(S[E.H_DRAWN]), hands=[self.hand(p) for p in range(NP)], win=int(S[E.H_WIN]),
                 void=[[int(S[E.OFF_K + p * E.NF + k]) for k in range(4)] for p in range(NP)])
        d.update(kw)
        return d

    def step(self):
        S, R = self.S, self.R
        p = int(S[E.H_CUR])
        pol = int(self.seats[p])
        before = self.snap('decision')
        n = E.legal_list(S, R, self.T, p, self.buf) if S[E.H_PHASE] != E.PH_STARTCOL else 0
        before['legal'] = [int(x) for x in self.buf[:n]]
        if pol in (E.P_SEARCH, E.P_CHEAT):
            a = E.search(S, R, self.T, p, self.K, 1 if pol == E.P_CHEAT else 0, self.rpol, self.opol, self.z, self.rs,
                         self.X, self.Dd, self.Da, self.cand, self.wins, self.sd, self.sd2, self.outc, self.buf,
                         self.pool, self.rrs, self.info)
            nc = int(self.info[0])
            if nc > 1:
                before['options'] = [(int(self.cand[i]), int(self.wins[i]) / self.K) for i in range(nc)]
                before['default_kept'] = int(self.info[2])
        else:
            a = E.fast_policy(S, R, self.T, pol, self.rs, self.buf, self.X)
        before['action'] = int(a)
        before['policy'] = pol
        self.events.append(before)
        E.apply(S, R, self.T, a, self.rs)
        self.events.append(self.snap('after', by=p, action=int(a)))
        return S[E.H_PHASE] != E.PH_OVER

    def play(self):
        while self.step():
            pass
        return int(self.S[E.H_WIN])


def act_name(T, a):
    if a < 0:
        return 'DRAW'
    c = a >> 2
    if T[c, 2] >= E.K_WILD:
        return f'{E.card_name(T, c)} -> {E.COLORS[a & 3]}'
    return E.card_name(T, c)
