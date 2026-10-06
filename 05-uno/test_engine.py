"""Tests for the UNO engine: deck, rules, invariants, an independent legality check, the search's sampling,
symmetry, and the replication of Sidajaya et al. (2024). Run: python test_engine.py"""
import numpy as np
import engine as E

T = E.uno_table()
X = E.expert_flags()
FAILS = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)
        print('  FAIL', msg)


def hand(S, p, nt=54):
    return [int(S[E.OFF_H + p * E.NTMAX + c]) for c in range(nt)]


# ----------------------------------------------------------------------------- independent rule checks
def legal_py(S, R, T, p):
    """Legal card types, written from the rule sheet without the engine's masks."""
    ph = S[E.H_PHASE]
    h = hand(S, p, R[E.R_NT])
    if ph == E.PH_AFTER:
        return [int(S[E.H_DRAWN])]
    col, sym = S[E.H_COL], S[E.H_SYM]
    has_col = any(h[c] and T[c, 0] == col for c in range(R[E.R_NT]))
    out = []
    for c in range(R[E.R_NT]):
        if not h[c]:
            continue
        kind = T[c, 2]
        if kind == E.K_WILD:
            ok = True
        elif kind == E.K_WD4:
            ok = (not R[E.R_WD4R]) or not has_col
        else:
            ok = T[c, 0] == col or (sym >= 0 and T[c, 1] == sym)
        if ph == E.PH_STACK:
            ok = ok and kind == S[E.H_PKIND]
        if ok:
            out.append(c)
    return out


def invariants(S, R, T, total):
    NP = S[E.H_NP]
    nt = R[E.R_NT]
    tot = int(S[E.H_DN] + S[E.H_XN])
    for p in range(NP):
        h = hand(S, p, nt)
        tot += sum(h)
        check(S[E.OFF_HS + p] == sum(h), 'hand size')
        m = sum(1 << c for c in range(nt) if h[c])
        check(S[E.OFF_HM + p] == m, 'hand mask')
        for k in range(4):
            check(S[E.OFF_CC + p * 4 + k] == sum(h[c] for c in range(nt) if T[c, 0] == k), 'colour count')
        check(min(h) >= 0, 'negative count')
    check(tot == total, f'card count {tot} != {total}')
    if S[E.H_PHASE] != E.PH_OVER:
        check(S[E.OFF_X + S[E.H_XN] - 1] == S[E.H_TOP], 'top card')


def random_walk(R, T, games, pols, seed, total):
    """Play games step by step from Python, checking invariants and legality at every decision."""
    buf = np.zeros(E.NTMAX, np.int64)
    rs = np.zeros(1, np.uint64)
    S = np.zeros(E.SLEN, np.int64)
    ndec = 0
    lengths = []
    for g in range(games):
        E.seed_game(rs, seed, g)
        E.new_game(S, R, T, rs, g % R[E.R_NP])
        steps = 0
        while S[E.H_PHASE] != E.PH_OVER:
            invariants(S, R, T, total)
            p = S[E.H_CUR]
            if S[E.H_PHASE] != E.PH_STARTCOL:
                n = E.legal_list(S, R, T, p, buf)
                check(sorted(buf[:n].tolist()) == legal_py(S, R, T, p), 'legal list')
                check(n > 0, 'decision without a legal card')
            a = E.fast_policy(S, R, T, pols[p], rs, buf, X)
            if a >= 0 and S[E.H_PHASE] != E.PH_STARTCOL:
                check((a >> 2) in legal_py(S, R, T, p), 'policy played an illegal card')
            E.apply(S, R, T, a, rs)
            ndec += 1
            steps += 1
        invariants(S, R, T, total)
        check(S[E.H_WIN] >= 0 and S[E.OFF_HS + S[E.H_WIN]] == 0, 'winner has cards')
        lengths.append(steps)
    return ndec


# ----------------------------------------------------------------------------- scripted positions
def make_state(R, hands, top, color=None, cur=0, deck_top=(), direction=1):
    """A position with the given hands (lists of card types), top card and current player; every other card
    goes into the deck (deck_top first, so it is drawn first)."""
    S = np.zeros(E.SLEN, np.int64)
    rs = np.zeros(1, np.uint64)
    E.seed_game(rs, 1, 0)
    E.new_game(S, R, T, rs, 0)
    nt = R[E.R_NT]
    left = [int(T[c, 4]) for c in range(nt)]
    for p in range(R[E.R_NP]):
        for c in range(nt):
            S[E.OFF_H + p * E.NTMAX + c] = 0
        for c in hands.get(p, []):
            S[E.OFF_H + p * E.NTMAX + c] += 1
            left[c] -= 1
        E.rebuild_hand(S, T, p, nt)
    left[top] -= 1
    for c in deck_top:
        left[c] -= 1
    deck = [c for c in range(nt) for _ in range(left[c])] + list(reversed(deck_top))
    for i, c in enumerate(deck):
        S[E.OFF_D + i] = c
    S[E.H_DN] = len(deck)
    S[E.OFF_X] = top
    S[E.H_XN] = 1
    S[E.H_TOP] = top
    S[E.H_COL] = T[top, 0] if color is None else color
    S[E.H_SYM] = T[top, 1]
    S[E.H_CUR] = cur
    S[E.H_DIR] = direction
    S[E.H_PEND] = 0
    S[E.H_PKIND] = -1
    S[E.H_PHASE] = E.PH_TURN
    S[E.H_WIN] = -1
    S[E.OFF_K:E.OFF_K + E.NPMAX * E.NF] = 0
    E.advance(S, R, T, rs)
    return S, rs


R_, Y_, G_, B_ = 0, 1, 2, 3


def c(colour, sym):
    return colour * 13 + sym


SKIP, REV, D2 = 10, 11, 12


def scripted():
    R2 = E.official(2)
    R4 = E.official(4)
    # two players: Skip and Reverse let you play again; Draw Two makes the other draw and play comes back
    for sym, name in ((SKIP, 'skip'), (REV, 'reverse')):
        S, rs = make_state(R2, {0: [c(R_, sym), c(R_, 5), c(B_, 1)], 1: [c(G_, 3)] * 3}, top=c(R_, 2))
        E.apply(S, R2, T, c(R_, sym) * 4 + R_, rs)
        check(S[E.H_CUR] == 0 and S[E.H_PHASE] == E.PH_TURN, f'2p {name}: play again')
    S, rs = make_state(R2, {0: [c(R_, D2), c(R_, 5), c(B_, 1)], 1: [c(G_, 3)] * 3}, top=c(R_, 2))
    E.apply(S, R2, T, c(R_, D2) * 4 + R_, rs)
    check(S[E.OFF_HS + 1] == 5 and S[E.H_CUR] == 0, '2p draw two')
    # Wild Draw Four: only without a card of the current colour; the next player draws four and is skipped
    buf = np.zeros(E.NTMAX, np.int64)
    S, rs = make_state(R2, {0: [E.WD4, c(R_, 5), c(B_, 2)], 1: [c(G_, 3)] * 3}, top=c(R_, 2))
    n = E.legal_list(S, R2, T, 0, buf)
    check(E.WD4 not in buf[:n].tolist(), 'WD4 illegal with a card of the colour')
    S, rs = make_state(R2, {0: [E.WD4, c(Y_, 2), c(B_, 5)], 1: [c(G_, 3)] * 3}, top=c(R_, 2))
    n = E.legal_list(S, R2, T, 0, buf)
    check(E.WD4 in buf[:n].tolist() and c(Y_, 2) in buf[:n].tolist(), 'WD4 legal with only a matching number')
    E.apply(S, R2, T, E.WD4 * 4 + B_, rs)
    check(S[E.OFF_HS + 1] == 7 and S[E.H_CUR] == 0 and S[E.H_COL] == B_, '2p WD4: four cards, colour, play returns')
    # four players: Skip skips one, Reverse turns play around, Draw Two
    S, rs = make_state(R4, {0: [c(R_, SKIP), c(R_, 5)], 1: [c(G_, 3)] * 2, 2: [c(R_, 4)] * 2, 3: [c(Y_, 3)] * 2},
                       top=c(R_, 2))
    E.apply(S, R4, T, c(R_, SKIP) * 4 + R_, rs)
    check(S[E.H_CUR] == 2, '4p skip')
    S, rs = make_state(R4, {0: [c(R_, REV), c(R_, 5)], 1: [c(G_, 3)] * 2, 2: [c(R_, 4)] * 2, 3: [c(R_, 3)] * 2},
                       top=c(R_, 2))
    E.apply(S, R4, T, c(R_, REV) * 4 + R_, rs)
    check(S[E.H_CUR] == 3 and S[E.H_DIR] == -1, '4p reverse')
    S, rs = make_state(R4, {0: [c(R_, D2), c(R_, 5)], 1: [c(G_, 3)] * 2, 2: [c(R_, 4)] * 2, 3: [c(R_, 3)] * 2},
                       top=c(R_, 2))
    E.apply(S, R4, T, c(R_, D2) * 4 + R_, rs)
    check(S[E.OFF_HS + 1] == 4 and S[E.H_CUR] == 2, '4p draw two')
    # nothing to play: draw one; a playable drawn card waits for a decision, an unplayable one ends the turn
    S, rs = make_state(R2, {0: [c(G_, 3), c(Y_, 4)], 1: [c(G_, 3)] * 2}, top=c(R_, 2), deck_top=[c(R_, 9)])
    check(S[E.H_PHASE] == E.PH_AFTER and S[E.H_DRAWN] == c(R_, 9) and S[E.OFF_HS] == 3, 'forced draw, playable')
    E.apply(S, R2, T, E.DRAW, rs)
    check(S[E.H_CUR] == 1 and S[E.OFF_HS] == 3, 'keep the drawn card')
    S, rs = make_state(R2, {0: [c(G_, 3), c(Y_, 4)], 1: [c(R_, 3)] * 2}, top=c(R_, 2), deck_top=[c(B_, 9)])
    check(S[E.H_CUR] == 1 and S[E.OFF_HS] == 3 and S[E.H_PHASE] == E.PH_TURN, 'forced draw, unplayable')
    # the draw reveals a missing colour
    check(S[E.OFF_K + 0 * E.NF + R_] == 2 and S[E.OFF_K + E.F_WILD] == 2, 'inference after a draw')
    # draw by choice
    S, rs = make_state(R2, {0: [c(R_, 3), c(Y_, 4)], 1: [c(G_, 3)] * 2}, top=c(R_, 2), deck_top=[c(B_, 9)])
    E.apply(S, R2, T, E.DRAW, rs)
    check(S[E.OFF_HS] == 3 and S[E.H_CUR] == 1, 'voluntary draw')
    # going out on a Draw Two: the other player still draws (and it counts for the score)
    S, rs = make_state(R2, {0: [c(R_, D2)], 1: [c(G_, 3)] * 2}, top=c(R_, 2))
    E.apply(S, R2, T, c(R_, D2) * 4 + R_, rs)
    check(S[E.H_WIN] == 0 and S[E.H_PHASE] == E.PH_OVER and S[E.OFF_HS + 1] == 4, 'last card draw two')
    # progressive UNO: stack Draw Two on Draw Two
    Rs = E.rules(stack=1)
    S, rs = make_state(Rs, {0: [c(R_, D2), c(R_, 5)], 1: [c(G_, D2), c(G_, 4)]}, top=c(R_, 2))
    E.apply(S, Rs, T, c(R_, D2) * 4 + R_, rs)
    check(S[E.H_PHASE] == E.PH_STACK and S[E.H_CUR] == 1 and S[E.H_PEND] == 2, 'stack offered')
    E.apply(S, Rs, T, c(G_, D2) * 4 + G_, rs)
    check(S[E.H_PEND] == 0 and S[E.OFF_HS] == 5 and S[E.H_CUR] == 1, 'stack: four cards for the one who cannot stack')
    # draw until you can play
    Ru = E.rules(drawuntil=1, voldraw=0)
    S, rs = make_state(Ru, {0: [c(G_, 3)], 1: [c(G_, 3)] * 2}, top=c(R_, 2), deck_top=[c(B_, 9), c(Y_, 1), c(R_, 7)])
    check(S[E.H_PHASE] == E.PH_AFTER and S[E.OFF_HS] == 4 and S[E.H_MUST] == 1, 'draw until playable')
    # reshuffle keeps the top card
    S, rs = make_state(R2, {0: [c(G_, 3)], 1: [c(G_, 3)]}, top=c(R_, 2))
    S[E.H_DN] = 0
    for i in range(5):
        S[E.OFF_X + i] = c(Y_, i + 1)
    S[E.OFF_X + 5] = c(R_, 2)
    S[E.H_XN] = 6
    got = E.draw1(S, T, 0, rs)
    check(S[E.H_XN] == 1 and S[E.OFF_X] == c(R_, 2) and S[E.H_DN] == 4 and got in [c(Y_, i + 1) for i in range(5)],
          'reshuffle')


def start_cards():
    """Start-card rules over many deals."""
    R2 = E.official(2)
    R4 = E.official(4)
    S = np.zeros(E.SLEN, np.int64)
    rs = np.zeros(1, np.uint64)
    seen = set()
    for R in (R2, R4):
        NP = R[E.R_NP]
        for g in range(20000):
            E.seed_game(rs, 77, g)
            dealer = g % NP
            E.new_game(S, R, T, rs, dealer)
            top = S[E.OFF_X]
            kind = T[top, 2]
            first = (dealer + 1) % NP
            seen.add(int(kind))
            check(kind != E.K_WD4, 'WD4 start card')
            if kind == E.K_WILD:
                check(S[E.H_PHASE] == E.PH_STARTCOL and S[E.H_CUR] == first, 'wild start')
            elif kind == E.K_D2:
                check(S[E.OFF_HS + first] == 9 and S[E.H_FIRST] == (first + 1) % NP, 'draw two start')
            elif kind == E.K_REV:
                check(S[E.H_FIRST] == dealer and S[E.H_DIR] == -1, 'reverse start')
            elif kind == E.K_SKIP:
                check(S[E.H_FIRST] == (first + 1) % NP, 'skip start')
            else:
                check(S[E.H_FIRST] == first, 'number start')
    check(seen == {0, 1, 2, 3, 4}, 'every start kind seen')


def sampling():
    """Search sampling: hand sizes and card totals kept, own hand untouched, revealed gaps respected."""
    R2 = E.official(2)
    S = np.zeros(E.SLEN, np.int64)
    D = np.zeros(E.SLEN, np.int64)
    rs = np.zeros(1, np.uint64)
    pool = np.zeros(E.NTMAX, np.int64)
    buf = np.zeros(E.NTMAX, np.int64)
    for g in range(3000):
        E.seed_game(rs, 5, g)
        E.new_game(S, R2, T, rs, g % 2)
        for _ in range(g % 25):
            if S[E.H_PHASE] == E.PH_OVER:
                break
            E.apply(S, R2, T, E.fast_policy(S, R2, T, E.P_RANDOM, rs, buf, X), rs)
        if S[E.H_PHASE] == E.PH_OVER:
            continue
        p = S[E.H_CUR]
        q = 1 - p
        E.determinize(S, R2, T, p, 0, D, rs, pool)
        invariants(D, R2, T, 108)
        check(hand(D, p) == hand(S, p), 'own hand kept')
        check(D[E.OFF_HS + q] == S[E.OFF_HS + q], 'opponent hand size kept')
        check(D[E.H_DN] == S[E.H_DN], 'deck size kept')
        # every card the opponent is known to lack (all of the hand) stays out of the sampled hand
        k = E.OFF_K + q * E.NF
        hs = S[E.OFF_HS + q]
        for col in range(4):
            if S[k + col] == hs and hs > 0:
                check(D[E.OFF_HM + q] & D[E.OFF_M + col] == 0, 'known missing colour respected')
        E.determinize(S, R2, T, p, 1, D, rs, pool)
        check(hand(D, q) == hand(S, q), 'cheat keeps the real hand')


def run_batch_checks():
    R2 = E.official(2)
    res = E.run_batch(R2, T, np.array([0, 0], np.int32), 0, 2, 2, 0.0, X, 31, 0, 4_000_000, 192)
    p = (res[:, E.O_WIN] == 1).mean()
    se = np.sqrt(0.25 / len(res))
    print(f'  random v random: {p:.4%} (4 SE = {4 * se:.3%})')
    check(abs(p - 0.5) < 4 * se, 'symmetry')
    check((res[:, E.O_WIN] < 0).sum() == 0, 'unfinished games')
    # Sidajaya et al. (2024): 31.7% (holds wild cards) and 24.7% (plays them at once), Alice vs 3 random bots
    R4 = E.sidajaya(4)
    lo, hi = 1.0, 0.0
    for tie in range(5):
        Xs = E.expert_flags()
        Xs[E.X_SIDATIE] = tie
        r = E.run_batch(R4, T, np.array([E.P_SSMART, 5, 5, 5], np.int32), 0, 2, 2, 0.0, Xs, 41, 0, 2_000_000, 192)
        v = (r[:, E.O_WIN] == 1).mean()
        lo, hi = min(lo, v), max(hi, v)
    r = E.run_batch(R4, T, np.array([E.P_SSTUPID, 5, 5, 5], np.int32), 0, 2, 2, 0.0, X, 43, 0, 4_000_000, 192)
    rev = (r[:, E.O_WIN] == 1).mean()
    print(f'  Sidajaya forward: {lo:.2%} to {hi:.2%} depending on the colour tie order (paper 31.7%); '
          f'reverse {rev:.3%} (paper 24.7%)')
    check(lo < 0.317 < hi, 'Sidajaya forward')
    check(abs(rev - 0.247) < 0.0015, 'Sidajaya reverse')


if __name__ == '__main__':
    tc = T[:, E.T_COP]
    check(tc.sum() == 108 and tc[T[:, 2] == E.K_NUM].sum() == 76 and tc[T[:, 2] == E.K_WD4].sum() == 4
          and all(tc[T[:, 2] == k].sum() == 8 for k in (1, 2, 3)), 'UNO deck')
    check(E.c8_table()[:, E.T_COP].sum() == 52, 'Crazy Eights deck')
    print('scripted rules'); scripted()
    print('start cards'); start_cards()
    print('random walks')
    n = 0
    for R, pols, total in ((E.official(2), [0, 2], 108), (E.official(3), [1, 0, 2], 108), (E.official(4), [2, 2, 0, 1], 108),
                           (E.rules(stack=1), [0, 2], 108), (E.rules(np_=4, stack=1), [0, 2, 1, 0], 108),
                           (E.rules(drawuntil=1, voldraw=0), [0, 1], 108), (E.sidajaya(4), [5, 6, 7, 5], 108)):
        n += random_walk(R, T, 300, pols, 3, total)
    n += random_walk(E.crazy_eights(2), E.c8_table(), 300, [0, 2], 3, 52)
    print(f'  {n} decisions checked')
    print('sampling'); sampling()
    print('long runs'); run_batch_checks()
    print('ALL PASS' if not FAILS else f'{len(FAILS)} FAILURES: {sorted(set(FAILS))}')
