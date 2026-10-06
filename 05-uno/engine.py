"""UNO engine for experiment 05: how much of UNO is luck?

Official Mattel rules (2008 instruction sheet), 2 to 10 players:
  * 108 cards: per colour one 0, two each of 1-9, Skip, Reverse and Draw Two; four Wild, four Wild Draw Four.
  * Match the colour or the number/symbol of the top card, or play a Wild. A Wild Draw Four is legal only
    when you hold no card of the current colour (every player here follows that rule, so nobody ever needs
    to challenge). You may also decline to play and draw instead.
  * Can't play: draw one card; if it can be played you may play it (only that card).
  * Two players: Reverse and Skip let you play again; after a Draw Two or Wild Draw Four the other player
    draws and play comes back to you.
  * Start card: a Wild Draw Four goes back into the deck; a Wild lets the first player name the colour; a
    Draw Two makes the first player draw two and miss the turn; a Reverse lets the dealer start and turns
    play around; a Skip skips the first player.
  * When the deck runs out, the discard pile (minus the top card) is shuffled into a new deck.
House rules from the same sheet: Progressive UNO (stack Draw Two on Draw Two, Wild Draw Four on Wild
Draw Four), plus "draw until you can play". A Crazy Eights table runs on the same engine.

The game is a state machine. advance() resolves everything automatic (forced draws, penalties) and stops
at the next real decision; apply() takes that decision. A whole game state is one int64 array (hands are
kept both as counts and as bit masks, so legal moves are a few bit operations), and a search can copy a
state in one step.
"""
import numpy as np
from numba import njit, prange

# ----------------------------------------------------------------------------------------- card tables
K_NUM, K_SKIP, K_REV, K_D2, K_WILD, K_WD4 = 0, 1, 2, 3, 4, 5
T_COL, T_SYM, T_KIND, T_PTS, T_COP = 0, 1, 2, 3, 4
COLORS = ('red', 'yellow', 'green', 'blue')
SYMS = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'Skip', 'Reverse', 'Draw Two')
WILD, WD4 = 52, 53


def uno_table():
    """Card type c = colour * 13 + symbol (0-9, 10 Skip, 11 Reverse, 12 Draw Two); 52 Wild, 53 Wild Draw Four."""
    rows = []
    for col in range(4):
        for sym in range(13):
            kind = K_NUM if sym < 10 else (K_SKIP, K_REV, K_D2)[sym - 10]
            rows.append((col, sym, kind, sym if sym < 10 else 20, 1 if sym == 0 else 2))
    rows.append((-1, -1, K_WILD, 50, 4))
    rows.append((-1, -1, K_WD4, 50, 4))
    return np.array(rows, np.int32)


def c8_table():
    """Crazy Eights with a 52-card deck: type = suit * 13 + rank (0 = ace ... 12 = king). The four eights are wild."""
    rows = []
    for suit in range(4):
        for rank in range(13):
            if rank == 7:
                rows.append((-1, -1, K_WILD, 50, 1))
            else:
                rows.append((suit, rank, K_NUM, 1 if rank == 0 else (rank + 1 if rank < 10 else 10), 1))
    return np.array(rows, np.int32)


def card_name(T, c):
    if len(T) == 54:
        if T[c, T_KIND] == K_WILD:
            return 'Wild'
        if T[c, T_KIND] == K_WD4:
            return 'Wild Draw Four'
        return f'{COLORS[T[c, T_COL]]} {SYMS[T[c, T_SYM]]}'
    return f'{"A23456789TJQK"[c % 13]}{"shdc"[c // 13]}'


# ----------------------------------------------------------------------------------------- rules
R_NP, R_HAND, R_DRAWUNTIL, R_VOLDRAW, R_WD4R, R_STACK, R_STARTFX, R_SIDA, R_NT, R_MAXSTEPS = range(10)
NR = 12


def rules(np_=2, hand=7, drawuntil=0, voldraw=1, wd4r=1, stack=0, startfx=1, sida=0, nt=54, maxsteps=20000):
    R = np.zeros(NR, np.int32)
    R[R_NP], R[R_HAND], R[R_DRAWUNTIL], R[R_VOLDRAW], R[R_WD4R] = np_, hand, drawuntil, voldraw, wd4r
    R[R_STACK], R[R_STARTFX], R[R_SIDA], R[R_NT], R[R_MAXSTEPS] = stack, startfx, sida, nt, maxsteps
    return R


def official(np_=2):
    return rules(np_=np_)


def sidajaya(np_=4):
    """The setup of Sidajaya et al. (2024): Wild Draw Four playable any time, no start-card effects,
    a wild start card goes to the bottom of the deck, and nobody draws while holding a playable card."""
    return rules(np_=np_, voldraw=0, wd4r=0, startfx=0, sida=1)


def crazy_eights(np_=2):
    """Crazy Eights (Hoyle/pagat.com): 7 cards each with two players (5 with more), match suit or rank or
    play an eight and name a suit; if you can't play you draw until you can. An eight turned up as the
    start card goes back into the deck."""
    return rules(np_=np_, hand=7 if np_ == 2 else 5, drawuntil=1, voldraw=0, wd4r=1, startfx=0, nt=52)


# ----------------------------------------------------------------------------------------- state layout
(H_NP, H_CUR, H_DIR, H_COL, H_SYM, H_TOP, H_DN, H_XN, H_WIN, H_STEPS, H_PEND, H_PKIND, H_PHASE, H_DRAWN,
 H_DEALER, H_FIRST, H_MUST, H_TURNS) = range(18)
HDR = 20
NTMAX = 56
NPMAX = 10
DMAX = 112
NF = 20                      # inference features: 4 colours, 13 symbols (4 + sym), 17 Wild, 18 Wild Draw Four
F_WILD, F_WD4 = 17, 18
M_WILD, M_WD4, M_KIND = 17, 18, 19   # mask slots: 0-3 colours, 4-16 symbols, then Wild, Wild Draw Four, kinds
OFF_M = HDR                  # card-type bit masks
OFF_H = OFF_M + 28           # hand counts [player][type]
OFF_HS = OFF_H + NPMAX * NTMAX
OFF_HM = OFF_HS + NPMAX      # hand bit masks
OFF_CC = OFF_HM + NPMAX      # colour counts [player][colour]
OFF_D = OFF_CC + NPMAX * 4   # deck, top card at the end
OFF_X = OFF_D + DMAX         # discard pile, top card at the end
OFF_K = OFF_X + DMAX         # ncon[player][feature]: how many of the player's oldest cards are known not to have it
OFF_ST = OFF_K + NPMAX * NF  # per-player counters
ST_TURNS, ST_FORCED, ST_ONE, ST_MULTI, ST_DRAWN, ST_PLAYED, ST_PEN, ST_VOL = range(8)
NST = 8
SLEN = OFF_ST + NPMAX * NST

PH_TURN, PH_AFTER, PH_STACK, PH_STARTCOL, PH_OVER = 0, 1, 2, 3, 5
DRAW = -1                    # action: draw instead of playing / keep the drawn card / take the stacked penalty

P_RANDOM, P_CASUAL, P_EXPERT, P_SEARCH, P_CHEAT, P_SRANDOM, P_SSMART, P_SSTUPID = range(8)
POLICIES = {'random': P_RANDOM, 'casual': P_CASUAL, 'expert': P_EXPERT, 'search': P_SEARCH, 'cheat': P_CHEAT,
            'sida_random': P_SRANDOM, 'sida_smart': P_SSMART, 'sida_stupid': P_SSTUPID}

# expert feature switches (for the ablation); X_SIDATIE: colour tie order of the Sidajaya bot (-1 = random);
# X_SDRAW = 0 stops the search from ever drawing by choice; X_STAGE = 1: two-stage search (see search())
X_HOLDWILD, X_COLOR, X_ATTACK, X_CHAIN, X_VOID, X_POINTS, X_SIDATIE, X_SDRAW, X_STAGE, X_BLOCK = range(10)
NX = 12


def expert_flags(**off):
    X = np.ones(NX, np.int32)
    X[X_SIDATIE] = -1
    X[X_STAGE] = 0
    X[X_BLOCK] = 0
    names = {'holdwild': X_HOLDWILD, 'color': X_COLOR, 'attack': X_ATTACK, 'chain': X_CHAIN, 'void': X_VOID,
             'points': X_POINTS, 'sdraw': X_SDRAW, 'stage': X_STAGE, 'block': X_BLOCK}
    for k, v in off.items():
        X[names[k]] = int(v)
    return X


_DB = np.array([0, 1, 48, 2, 57, 49, 28, 3, 61, 58, 50, 42, 38, 29, 17, 4, 62, 55, 59, 36, 53, 51, 43, 22, 45, 39,
                33, 30, 24, 18, 12, 5, 63, 47, 56, 27, 60, 41, 37, 16, 54, 35, 52, 21, 44, 32, 23, 11, 46, 26, 40, 15,
                34, 20, 31, 10, 25, 14, 19, 9, 13, 8, 7, 6], np.int64)


# ----------------------------------------------------------------------------------------- random numbers (splitmix64)
@njit(inline='always')
def _next(rs):
    s = rs[0] + np.uint64(0x9E3779B97F4A7C15)
    rs[0] = s
    z = (s ^ (s >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


@njit(inline='always')
def _randint(rs, n):
    return np.int64(np.float64(_next(rs) >> np.uint64(11)) * (1.0 / 9007199254740992.0) * n)


@njit(inline='always')
def seed_game(rs, seed, game):
    rs[0] = np.uint64(seed) * np.uint64(0x9E3779B97F4A7C15) + np.uint64(game) * np.uint64(0xD1B54A32D192ED03)
    _next(rs)
    _next(rs)


@njit(inline='always')
def lowbit(m):
    """Index of the lowest set bit of m (m > 0)."""
    b = m & -m
    return _DB[np.int64((np.uint64(b) * np.uint64(0x03F79D71B4CB0A89)) >> np.uint64(58))]


# ----------------------------------------------------------------------------------------- hands
@njit(inline='always')
def add_card(S, T, p, c):
    S[OFF_H + p * NTMAX + c] += 1
    S[OFF_HS + p] += 1
    S[OFF_HM + p] |= np.int64(1) << c
    col = T[c, 0]
    if col >= 0:
        S[OFF_CC + p * 4 + col] += 1


@njit(inline='always')
def remove_card(S, T, p, c):
    i = OFF_H + p * NTMAX + c
    S[i] -= 1
    if S[i] == 0:
        S[OFF_HM + p] &= ~(np.int64(1) << c)
    S[OFF_HS + p] -= 1
    col = T[c, 0]
    if col >= 0:
        S[OFF_CC + p * 4 + col] -= 1


@njit
def rebuild_hand(S, T, p, nt):
    """Recompute size, bit mask and colour counts of p's hand from the card counts."""
    base = OFF_H + p * NTMAX
    hs = 0
    hm = np.int64(0)
    for k in range(4):
        S[OFF_CC + p * 4 + k] = 0
    for c in range(nt):
        m = S[base + c]
        if m > 0:
            hs += m
            hm |= np.int64(1) << c
            col = T[c, 0]
            if col >= 0:
                S[OFF_CC + p * 4 + col] += m
    S[OFF_HS + p] = hs
    S[OFF_HM + p] = hm


# ----------------------------------------------------------------------------------------- basic moves
@njit(inline='always')
def nextp(S, p):
    n = S[H_NP]
    return (p + S[H_DIR] + n) % n


@njit
def shuffle_range(S, off, n, rs):
    for i in range(n - 1, 0, -1):
        j = _randint(rs, i + 1)
        t = S[off + i]
        S[off + i] = S[off + j]
        S[off + j] = t


@njit
def draw1(S, T, p, rs):
    """Draw one card for player p, reshuffling the discard pile into a new deck when needed. -1 if none left."""
    if S[H_DN] == 0:
        xn = S[H_XN]
        if xn <= 1:
            return -1
        top = S[OFF_X + xn - 1]
        for i in range(xn - 1):
            S[OFF_D + i] = S[OFF_X + i]
        S[OFF_X] = top
        S[H_XN] = 1
        S[H_DN] = xn - 1
        shuffle_range(S, OFF_D, xn - 1, rs)
    dn = S[H_DN] - 1
    c = S[OFF_D + dn]
    S[H_DN] = dn
    add_card(S, T, p, c)
    S[OFF_ST + p * NST + ST_DRAWN] += 1
    return c


@njit(inline='always')
def has_color(S, p, col):
    return (S[OFF_HM + p] & S[OFF_M + col]) != 0


@njit(inline='always')
def legal_mask(S, R, p, stackonly):
    hm = S[OFF_HM + p]
    colm = S[OFF_M + S[H_COL]]
    m = colm | S[OFF_M + M_WILD]
    sym = S[H_SYM]
    if sym >= 0:
        m |= S[OFF_M + 4 + sym]
    if R[R_WD4R] == 0 or (hm & colm) == 0:
        m |= S[OFF_M + M_WD4]
    m &= hm
    if stackonly:
        m &= S[OFF_M + M_KIND + S[H_PKIND]]
    return m


@njit
def legal_list(S, R, T, p, out):
    """Distinct card types p may play at the current decision, in increasing order."""
    ph = S[H_PHASE]
    if ph == PH_AFTER:
        out[0] = S[H_DRAWN]
        return 1
    m = legal_mask(S, R, p, ph == PH_STACK)
    n = 0
    while m != 0:
        out[n] = lowbit(m)
        n += 1
        m &= m - 1
    return n


@njit(inline='always')
def count_legal(S, R, p, stackonly):
    """Number of distinct legal card types for p (stopping at 2)."""
    m = legal_mask(S, R, p, stackonly)
    if m == 0:
        return 0
    if (m & (m - 1)) == 0:
        return 1
    return 2


@njit
def infer_void(S, R, T, p):
    """Everybody saw p draw: assume p had nothing playable (true for every policy except a deliberate draw)."""
    hs = S[OFF_HS + p]
    k = OFF_K + p * NF
    col = S[H_COL]
    if col >= 0:
        S[k + col] = hs
    sym = S[H_SYM]
    if sym >= 0:
        S[k + 4 + sym] = hs
    S[k + F_WILD] = hs
    S[k + F_WD4] = hs


@njit
def penalty(S, T, p, n, rs):
    for _ in range(n):
        if draw1(S, T, p, rs) >= 0:
            S[OFF_ST + p * NST + ST_PEN] += 1


@njit
def new_game(S, R, T, rs, dealer):
    for i in range(SLEN):
        S[i] = 0
    NP = R[R_NP]
    nt = R[R_NT]
    for c in range(nt):
        b = np.int64(1) << c
        col = T[c, 0]
        if col >= 0:
            S[OFF_M + col] |= b
        if T[c, 1] >= 0:
            S[OFF_M + 4 + T[c, 1]] |= b
        if T[c, 2] == K_WILD:
            S[OFF_M + M_WILD] |= b
        if T[c, 2] == K_WD4:
            S[OFF_M + M_WD4] |= b
        S[OFF_M + M_KIND + T[c, 2]] |= b
    S[H_NP] = NP
    S[H_DIR] = 1
    S[H_WIN] = -1
    S[H_DEALER] = dealer
    S[H_DRAWN] = -1
    S[H_PKIND] = -1
    n = 0
    for c in range(nt):
        for _ in range(T[c, 4]):
            S[OFF_D + n] = c
            n += 1
    S[H_DN] = n
    shuffle_range(S, OFF_D, n, rs)
    for p in range(NP):
        for _ in range(R[R_HAND]):
            draw1(S, T, p, rs)
        S[OFF_ST + p * NST + ST_DRAWN] = 0
    # start card
    while True:
        dn = S[H_DN] - 1
        c = S[OFF_D + dn]
        S[H_DN] = dn
        kind = T[c, 2]
        if not (kind == K_WD4 or (kind == K_WILD and R[R_STARTFX] == 0)):
            break
        pos = 0 if R[R_SIDA] else _randint(rs, dn + 1)
        for i in range(dn, pos, -1):
            S[OFF_D + i] = S[OFF_D + i - 1]
        S[OFF_D + pos] = c
        S[H_DN] = dn + 1
    S[OFF_X] = c
    S[H_XN] = 1
    S[H_TOP] = c
    S[H_COL] = T[c, 0]
    S[H_SYM] = T[c, 1]
    first = (dealer + 1) % NP
    S[H_CUR] = first
    S[H_PHASE] = PH_TURN
    if R[R_STARTFX]:
        kind = T[c, 2]
        if kind == K_WILD:
            S[H_PHASE] = PH_STARTCOL
        elif kind == K_D2:
            penalty(S, T, first, 2, rs)
            S[H_CUR] = nextp(S, first)
        elif kind == K_REV:
            S[H_DIR] = -1
            S[H_CUR] = dealer
        elif kind == K_SKIP:
            S[H_CUR] = nextp(S, first)
    S[H_FIRST] = S[H_CUR]
    advance(S, R, T, rs)


@njit(inline='always')
def drawn_playable(S, R, p, c):
    if c < 0:
        return False
    return (legal_mask(S, R, p, False) >> c) & 1 == 1


@njit
def draw_turn(S, R, T, p, rs):
    """p draws (forced or by choice). Leaves PH_AFTER if the drawn card can be played, otherwise ends the turn."""
    infer_void(S, R, T, p)
    if R[R_DRAWUNTIL]:
        while True:
            c = draw1(S, T, p, rs)
            if c < 0:
                break
            if drawn_playable(S, R, p, c):
                S[H_DRAWN] = c
                S[H_MUST] = 1
                S[H_PHASE] = PH_AFTER
                return
    else:
        c = draw1(S, T, p, rs)
        if drawn_playable(S, R, p, c):
            S[H_DRAWN] = c
            S[H_MUST] = 0
            S[H_PHASE] = PH_AFTER
            return
    S[H_CUR] = nextp(S, p)
    S[H_PHASE] = PH_TURN


@njit
def advance(S, R, T, rs):
    """Resolve everything automatic until a player has a real decision (or the game is over)."""
    while S[H_PHASE] == PH_TURN:
        if S[H_STEPS] > R[R_MAXSTEPS]:
            S[H_PHASE] = PH_OVER
            S[H_WIN] = -1
            return
        p = S[H_CUR]
        if S[H_PEND] > 0:
            S[H_PHASE] = PH_STACK
            if count_legal(S, R, p, True) > 0:
                return
            penalty(S, T, p, S[H_PEND], rs)
            S[H_PEND] = 0
            S[H_PKIND] = -1
            S[H_CUR] = nextp(S, p)
            S[H_PHASE] = PH_TURN
            continue
        S[H_TURNS] += 1
        st = OFF_ST + p * NST
        S[st + ST_TURNS] += 1
        n = count_legal(S, R, p, False)
        if n == 0:
            S[st + ST_FORCED] += 1
            S[H_STEPS] += 1
            draw_turn(S, R, T, p, rs)
            continue
        if n == 1:
            S[st + ST_ONE] += 1
        else:
            S[st + ST_MULTI] += 1
        return


@njit
def play_card(S, R, T, p, c, k, rs):
    remove_card(S, T, p, c)
    hs = S[OFF_HS + p]
    kb = OFF_K + p * NF
    for f in range(NF):
        if S[kb + f] > hs:
            S[kb + f] = hs
    S[OFF_X + S[H_XN]] = c
    S[H_XN] += 1
    S[H_TOP] = c
    S[OFF_ST + p * NST + ST_PLAYED] += 1
    kind = T[c, 2]
    if kind >= K_WILD:
        S[H_COL] = k
        S[H_SYM] = -1
    else:
        S[H_COL] = T[c, 0]
        S[H_SYM] = T[c, 1]
    S[H_DRAWN] = -1
    S[H_MUST] = 0
    q = nextp(S, p)
    if hs == 0:
        S[H_WIN] = p
        S[H_PHASE] = PH_OVER
        if kind == K_D2 or kind == K_WD4:   # the next player still draws (it counts for the score)
            penalty(S, T, q, S[H_PEND] + (2 if kind == K_D2 else 4), rs)
            S[H_PEND] = 0
        return
    S[H_PHASE] = PH_TURN
    if kind == K_SKIP:
        S[H_CUR] = nextp(S, q)
    elif kind == K_REV:
        if S[H_NP] == 2:
            S[H_CUR] = p
        else:
            S[H_DIR] = -S[H_DIR]
            S[H_CUR] = nextp(S, p)
    elif kind == K_D2 or kind == K_WD4:
        amt = 2 if kind == K_D2 else 4
        if R[R_STACK]:
            S[H_PEND] += amt
            S[H_PKIND] = kind
            S[H_CUR] = q
        else:
            penalty(S, T, q, amt, rs)
            S[H_CUR] = nextp(S, q)
    else:
        S[H_CUR] = q


@njit
def apply(S, R, T, a, rs):
    """Take decision a for the player at S[H_CUR]: a = card * 4 + colour, or DRAW. At PH_STARTCOL, a = colour."""
    ph = S[H_PHASE]
    p = S[H_CUR]
    S[H_STEPS] += 1
    if ph == PH_STARTCOL:
        S[H_COL] = a
        S[H_PHASE] = PH_TURN
        advance(S, R, T, rs)
        return
    if a < 0:
        if ph == PH_TURN:
            S[OFF_ST + p * NST + ST_VOL] += 1
            draw_turn(S, R, T, p, rs)
        elif ph == PH_AFTER:
            S[H_DRAWN] = -1
            S[H_CUR] = nextp(S, p)
            S[H_PHASE] = PH_TURN
        else:  # PH_STACK: take the stacked cards and miss the turn
            penalty(S, T, p, S[H_PEND], rs)
            S[H_PEND] = 0
            S[H_PKIND] = -1
            S[H_CUR] = nextp(S, p)
            S[H_PHASE] = PH_TURN
        advance(S, R, T, rs)
        return
    play_card(S, R, T, p, a >> 2, a & 3, rs)
    advance(S, R, T, rs)


# ----------------------------------------------------------------------------------------- simple policies
@njit
def best_color(S, T, p, rs, q, usevoid):
    """The colour p holds most of (ties: a colour the next player q is known to lack, then random)."""
    best = -1.0
    pick = 0
    nb = 0
    for k in range(4):
        v = float(S[OFF_CC + p * 4 + k])
        if usevoid and q >= 0 and S[OFF_K + q * NF + k] > 0:
            v += 0.5
        if v > best:
            best = v
            pick = k
            nb = 1
        elif v == best:
            nb += 1
            if _randint(rs, nb) == 0:
                pick = k
    return pick


@njit
def pick_weighted(S, p, buf, n, rs, kmin, kmax, T):
    """Random card from p's hand among buf[:n] whose kind lies in [kmin, kmax], weighted by copies held."""
    base = OFF_H + p * NTMAX
    tot = 0
    for i in range(n):
        kd = T[buf[i], 2]
        if kd >= kmin and kd <= kmax:
            tot += S[base + buf[i]]
    if tot == 0:
        return -1
    r = _randint(rs, tot)
    for i in range(n):
        kd = T[buf[i], 2]
        if kd >= kmin and kd <= kmax:
            r -= S[base + buf[i]]
            if r < 0:
                return buf[i]
    return buf[n - 1]


@njit(inline='always')
def act_of(T, c, color):
    if T[c, 2] >= K_WILD:
        return c * 4 + color
    return c * 4 + T[c, 0]


@njit
def pol_random(S, R, T, p, buf, n, rs):
    """Any legal card from the hand, at random; a random colour for a wild card."""
    c = pick_weighted(S, p, buf, n, rs, 0, 9, T)
    return act_of(T, c, _randint(rs, 4))


@njit
def pol_casual(S, R, T, p, buf, n, rs):
    """The usual advice: keep your wild cards for when nothing else fits, and name the colour you hold most of."""
    c = pick_weighted(S, p, buf, n, rs, 0, K_D2, T)
    if c < 0:
        c = pick_weighted(S, p, buf, n, rs, K_WILD, K_WD4, T)
    if T[c, 2] >= K_WILD:
        return c * 4 + best_color(S, T, p, rs, -1, False)
    return act_of(T, c, 0)


@njit
def pol_expert(S, R, T, p, buf, n, rs, X):
    """Hand-made strategy: every legal card gets a score, the best one is played."""
    NP = S[H_NP]
    base = OFF_H + p * NTMAX
    hm = S[OFF_HM + p]
    hs = S[OFF_HS + p]
    q = nextp(S, p)
    qs = S[OFF_HS + q]
    qk = OFF_K + q * NF
    nonwild = False
    for i in range(n):
        if T[buf[i], 2] < K_WILD:
            nonwild = True
    # blocking: the next player has shown they hold none of the current colour; if every card I could play
    # changes the colour, draw instead (keep the drawn card too, unless it keeps the colour)
    if X[X_BLOCK] and hs > 1 and qs > 0 and S[H_COL] >= 0 and S[qk + S[H_COL]] >= qs:
        keeps = False
        for i in range(n):
            if T[buf[i], 0] == S[H_COL]:
                keeps = True
        if not keeps:
            if S[H_PHASE] == PH_TURN and R[R_VOLDRAW]:
                return DRAW
            if S[H_PHASE] == PH_AFTER and S[H_MUST] == 0:
                return DRAW
    best = -1e18
    pick = 0
    nb = 0
    for i in range(n):
        c = buf[i]
        if hs == 1:
            return act_of(T, c, 0)
        kind = T[c, 2]
        # how well can I follow on the colour I leave behind? (the card itself already played)
        if kind >= K_WILD:
            col = best_color(S, T, p, rs, q, X[X_VOID] == 1) if X[X_COLOR] else _randint(rs, 4)
            follow = float(S[OFF_CC + p * 4 + col])
        else:
            col = T[c, 0]
            follow = float(S[OFF_CC + p * 4 + col] - 1)
            ms = hm & S[OFF_M + 4 + T[c, 1]] & ~S[OFF_M + col]
            while ms != 0:
                follow += 0.5 * S[base + lowbit(ms)]
                ms &= ms - 1
        s = follow if X[X_COLOR] else 0.0
        if kind >= K_WILD:
            if X[X_HOLDWILD] and nonwild:
                s -= 6.0
            if kind == K_WD4:
                s += 2.0
        again = kind == K_SKIP or kind == K_D2 or kind == K_WD4 or (kind == K_REV and NP == 2)
        if X[X_CHAIN] and NP == 2 and again:
            s += 1.5 if follow >= 1.0 else 0.5
        if X[X_ATTACK]:
            if (kind == K_D2 or kind == K_WD4) and qs <= 2:
                s += 6.0
            elif again and qs <= 1:
                s += 4.0
        if X[X_VOID] and S[qk + col] > 0 and qs > 0:
            s += 2.0 * min(1.0, S[qk + col] / qs)
        if X[X_POINTS]:
            s += T[c, 3] / 100.0
        if s > best + 1e-9:
            best = s
            pick = c * 4 + col
            nb = 1
        elif s > best - 1e-9:
            nb += 1
            if _randint(rs, nb) == 0:
                pick = c * 4 + col
    return pick


@njit
def pol_sida(S, R, T, p, buf, n, rs, pol, tie):
    """The bots of Sidajaya et al. (2024), including how their code weighs moves: every distinct legal
    card is one move and a wild card is four moves (one per colour)."""
    base = OFF_H + p * NTMAX
    nwild = S[base + WILD] + S[base + WD4]
    if pol == P_SSTUPID and nwild > 0:
        nw = 0
        for c in (WILD, WD4):
            if S[base + c] > 0:
                nw += 1
        r = _randint(rs, nw)
        for c in (WILD, WD4):
            if S[base + c] > 0:
                if r == 0:
                    return c * 4 + _randint(rs, 4)
                r -= 1
    if pol == P_SSMART and nwild > 0:
        nnw = 0
        for i in range(n):
            if T[buf[i], 2] < K_WILD:
                nnw += 1
        if nnw > 0:
            r = _randint(rs, nnw)
            for i in range(n):
                if T[buf[i], 2] < K_WILD:
                    if r == 0:
                        return buf[i] * 4 + T[buf[i], 0]
                    r -= 1
        # most common "suit" in the hand, the wild cards counting as a fifth suit
        bestv = -1
        pick = 0
        nb = 0
        for j in range(5):
            # tie < 0: random tie-break; otherwise a fixed priority order (Python's set order in their code)
            k = j if tie < 0 else (tie + j) % 5
            v = S[OFF_CC + p * 4 + k] if k < 4 else nwild
            if v > bestv:
                bestv = v
                pick = k
                nb = 1
            elif v == bestv and tie < 0:
                nb += 1
                if _randint(rs, nb) == 0:
                    pick = k
        col = pick if pick < 4 else _randint(rs, 4)
        nw = 0
        for i in range(n):
            if T[buf[i], 2] >= K_WILD:
                nw += 1
        r = _randint(rs, nw)
        for i in range(n):
            if T[buf[i], 2] >= K_WILD:
                if r == 0:
                    return buf[i] * 4 + col
                r -= 1
    tot = 0
    for i in range(n):
        tot += 4 if T[buf[i], 2] >= K_WILD else 1
    r = _randint(rs, tot)
    for i in range(n):
        w = 4 if T[buf[i], 2] >= K_WILD else 1
        if r < w:
            return buf[i] * 4 + (r if T[buf[i], 2] >= K_WILD else T[buf[i], 0])
        r -= w
    return buf[n - 1] * 4


@njit
def fast_policy(S, R, T, pol, rs, buf, X):
    """Decision of a non-searching policy at the current decision point."""
    p = S[H_CUR]
    ph = S[H_PHASE]
    if ph == PH_STARTCOL:
        if pol == P_RANDOM or pol >= P_SRANDOM:
            return _randint(rs, 4)
        return best_color(S, T, p, rs, nextp(S, p), pol == P_EXPERT and X[X_VOID] == 1)
    n = legal_list(S, R, T, p, buf)
    if pol == P_RANDOM:
        return pol_random(S, R, T, p, buf, n, rs)
    if pol == P_CASUAL:
        return pol_casual(S, R, T, p, buf, n, rs)
    if pol >= P_SRANDOM:
        return pol_sida(S, R, T, p, buf, n, rs, pol, X[X_SIDATIE])
    return pol_expert(S, R, T, p, buf, n, rs, X)


# ----------------------------------------------------------------------------------------- search
@njit
def determinize(S, R, T, p, cheat, D, rs, pool):
    """A full game state consistent with what player p knows: the other hands are dealt at random from
    the cards p has not seen, respecting what their draws revealed; the deck order is random.
    cheat = 1 keeps the real hands (only the deck order stays hidden)."""
    D[:] = S
    nt = R[R_NT]
    NP = S[H_NP]
    if cheat:
        shuffle_range(D, OFF_D, D[H_DN], rs)
        return
    for c in range(nt):
        pool[c] = T[c, 4] - S[OFF_H + p * NTMAX + c]
    for i in range(S[H_XN]):
        pool[S[OFF_X + i]] -= 1
    for q in range(NP):
        if q == p:
            continue
        base = OFF_H + q * NTMAX
        for c in range(nt):
            D[base + c] = 0
        kb = OFF_K + q * NF
        for slot in range(S[OFF_HS + q]):
            tot = 0
            for c in range(nt):
                if pool[c] > 0:
                    ok = True
                    col = T[c, 0]
                    if col >= 0 and slot < S[kb + col]:
                        ok = False
                    sym = T[c, 1]
                    if sym >= 0 and slot < S[kb + 4 + sym]:
                        ok = False
                    kd = T[c, 2]
                    if kd == K_WILD and slot < S[kb + F_WILD]:
                        ok = False
                    if kd == K_WD4 and slot < S[kb + F_WD4]:
                        ok = False
                    if ok:
                        tot += pool[c]
            strict = tot > 0
            if not strict:
                for c in range(nt):
                    tot += pool[c]
            r = _randint(rs, tot)
            for c in range(nt):
                if pool[c] > 0:
                    ok = True
                    if strict:
                        col = T[c, 0]
                        if col >= 0 and slot < S[kb + col]:
                            ok = False
                        sym = T[c, 1]
                        if sym >= 0 and slot < S[kb + 4 + sym]:
                            ok = False
                        kd = T[c, 2]
                        if kd == K_WILD and slot < S[kb + F_WILD]:
                            ok = False
                        if kd == K_WD4 and slot < S[kb + F_WD4]:
                            ok = False
                    if ok:
                        r -= pool[c]
                        if r < 0:
                            D[base + c] += 1
                            pool[c] -= 1
                            break
        rebuild_hand(D, T, q, nt)
    n = 0
    for c in range(nt):
        for _ in range(pool[c]):
            D[OFF_D + n] = c
            n += 1
    D[H_DN] = n
    shuffle_range(D, OFF_D, n, rs)


@njit
def candidates(S, R, T, p, cand, buf, X):
    """Every option at the current decision (each colour of a wild card counts as its own option)."""
    ph = S[H_PHASE]
    if ph == PH_STARTCOL:
        for k in range(4):
            cand[k] = k
        return 4
    n = legal_list(S, R, T, p, buf)
    hs = S[OFF_HS + p]
    nc = 0
    for i in range(n):
        c = buf[i]
        if T[c, 2] >= K_WILD:
            if hs == 1:
                cand[nc] = c * 4
                nc += 1
            else:
                for k in range(4):
                    cand[nc] = c * 4 + k
                    nc += 1
        else:
            cand[nc] = c * 4 + T[c, 0]
            nc += 1
    if ph == PH_TURN:
        if R[R_VOLDRAW] and X[X_SDRAW]:
            cand[nc] = DRAW
            nc += 1
    elif ph == PH_AFTER:
        if S[H_MUST] == 0:
            cand[nc] = DRAW
            nc += 1
    else:
        cand[nc] = DRAW
        nc += 1
    return nc


@njit
def rollout(D, R, T, p, rpol, opol, rs, buf, X):
    """Finish the game: p plays rpol, everybody else opol."""
    while D[H_PHASE] != PH_OVER:
        a = fast_policy(D, R, T, rpol if D[H_CUR] == p else opol, rs, buf, X)
        apply(D, R, T, a, rs)
    return D[H_WIN]


@njit
def search(S, R, T, p, K, cheat, rpol, opol, z, rs, X, Dd, Da, cand, wins, sd, sd2, outc, buf, pool, rrs, info):
    """Determinized Monte Carlo search. For K random deals consistent with what p knows, every option is
    tried and the rest of the game is played out (p with rpol, the others with opol); all options share
    the same deals and the same random seeds, so they are compared on equal terms. The default is the
    option the hand-made strategy would pick; another option replaces it only if it wins more often by
    at least z standard errors of the paired difference (z = 0: plain majority)."""
    nc = candidates(S, R, T, p, cand, buf, X)
    info[0] = nc
    info[1] = 0
    info[2] = 0
    if nc == 1:
        return cand[0]
    rrs[0] = _next(rs)
    ad = fast_policy(S, R, T, P_EXPERT, rrs, buf, X)
    dflt = 0
    for i in range(nc):
        if cand[i] == ad:
            dflt = i
        wins[i] = 0
        sd[i] = 0
        sd2[i] = 0
    # two-stage mode: every option gets K/4 deals, then only the best three (and the default) get the rest
    K1 = K // 4 if X[X_STAGE] else K
    for i in range(nc):
        outc[i] = 1
    act = np.ones(64, np.bool_)
    for k in range(K):
        if k == K1 and K1 < K:
            nkeep = 0
            for i in range(nc):
                act[i] = False
            for _ in range(3):
                bi = -1
                for i in range(nc):
                    if not act[i] and (bi < 0 or wins[i] > wins[bi]):
                        bi = i
                if bi >= 0:
                    act[bi] = True
            act[dflt] = True
        sk = _next(rs)
        rrs[0] = sk
        determinize(S, R, T, p, cheat, Dd, rrs, pool)
        for i in range(nc):
            if not act[i]:
                continue
            Da[:] = Dd
            rrs[0] = sk ^ np.uint64(0x5DEECE66D)
            apply(Da, R, T, cand[i], rrs)
            outc[i] = 1 if rollout(Da, R, T, p, rpol, opol, rrs, buf, X) == p else 0
        for i in range(nc):
            if not act[i]:
                continue
            wins[i] += outc[i]
            d = outc[i] - outc[dflt]
            sd[i] += d
            sd2[i] += d * d
    if K1 < K:
        for i in range(nc):
            if not act[i]:
                wins[i] = -1
    best = -1
    lo = K + 1
    pick = 0
    nb = 0
    for i in range(nc):
        if wins[i] >= 0 and wins[i] < lo:
            lo = wins[i]
        if wins[i] > best:
            best = wins[i]
            pick = i
            nb = 1
        elif wins[i] == best:
            nb += 1
            if i == dflt or (pick != dflt and _randint(rs, nb) == 0):
                pick = i
    if pick != dflt and z > 0:
        m = sd[pick] / K
        var = sd2[pick] / K - m * m
        if m <= 0 or (var > 0 and m / np.sqrt(var / K) < z):
            pick = dflt
    info[1] = best - lo
    info[2] = 1 if pick == dflt else 0
    return cand[pick]


# ----------------------------------------------------------------------------------------- one game
O_WIN, O_FSEAT, O_FIRST, O_STEPS, O_PTS, O_TURNS, O_FORCED, O_ONE, O_MULTI, O_DRAWN, O_PLAYED, O_PEN, O_VOL, \
    O_W0, O_WD40, O_ACT0, O_SDEC, O_SSPREAD, O_SAGREE, O_WINNER, O_OW0 = range(21)
NOUT = 22


@njit
def hand_points(S, T, p, nt):
    s = 0
    base = OFF_H + p * NTMAX
    for c in range(nt):
        s += S[base + c] * T[c, 3]
    return s


@njit
def play_game(S, R, T, seats, K, rpol, opol, z, rs, X, Dd, Da, cand, wins, sd, sd2, outc, buf, pool, rrs, info,
              focal, out):
    """Play one game from the state already dealt in S. seats[i] = policy of seat i. Fills out[] for the focal seat."""
    nt = R[R_NT]
    sdec = 0
    sspread = 0
    sagree = 0
    while S[H_PHASE] != PH_OVER:
        p = S[H_CUR]
        pol = seats[p]
        if pol == P_SEARCH or pol == P_CHEAT:
            a = search(S, R, T, p, K, 1 if pol == P_CHEAT else 0, rpol, opol, z, rs, X, Dd, Da, cand, wins, sd, sd2,
                       outc, buf, pool, rrs, info)
            if p == focal and info[0] > 1:
                sdec += 1
                sspread += info[1]
                sagree += info[2]
        else:
            a = fast_policy(S, R, T, pol, rs, buf, X)
        apply(S, R, T, a, rs)
    w = S[H_WIN]
    out[O_WINNER] = w
    out[O_WIN] = 1 if w == focal else (0 if w >= 0 else -1)
    out[O_FSEAT] = focal
    out[O_FIRST] = 1 if S[H_FIRST] == focal else 0
    out[O_STEPS] = S[H_STEPS]
    pts = 0
    if w >= 0:
        for q in range(S[H_NP]):
            if q != w:
                pts += hand_points(S, T, q, nt)
    out[O_PTS] = pts
    st = OFF_ST + focal * NST
    out[O_TURNS] = S[st + ST_TURNS]
    out[O_FORCED] = S[st + ST_FORCED]
    out[O_ONE] = S[st + ST_ONE]
    out[O_MULTI] = S[st + ST_MULTI]
    out[O_DRAWN] = S[st + ST_DRAWN]
    out[O_PLAYED] = S[st + ST_PLAYED]
    out[O_PEN] = S[st + ST_PEN]
    out[O_VOL] = S[st + ST_VOL]
    out[O_SDEC] = sdec
    out[O_SSPREAD] = sspread
    out[O_SAGREE] = sagree


@njit
def hand_features(S, T, p, nt, out):
    base = OFF_H + p * NTMAX
    w = 0
    w4 = 0
    act = 0
    for c in range(nt):
        m = S[base + c]
        kd = T[c, 2]
        if kd == K_WILD:
            w += m
        elif kd == K_WD4:
            w4 += m
        elif kd != K_NUM:
            act += m
    out[O_W0] = w + w4
    out[O_WD40] = w4
    out[O_ACT0] = act


@njit(parallel=True)
def run_batch(R, T, lineup, K, rpol, opol, z, X, seed, g0, games, nchunks):
    """games games; lineup[0] is the focal policy, lineup[1:] the others. Game g = g0 + i: the deal is
    shared by the NP games of a block (k = g // NP), and the focal policy takes a different seat in each,
    so every seat and every deal is played by everybody."""
    NP = R[R_NP]
    nt = R[R_NT]
    res = np.zeros((games, NOUT), np.int32)
    per = (games + nchunks - 1) // nchunks
    for ch in prange(nchunks):
        S = np.zeros(SLEN, np.int64)
        Dd = np.zeros(SLEN, np.int64)
        Da = np.zeros(SLEN, np.int64)
        cand = np.zeros(64, np.int64)
        wins = np.zeros(64, np.int64)
        sd = np.zeros(64, np.int64)
        sd2 = np.zeros(64, np.int64)
        outc = np.zeros(64, np.int64)
        buf = np.zeros(NTMAX, np.int64)
        pool = np.zeros(NTMAX, np.int64)
        rrs = np.zeros(1, np.uint64)
        rs = np.zeros(1, np.uint64)
        info = np.zeros(4, np.int64)
        tmp = np.zeros(NOUT, np.int32)
        seats = np.zeros(NPMAX, np.int64)
        for i in range(ch * per, min(games, (ch + 1) * per)):
            g = g0 + i
            k = g // NP
            r = g % NP
            for s in range(NP):
                seats[s] = lineup[(s - r + NP) % NP]
            seed_game(rs, seed, k)
            new_game(S, R, T, rs, k % NP)
            seed_game(rs, seed + 7919, g)
            out = res[i]
            hand_features(S, T, r, nt, out)
            if NP == 2:
                hand_features(S, T, 1 - r, nt, tmp)
                out[O_OW0] = tmp[O_W0]
            play_game(S, R, T, seats, K, rpol, opol, z, rs, X, Dd, Da, cand, wins, sd, sd2, outc, buf, pool, rrs,
                      info, r, out)
    return res


# ----------------------------------------------------------------------------------------- matches to 500 points
M_WIN, M_GAMES, M_FSCORE, M_OSCORE, M_FGAMES = range(5)
NMOUT = 6


@njit(parallel=True)
def run_matches(R, T, lineup, K, rpol, opol, z, X, seed, m0, matches, target, nchunks):
    """Official scoring: the winner of each game scores the points left in the other hands; the first to reach
    target wins the match. The deal passes to the left after every game. The focal policy (lineup[0]) sits in
    seat m % NP of match m, and the first dealer is seat (m // NP) % NP."""
    NP = R[R_NP]
    nt = R[R_NT]
    res = np.zeros((matches, NMOUT), np.int32)
    per = (matches + nchunks - 1) // nchunks
    for ch in prange(nchunks):
        S = np.zeros(SLEN, np.int64)
        Dd = np.zeros(SLEN, np.int64)
        Da = np.zeros(SLEN, np.int64)
        cand = np.zeros(64, np.int64)
        wins = np.zeros(64, np.int64)
        sd = np.zeros(64, np.int64)
        sd2 = np.zeros(64, np.int64)
        outc = np.zeros(64, np.int64)
        buf = np.zeros(NTMAX, np.int64)
        pool = np.zeros(NTMAX, np.int64)
        rrs = np.zeros(1, np.uint64)
        rs = np.zeros(1, np.uint64)
        info = np.zeros(4, np.int64)
        out = np.zeros(NOUT, np.int32)
        seats = np.zeros(NPMAX, np.int64)
        score = np.zeros(NPMAX, np.int64)
        for i in range(ch * per, min(matches, (ch + 1) * per)):
            m = m0 + i
            r = m % NP
            for s in range(NP):
                seats[s] = lineup[(s - r + NP) % NP]
                score[s] = 0
            dealer = (m // NP) % NP
            ng = 0
            fg = 0
            while True:
                seed_game(rs, seed, m * 4096 + ng)
                new_game(S, R, T, rs, dealer)
                play_game(S, R, T, seats, K, rpol, opol, z, rs, X, Dd, Da, cand, wins, sd, sd2, outc, buf, pool, rrs,
                          info, r, out)
                ng += 1
                w = S[H_WIN]
                if w >= 0:
                    score[w] += out[O_PTS]
                    if w == r:
                        fg += 1
                    if score[w] >= target:
                        break
                dealer = (dealer + 1) % NP
                if ng > 1000:
                    break
            best = 0
            for s in range(NP):
                if score[s] > score[best]:
                    best = s
            res[i, M_WIN] = 1 if best == r else 0
            res[i, M_GAMES] = ng
            res[i, M_FSCORE] = score[r]
            res[i, M_OSCORE] = score[(r + 1) % NP]
            res[i, M_FGAMES] = fg
    return res
