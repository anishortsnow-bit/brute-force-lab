"""Monopoly, full games, standard US board and the official rules (Hasbro rule sheet; 2008 amounts for the two tax squares).

What is in:
  dice, doubles, three doubles to Jail, GO salary, both card decks (16 cards each, drawn in order and returned to the bottom),
  buying at the printed price, auctions when a player declines, rents (double on a complete unimproved color group, railroads,
  utilities), even building, the 32-house / 12-hotel supply, selling buildings back at half price, mortgages with 10% interest,
  Jail (pay $50, card, or roll doubles; forced out on the third turn), bankruptcy to a player or to the Bank.
What is simplified (see README):
  trades are one fixed rule (two players swap one lot each when that completes a color group for both);
  building shortages are first come, first served (no auctions of houses); Get Out of Jail Free cards are not sold;
  a creditor who receives mortgaged lots does not pay the 10% transfer interest.

Players are simple policies (a row of numbers), so one rule can be changed at a time.
Everything is seeded: the same seed gives the same game.
"""
import numpy as np
from numba import njit, prange

# ------------------------------------------------------------------ the board
NAMES = ['GO', 'Mediterranean Avenue', 'Community Chest', 'Baltic Avenue', 'Income Tax', 'Reading Railroad', 'Oriental Avenue', 'Chance',
         'Vermont Avenue', 'Connecticut Avenue', 'Jail / Just Visiting', 'St. Charles Place', 'Electric Company', 'States Avenue',
         'Virginia Avenue', 'Pennsylvania Railroad', 'St. James Place', 'Community Chest', 'Tennessee Avenue', 'New York Avenue',
         'Free Parking', 'Kentucky Avenue', 'Chance', 'Indiana Avenue', 'Illinois Avenue', 'B&O Railroad', 'Atlantic Avenue',
         'Ventnor Avenue', 'Water Works', 'Marvin Gardens', 'Go To Jail', 'Pacific Avenue', 'North Carolina Avenue', 'Community Chest',
         'Pennsylvania Avenue', 'Short Line', 'Chance', 'Park Place', 'Luxury Tax', 'Boardwalk']
GROUP_NAMES = ['brown', 'light blue', 'pink', 'orange', 'red', 'yellow', 'green', 'dark blue', 'railroads', 'utilities']
_GROUPS = [[1, 3], [6, 8, 9], [11, 13, 14], [16, 18, 19], [21, 23, 24], [26, 27, 29], [31, 32, 34], [37, 39], [5, 15, 25, 35], [12, 28]]
_STREETS = {  # square: (price, house cost, rent with 0..4 houses and a hotel)
    1: (60, 50, (2, 10, 30, 90, 160, 250)), 3: (60, 50, (4, 20, 60, 180, 320, 450)),
    6: (100, 50, (6, 30, 90, 270, 400, 550)), 8: (100, 50, (6, 30, 90, 270, 400, 550)), 9: (120, 50, (8, 40, 100, 300, 450, 600)),
    11: (140, 100, (10, 50, 150, 450, 625, 750)), 13: (140, 100, (10, 50, 150, 450, 625, 750)), 14: (160, 100, (12, 60, 180, 500, 700, 900)),
    16: (180, 100, (14, 70, 200, 550, 750, 950)), 18: (180, 100, (14, 70, 200, 550, 750, 950)), 19: (200, 100, (16, 80, 220, 600, 800, 1000)),
    21: (220, 150, (18, 90, 250, 700, 875, 1050)), 23: (220, 150, (18, 90, 250, 700, 875, 1050)), 24: (240, 150, (20, 100, 300, 750, 925, 1100)),
    26: (260, 150, (22, 110, 330, 800, 975, 1150)), 27: (260, 150, (22, 110, 330, 800, 975, 1150)), 29: (280, 150, (24, 120, 360, 850, 1025, 1200)),
    31: (300, 200, (26, 130, 390, 900, 1100, 1275)), 32: (300, 200, (26, 130, 390, 900, 1100, 1275)), 34: (320, 200, (28, 150, 450, 1000, 1200, 1400)),
    37: (350, 200, (35, 175, 500, 1100, 1300, 1500)), 39: (400, 200, (50, 200, 600, 1400, 1700, 2000))}

KIND = np.zeros(40, np.int64)          # 0 nothing, 1 street, 2 railroad, 3 utility, 4 Chance, 5 Community Chest, 6 tax, 7 Go To Jail
GROUP = np.full(40, -1, np.int64)
PRICE = np.zeros(40, np.int64)
HCOST = np.zeros(40, np.int64)
RENT = np.zeros((40, 6), np.int64)
TAX = np.zeros(40, np.int64)
GMEM = np.full((10, 4), -1, np.int64)
GSIZE = np.zeros(10, np.int64)
for _g, _m in enumerate(_GROUPS):
    GSIZE[_g] = len(_m)
    for _k, _s in enumerate(_m):
        GMEM[_g, _k] = _s
        GROUP[_s] = _g
for _s, (_p, _h, _r) in _STREETS.items():
    KIND[_s] = 1
    PRICE[_s] = _p
    HCOST[_s] = _h
    RENT[_s] = _r
for _s in (5, 15, 25, 35):
    KIND[_s] = 2
    PRICE[_s] = 200
for _s in (12, 28):
    KIND[_s] = 3
    PRICE[_s] = 150
for _s in (7, 22, 36):
    KIND[_s] = 4
for _s in (2, 17, 33):
    KIND[_s] = 5
KIND[4] = 6
KIND[38] = 6
TAX[4] = 200
TAX[38] = 100
KIND[30] = 7
JAIL = 10

# policy columns
NPOL = 10
P_BUY_RESERVE, P_BUILD_RESERVE, P_MAX_LEVEL, P_JAIL, P_BUY_MASK, P_BID, P_BID_SET, P_RANDOM, P_TRADE, P_SPARE = range(NPOL)
# options
NOPT = 8
O_TRADES, O_JACKPOT, O_NO_AUCTION, O_MAX_ROUNDS, O_START_CASH, O_CHECK, O_SPARE1, O_SPARE2 = range(NOPT)


def policy(**kw):
    """A player. Defaults: buys everything it can afford, builds up to hotels keeping $100, pays to leave Jail at once."""
    p = np.zeros(NPOL)
    p[P_BUY_RESERVE] = 0        # cash it wants left after buying a lot
    p[P_BUILD_RESERVE] = 100    # cash it wants left after building or lifting a mortgage
    p[P_MAX_LEVEL] = 5          # 3 = stop at three houses, 4 = stop at four houses, 5 = hotels
    p[P_JAIL] = 0               # 0 leave at once, 1 stay as long as allowed, 2 stay only once an opponent has buildings
    p[P_BUY_MASK] = 1023        # which groups it buys (bit g = group g)
    p[P_BID] = 1.0              # highest auction bid, as a multiple of the printed price
    p[P_BID_SET] = 2.0          # same, when the lot completes its color group or would complete an opponent's
    p[P_RANDOM] = 0             # > 0: a coin-flip player (buys, builds and leaves Jail with this probability)
    p[P_TRADE] = 1              # 0: never trades
    names = dict(buy_reserve=P_BUY_RESERVE, build_reserve=P_BUILD_RESERVE, max_level=P_MAX_LEVEL, jail=P_JAIL, buy_mask=P_BUY_MASK,
                 bid=P_BID, bid_set=P_BID_SET, random=P_RANDOM, trade=P_TRADE)
    for k, v in kw.items():
        p[names[k]] = v
    return p


def options(**kw):
    o = np.zeros(NOPT)
    o[O_TRADES] = 1             # the set-completing swap rule
    o[O_JACKPOT] = 0            # house rule: 1 = taxes and fines go to Free Parking; 2 = the same, plus $500 from the Bank each time
    o[O_NO_AUCTION] = 0         # house rule: a declined lot stays with the Bank
    o[O_MAX_ROUNDS] = 1000      # a game still running after this many rounds counts as unfinished
    o[O_START_CASH] = 1500
    o[O_CHECK] = 0              # 1: verify the invariants after every turn (slow; for tests)
    names = dict(trades=O_TRADES, jackpot=O_JACKPOT, no_auction=O_NO_AUCTION, max_rounds=O_MAX_ROUNDS, start_cash=O_START_CASH, check=O_CHECK)
    for k, v in kw.items():
        o[names[k]] = v
    return o


# ------------------------------------------------------------------ where the tokens land (no money)
@njit(cache=True)
def movement(seed, nrolls, stay):
    """One token walking the board. Counts the square where each roll ends (index 40 = in Jail, 10 = just visiting).
    stay = 0: pays to leave Jail on the next turn. stay = 1: stays until doubles or the third turn."""
    np.random.seed(seed)
    cnt = np.zeros(41, np.int64)
    ch = np.random.permutation(16)
    cc = np.random.permutation(16)
    pc = 0
    pk = 0
    pos = 0
    jail = 0
    doubles = 0
    for _ in range(nrolls):
        d1 = np.random.randint(1, 7)
        d2 = np.random.randint(1, 7)
        if jail > 0:
            if d1 == d2 or jail == 3 or stay == 0:
                jail = 0
                doubles = 0
                extra = False
            else:
                jail += 1
                cnt[40] += 1
                continue
        else:
            extra = d1 == d2
            if extra:
                doubles += 1
                if doubles == 3:
                    pos = JAIL
                    jail = 1
                    doubles = 0
                    cnt[40] += 1
                    continue
            else:
                doubles = 0
        pos = (pos + d1 + d2) % 40
        while True:
            k = KIND[pos]
            if k == 7:
                pos = JAIL
                jail = 1
                break
            if k == 4:
                c = ch[pc]
                pc = (pc + 1) % 16
                if c == 0:
                    pos = 0
                elif c == 1:
                    pos = 24
                elif c == 2:
                    pos = 11
                elif c == 3:
                    pos = 12 if (pos < 12 or pos >= 28) else 28
                elif c == 4 or c == 5:
                    pos = ((pos + 5) // 10 * 10 + 5) % 40
                elif c == 8:
                    pos -= 3
                    continue
                elif c == 9:
                    pos = JAIL
                    jail = 1
                elif c == 12:
                    pos = 5
                elif c == 13:
                    pos = 39
                break
            if k == 5:
                c = cc[pk]
                pk = (pk + 1) % 16
                if c == 0:
                    pos = 0
                elif c == 5:
                    pos = JAIL
                    jail = 1
                break
            break
        if jail > 0:
            doubles = 0
            cnt[40] += 1
        else:
            cnt[pos] += 1
    return cnt


# ------------------------------------------------------------------ helpers for the full game
@njit(cache=True)
def _count(owner, g, p):
    n = 0
    for k in range(GSIZE[g]):
        if owner[GMEM[g, k]] == p:
            n += 1
    return n


@njit(cache=True)
def _rent(sq, owner, houses, mort, dice):
    """Rent for landing on sq with this dice total (0 if mortgaged)."""
    if mort[sq]:
        return 0
    o = owner[sq]
    k = KIND[sq]
    g = GROUP[sq]
    if k == 1:
        h = houses[sq]
        if h > 0:
            return RENT[sq, h]
        return RENT[sq, 0] * (2 if _count(owner, g, o) == GSIZE[g] else 1)
    if k == 2:
        return 25 * 2 ** (_count(owner, g, o) - 1)
    return dice * (10 if _count(owner, g, o) == 2 else 4)


@njit(cache=True)
def _sell_one(p, cash, owner, houses, bank):
    """Sells one building level of player p (from the cheapest group, the tallest lot first). False if none."""
    best = -1
    for g in range(8):
        if _count(owner, g, p) != GSIZE[g]:
            continue
        top = -1
        for k in range(GSIZE[g]):
            s = GMEM[g, k]
            if houses[s] > 0 and (top < 0 or houses[s] > houses[top]):
                top = s
        if top >= 0 and (best < 0 or HCOST[top] < HCOST[best]):
            best = top
    if best < 0:
        return False
    if houses[best] == 5:
        bank[1] += 1
        if bank[0] >= 4:
            bank[0] -= 4
            houses[best] = 4
            cash[p] += HCOST[best] // 2
        else:                               # no houses left to put back: the whole group is sold down to bare lots
            bank[1] -= 1
            g = GROUP[best]
            for k in range(GSIZE[g]):
                s = GMEM[g, k]
                if houses[s] == 5:
                    bank[1] += 1
                    cash[p] += 5 * HCOST[s] // 2
                else:
                    bank[0] += houses[s]
                    cash[p] += houses[s] * HCOST[s] // 2
                houses[s] = 0
    else:
        houses[best] -= 1
        bank[0] += 1
        cash[p] += HCOST[best] // 2
    return True


@njit(cache=True)
def _raise(p, need, cash, owner, houses, mort, bank):
    """Raises cash until cash[p] >= need: mortgages lots outside complete groups, then sells buildings, then mortgages the rest."""
    for phase in range(3):
        while cash[p] < need:
            if phase == 1:
                if not _sell_one(p, cash, owner, houses, bank):
                    break
                continue
            pick = -1
            for s in range(40):
                if owner[s] != p or mort[s] or PRICE[s] == 0:
                    continue
                g = GROUP[s]
                full = g < 8 and _count(owner, g, p) == GSIZE[g]
                if phase == 0 and full:
                    continue
                if full:
                    built = False
                    for k in range(GSIZE[g]):
                        if houses[GMEM[g, k]] > 0:
                            built = True
                    if built:
                        continue
                if pick < 0 or PRICE[s] < PRICE[pick]:
                    pick = s
            if pick < 0:
                break
            mort[pick] = True
            cash[p] += PRICE[pick] // 2
        if cash[p] >= need:
            return


@njit(cache=True)
def _manage(p, pol, cash, owner, houses, mort, bank, others_built):
    """Before and after the roll: lift mortgages and build, keeping the reserve."""
    reserve = pol[P_BUILD_RESERVE]
    rnd = pol[P_RANDOM]
    maxlv = int(pol[P_MAX_LEVEL])
    while True:
        pick = -1
        pfull = False
        for s in range(40):
            if owner[s] == p and mort[s]:
                g = GROUP[s]
                full = g < 8 and _count(owner, g, p) == GSIZE[g]
                if pick < 0 or (full and not pfull) or (full == pfull and PRICE[s] > PRICE[pick]):
                    pick = s
                    pfull = full
        if pick < 0:
            break
        cost = PRICE[pick] // 2 + (PRICE[pick] // 2 + 9) // 10
        if cash[p] - cost < reserve or (rnd > 0 and np.random.random() >= rnd):
            break
        cash[p] -= cost
        mort[pick] = False
    while True:
        best = -1
        bestv = -1.0
        for g in range(8):
            if _count(owner, g, p) != GSIZE[g]:
                continue
            low = -1
            ok = True
            for k in range(GSIZE[g]):
                s = GMEM[g, k]
                if mort[s]:
                    ok = False
                if low < 0 or houses[s] < houses[low]:
                    low = s
            if not ok or houses[low] >= maxlv:
                continue
            if houses[low] == 4:
                if bank[1] < 1:
                    continue
            elif bank[0] < 1:
                continue
            if cash[p] - HCOST[low] < reserve:
                continue
            v = (RENT[low, houses[low] + 1] - RENT[low, houses[low]]) / HCOST[low]
            if v > bestv:
                bestv = v
                best = low
        if best < 0:
            break
        if rnd > 0 and np.random.random() >= rnd:
            break
        cash[p] -= HCOST[best]
        if houses[best] == 4:
            houses[best] = 5
            bank[1] -= 1
            bank[0] += 4
        else:
            houses[best] += 1
            bank[0] -= 1


@njit(cache=True)
def _auction(sq, first, npl, POL, alive, cash, owner):
    """The Bank auctions sq. Each player has a top bid; the highest wins and pays one dollar over the runner-up."""
    g = GROUP[sq]
    best = -1
    bestb = 0.0
    second = 0.0
    for j in range(npl):
        p = (first + j) % npl
        if not alive[p]:
            continue
        pol = POL[p]
        if pol[P_RANDOM] > 0:
            b = PRICE[sq] * np.random.random() * 1.2
        else:
            if (int(pol[P_BUY_MASK]) >> g) & 1 == 0:
                continue
            f = pol[P_BID]
            mine = _count(owner, g, p)
            if mine == GSIZE[g] - 1:
                f = pol[P_BID_SET]
            else:
                for q in range(npl):
                    if q != p and alive[q] and _count(owner, g, q) == GSIZE[g] - 1:
                        f = pol[P_BID_SET]
            b = PRICE[sq] * f
        cap = cash[p] - pol[P_BUY_RESERVE]
        if b > cap:
            b = cap
        if b <= 0:
            continue
        if b > bestb:
            second = bestb
            bestb = b
            best = p
        elif b > second:
            second = b
    if best < 0:
        return
    pay = int(second) + 1
    if pay > bestb:
        pay = int(bestb)
    if pay < 1:
        pay = 1
    cash[best] -= pay
    owner[sq] = best


@njit(cache=True)
def _trades(npl, POL, alive, cash, owner, houses):
    """Two players swap one lot each when that completes a color group for both. The price difference is paid in cash."""
    done = False
    for a in range(npl):
        if not alive[a] or POL[a, P_TRADE] == 0 or POL[a, P_RANDOM] > 0:
            continue
        for b in range(a + 1, npl):
            if not alive[b] or POL[b, P_TRADE] == 0 or POL[b, P_RANDOM] > 0:
                continue
            for ga in range(8):                       # a needs one lot of ga, and b holds it
                if _count(owner, ga, a) != GSIZE[ga] - 1 or _count(owner, ga, b) != 1:
                    continue
                for gb in range(8):                   # b needs one lot of gb, and a holds it
                    if gb == ga or _count(owner, gb, b) != GSIZE[gb] - 1 or _count(owner, gb, a) != 1:
                        continue
                    sa = -1
                    sb = -1
                    for k in range(GSIZE[ga]):
                        if owner[GMEM[ga, k]] == b:
                            sa = GMEM[ga, k]
                    for k in range(GSIZE[gb]):
                        if owner[GMEM[gb, k]] == a:
                            sb = GMEM[gb, k]
                    diff = PRICE[sa] - PRICE[sb]      # a receives sa and gives sb
                    if diff > 0 and cash[a] < diff:
                        continue
                    if diff < 0 and cash[b] < -diff:
                        continue
                    cash[a] -= diff
                    cash[b] += diff
                    owner[sa] = a
                    owner[sb] = b
                    done = True
                    break
    return done


@njit(cache=True)
def _to_bank(p, npl, POL, alive, cash, owner, mort, auction):
    """Player p went bankrupt to the Bank: the lots go back unmortgaged and are auctioned at once."""
    lots = np.zeros(40, np.bool_)
    for s in range(40):
        if owner[s] == p:
            owner[s] = -1
            mort[s] = False
            lots[s] = True
    if auction:
        for s in range(40):
            if lots[s]:
                _auction(s, p, npl, POL, alive, cash, owner)


@njit(cache=True)
def _check(npl, alive, cash, owner, houses, mort, bank):
    """Invariants. Returns 0 if everything is consistent."""
    nh = 0
    nt = 0
    for s in range(40):
        if houses[s] == 5:
            nt += 1
        else:
            nh += houses[s]
        if houses[s] > 0 and (KIND[s] != 1 or mort[s]):
            return 1
        if owner[s] >= 0 and not alive[owner[s]]:
            return 2
        if owner[s] < 0 and (mort[s] or houses[s] > 0):
            return 3
    if nh + bank[0] != 32 or nt + bank[1] != 12:
        return 4
    for g in range(8):
        lo = 9
        hi = 0
        o = owner[GMEM[g, 0]]
        same = True
        anym = False
        for k in range(GSIZE[g]):
            s = GMEM[g, k]
            lo = min(lo, houses[s])
            hi = max(hi, houses[s])
            if owner[s] != o:
                same = False
            if mort[s]:
                anym = True
        if hi - lo > 1:
            return 5
        if hi > 0 and (not same or o < 0 or anym):
            return 6
    for p in range(npl):
        if alive[p] and cash[p] < 0:
            return 7
    return 0


@njit(cache=True)
def play(seed, npl, POL, start_owner, opt, res, rec):
    """One game. res: [winner (-1 unfinished), rounds, first complete group, who completed it, round of that, turns,
    winner's complete groups (bit mask), error code]. Returns the winner."""
    np.random.seed(seed)
    cash = np.full(npl, int(opt[O_START_CASH]), np.int64)
    pos = np.zeros(npl, np.int64)
    jail = np.zeros(npl, np.int64)
    card = np.zeros(npl, np.int64)                 # Get Out of Jail Free cards held (bit 1 Chance, bit 2 Community Chest)
    alive = np.ones(npl, np.bool_)
    owner = np.full(40, -1, np.int64)
    houses = np.zeros(40, np.int64)
    mort = np.zeros(40, np.bool_)
    bank = np.array([32, 12], np.int64)
    for s in range(40):
        owner[s] = start_owner[s]
    ch = np.random.permutation(16)
    cc = np.random.permutation(16)
    pc = 0
    pk = 0
    ch_out = False
    cc_out = False
    seeded = opt[O_JACKPOT] > 1                    # house rule variant: Free Parking also starts with $500, and again after each win
    pot = 500 if seeded else 0
    trades = opt[O_TRADES] > 0
    jackpot = opt[O_JACKPOT] > 0
    no_auction = opt[O_NO_AUCTION] > 0
    max_rounds = int(opt[O_MAX_ROUNDS])
    chk = opt[O_CHECK] > 0
    for k in range(8):
        res[k] = -1
    res[7] = 0
    left = npl
    turns = 0
    rounds = 0
    while left > 1 and rounds < max_rounds:
        rounds += 1
        for p in range(npl):
            if not alive[p] or left <= 1:
                continue
            turns += 1
            pol = POL[p]
            built = False
            for s in range(40):
                if houses[s] > 0 and owner[s] != p:
                    built = True
            _manage(p, pol, cash, owner, houses, mort, bank, built)
            doubles = 0
            again = True
            while again and alive[p]:
                again = False
                d1 = np.random.randint(1, 7)
                d2 = np.random.randint(1, 7)
                dice = d1 + d2
                if jail[p] > 0:
                    leave = False
                    rolled_out = False
                    if pol[P_RANDOM] > 0:
                        want_out = np.random.random() < pol[P_RANDOM]
                    else:
                        want_out = pol[P_JAIL] == 0 or (pol[P_JAIL] == 2 and not built)
                    if want_out and card[p] > 0:
                        if card[p] & 1:
                            card[p] &= ~1
                            ch_out = False
                        else:
                            card[p] &= ~2
                            cc_out = False
                        leave = True
                    elif want_out and cash[p] >= 50:
                        cash[p] -= 50
                        pot += 50
                        leave = True
                    elif d1 == d2:
                        leave = True
                        rolled_out = True
                    elif jail[p] == 3:
                        debt = 50
                        if cash[p] < debt:
                            _raise(p, debt, cash, owner, houses, mort, bank)
                        if cash[p] < debt:
                            alive[p] = False              # cannot pay the fine: bankrupt to the Bank
                            left -= 1
                            if card[p] & 1:
                                ch_out = False
                            if card[p] & 2:
                                cc_out = False
                            card[p] = 0
                            cash[p] = 0
                            _to_bank(p, npl, POL, alive, cash, owner, mort, (not no_auction) and left > 1)
                            break
                        cash[p] -= debt
                        pot += debt
                        leave = True
                    if not leave:
                        jail[p] += 1
                        break
                    jail[p] = 0
                    if d1 == d2 and not rolled_out:        # paid or used the card before rolling: doubles count as usual
                        doubles = 1
                        again = True
                else:
                    if d1 == d2:
                        doubles += 1
                        if doubles == 3:
                            pos[p] = JAIL
                            jail[p] = 1
                            break
                        again = True
                new = pos[p] + dice
                if new >= 40:
                    new -= 40
                    cash[p] += 200
                pos[p] = new
                mult = 1
                util10 = False
                debt = 0
                creditor = -1
                while True:                                # resolve the square (cards can move the token)
                    sq = pos[p]
                    k = KIND[sq]
                    if k == 7:
                        pos[p] = JAIL
                        jail[p] = 1
                        again = False
                    elif k == 6:
                        debt = TAX[sq]
                    elif k == 0:
                        if sq == 20 and jackpot:
                            cash[p] += pot
                            pot = 500 if seeded else 0
                    elif k <= 3:
                        o = owner[sq]
                        if o < 0:
                            want = False
                            if pol[P_RANDOM] > 0:
                                want = np.random.random() < pol[P_RANDOM]
                            else:
                                want = (int(pol[P_BUY_MASK]) >> GROUP[sq]) & 1 == 1
                            if want and cash[p] - PRICE[sq] >= pol[P_BUY_RESERVE] and cash[p] >= PRICE[sq]:
                                cash[p] -= PRICE[sq]
                                owner[sq] = p
                            elif not no_auction:
                                _auction(sq, p, npl, POL, alive, cash, owner)
                        elif o != p:
                            r = _rent(sq, owner, houses, mort, dice)
                            if util10 and not mort[sq]:
                                r = 10 * (np.random.randint(1, 7) + np.random.randint(1, 7))
                            debt = r * mult
                            creditor = o
                    elif k == 4:
                        while True:
                            c = ch[pc]
                            pc = (pc + 1) % 16
                            if not (c == 7 and ch_out):
                                break
                        if c == 0:
                            pos[p] = 0
                            cash[p] += 200
                        elif c == 1 or c == 2 or c == 12 or c == 13:
                            dest = 24 if c == 1 else (11 if c == 2 else (5 if c == 12 else 39))
                            if dest < sq:
                                cash[p] += 200
                            pos[p] = dest
                            continue
                        elif c == 3:
                            dest = 12 if (sq < 12 or sq >= 28) else 28
                            if dest < sq:
                                cash[p] += 200
                            pos[p] = dest
                            util10 = True
                            continue
                        elif c == 4 or c == 5:
                            dest = ((sq + 5) // 10 * 10 + 5) % 40
                            if dest < sq:
                                cash[p] += 200
                            pos[p] = dest
                            mult = 2
                            continue
                        elif c == 6:
                            cash[p] += 50
                        elif c == 7:
                            card[p] |= 1
                            ch_out = True
                        elif c == 8:
                            pos[p] = sq - 3
                            continue
                        elif c == 9:
                            pos[p] = JAIL
                            jail[p] = 1
                            again = False
                        elif c == 10:
                            for s in range(40):
                                if owner[s] == p:
                                    debt += 100 if houses[s] == 5 else 25 * houses[s]
                        elif c == 11:
                            debt = 15
                        elif c == 14:
                            for q in range(npl):
                                if q != p and alive[q]:
                                    if cash[p] < 50:
                                        _raise(p, 50, cash, owner, houses, mort, bank)
                                    amt = min(50, cash[p])
                                    cash[p] -= amt
                                    cash[q] += amt
                        elif c == 15:
                            cash[p] += 150
                    else:
                        while True:
                            c = cc[pk]
                            pk = (pk + 1) % 16
                            if not (c == 4 and cc_out):
                                break
                        if c == 0:
                            pos[p] = 0
                            cash[p] += 200
                        elif c == 1:
                            cash[p] += 200
                        elif c == 2 or c == 11:
                            debt = 50
                        elif c == 3:
                            cash[p] += 50
                        elif c == 4:
                            card[p] |= 2
                            cc_out = True
                        elif c == 5:
                            pos[p] = JAIL
                            jail[p] = 1
                            again = False
                        elif c == 6 or c == 9 or c == 15:
                            cash[p] += 100
                        elif c == 7:
                            cash[p] += 20
                        elif c == 8:
                            for q in range(npl):
                                if q != p and alive[q]:
                                    if cash[q] < 10:
                                        _raise(q, 10, cash, owner, houses, mort, bank)
                                    amt = min(10, cash[q])
                                    cash[q] -= amt
                                    cash[p] += amt
                        elif c == 10:
                            debt = 100
                        elif c == 12:
                            cash[p] += 25
                        elif c == 13:
                            for s in range(40):
                                if owner[s] == p:
                                    debt += 115 if houses[s] == 5 else 40 * houses[s]
                        elif c == 14:
                            cash[p] += 10
                    break
                if debt > 0:
                    if cash[p] < debt:
                        _raise(p, debt, cash, owner, houses, mort, bank)
                    if cash[p] >= debt:
                        cash[p] -= debt
                        if creditor >= 0:
                            cash[creditor] += debt
                        else:
                            pot += debt
                    else:                                  # bankrupt
                        while _sell_one(p, cash, owner, houses, bank):
                            pass
                        alive[p] = False
                        left -= 1
                        if creditor >= 0:
                            cash[creditor] += cash[p]
                            for s in range(40):
                                if owner[s] == p:
                                    owner[s] = creditor
                            card[creditor] |= card[p]
                        else:
                            if card[p] & 1:
                                ch_out = False
                            if card[p] & 2:
                                cc_out = False
                            _to_bank(p, npl, POL, alive, cash, owner, mort, (not no_auction) and left > 1)
                        cash[p] = 0
                        card[p] = 0
                        again = False
                if jail[p] > 0:
                    again = False
            if alive[p]:
                _manage(p, pol, cash, owner, houses, mort, bank, built)
            if trades and left > 1:
                _trades(npl, POL, alive, cash, owner, houses)
            if res[2] < 0:
                for g in range(8):
                    o = owner[GMEM[g, 0]]
                    if o >= 0 and _count(owner, g, o) == GSIZE[g]:
                        res[2] = g
                        res[3] = o
                        res[4] = rounds
                        break
            if rec.shape[0] > 1 and turns <= rec.shape[0]:     # history: [turn] = player, 40 owners, 40 building levels, positions, cash
                rec[turns - 1, 0] = p
                for s in range(40):
                    rec[turns - 1, 1 + s] = owner[s]
                    rec[turns - 1, 41 + s] = houses[s] + (10 if mort[s] else 0)
                for q in range(npl):
                    rec[turns - 1, 81 + q] = pos[q] if alive[q] else -1
                    rec[turns - 1, 89 + q] = cash[q]
            if chk:
                e = _check(npl, alive, cash, owner, houses, mort, bank)
                if e and res[7] == 0:
                    res[7] = e
    w = -1
    if left == 1:
        for p in range(npl):
            if alive[p]:
                w = p
    res[0] = w
    res[1] = rounds
    res[5] = turns
    if w >= 0:
        m = 0
        for g in range(8):
            if _count(owner, g, w) == GSIZE[g]:
                m |= 1 << g
        res[6] = m
    return w


@njit(parallel=True, cache=True)
def batch(seed, nchunk, per, npl, POL, start_owner, opt, rotate):
    """nchunk * per games. rotate = 1 rotates the seats, so a policy does not always move first (wins are counted
    by policy row, not by seat). Returns wins by policy, unfinished games, rounds histogram, and first-monopoly tables."""
    W = np.zeros((nchunk, npl), np.int64)             # wins by policy row
    SEAT = np.zeros((nchunk, npl), np.int64)          # wins by seat
    U = np.zeros(nchunk, np.int64)                    # unfinished
    R = np.zeros((nchunk, 1002), np.int64)            # rounds of finished games
    F = np.zeros((nchunk, 8, 3), np.int64)            # first complete group: games, its owner won, games finished
    WG = np.zeros((nchunk, 8), np.int64)              # the winner held this group complete at the end
    ERR = np.zeros(nchunk, np.int64)
    NOMONO = np.zeros(nchunk, np.int64)
    for c in prange(nchunk):
        res = np.zeros(8, np.int64)
        rec = np.zeros((1, 1), np.int64)
        P2 = np.empty((npl, NPOL))
        so = np.empty(40, np.int64)
        for r in range(per):
            n = c * per + r
            shift = n % npl if rotate else 0
            for p in range(npl):
                P2[(p + shift) % npl] = POL[p]
            for s in range(40):
                o = start_owner[s]
                so[s] = (o + shift) % npl if o >= 0 else o
            w = play(seed + n, npl, P2, so, opt, res, rec)
            if res[7] != 0:
                ERR[c] += 1
            if w < 0:
                U[c] += 1
            else:
                W[c, (w - shift) % npl] += 1
                SEAT[c, w] += 1
                R[c, min(1001, res[1])] += 1
                for g in range(8):
                    if (res[6] >> g) & 1:
                        WG[c, g] += 1
            if res[2] >= 0:
                F[c, res[2], 0] += 1
                if w >= 0:
                    F[c, res[2], 2] += 1
                    if w == res[3]:
                        F[c, res[2], 1] += 1
            else:
                NOMONO[c] += 1
    return W, SEAT, U, R, F, WG, ERR, NOMONO
