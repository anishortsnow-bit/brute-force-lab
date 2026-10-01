"""Ciclos infinitos no War com baralhos simplificados.

Baralho = `suits` naipes x `ranks` valores (1..ranks). Distribuição alternada (carta 0 para A, 1 para B, ...).
Devolução determinística: order 0 = cartas do vencedor primeiro, order 1 = cartas do perdedor primeiro.
Uma partida que passa de `cap` rodadas é contada como ciclo (o estado é finito e a regra é determinística,
então ela nunca termina). `find_cycle` confirma exatamente para um exemplo, guardando os estados vistos.
"""
import numpy as np
from numba import njit, prange


@njit(cache=True)
def play(deck, war_down, order, cap):
    n = deck.shape[0]
    a = np.empty(n, np.int8); b = np.empty(n, np.int8)
    ha = 0; hb = 0; na = 0; nb = 0
    for i in range(n):
        if i % 2 == 0:
            a[na] = deck[i]; na += 1
        else:
            b[nb] = deck[i]; nb += 1
    pa = np.empty(n, np.int8); pb = np.empty(n, np.int8); won = np.empty(n, np.int8)
    snap = np.empty(n + 1, np.int8); cur = np.empty(n + 1, np.int8)
    snap_len = _encode(a, ha, na, b, hb, nb, n, snap)
    power = 1; lam = 0
    rounds = 0
    while na > 0 and nb > 0:
        rounds += 1
        if rounds > cap:
            return -2
        ka = 0; kb = 0
        ca = a[ha]; ha = (ha + 1) % n; na -= 1
        cb = b[hb]; hb = (hb + 1) % n; nb -= 1
        pa[ka] = ca; ka += 1; pb[kb] = cb; kb += 1
        while ca == cb:
            need = war_down + 1
            if na < need or nb < need:
                return rounds
            for _ in range(war_down):
                pa[ka] = a[ha]; ha = (ha + 1) % n; na -= 1; ka += 1
                pb[kb] = b[hb]; hb = (hb + 1) % n; nb -= 1; kb += 1
            ca = a[ha]; ha = (ha + 1) % n; na -= 1
            cb = b[hb]; hb = (hb + 1) % n; nb -= 1
            pa[ka] = ca; ka += 1; pb[kb] = cb; kb += 1
        a_wins = ca > cb
        t = 0
        if (order == 0) == a_wins:
            for i in range(ka):
                won[t] = pa[i]; t += 1
            for i in range(kb):
                won[t] = pb[i]; t += 1
        else:
            for i in range(kb):
                won[t] = pb[i]; t += 1
            for i in range(ka):
                won[t] = pa[i]; t += 1
        if a_wins:
            for i in range(t):
                a[(ha + na) % n] = won[i]; na += 1
        else:
            for i in range(t):
                b[(hb + nb) % n] = won[i]; nb += 1
        # Brent: compara o estado atual com o instantâneo; ciclo => a partida nunca termina
        lam += 1
        cl = _encode(a, ha, na, b, hb, nb, n, cur)
        if cl == snap_len:
            same = True
            for i in range(cl):
                if cur[i] != snap[i]:
                    same = False; break
            if same:
                return -lam  # negativo = ciclo; valor absoluto = comprimento do ciclo
        if lam == power:
            for i in range(cl):
                snap[i] = cur[i]
            snap_len = cl; power *= 2; lam = 0
    return rounds


@njit(cache=True)
def _encode(a, ha, na, b, hb, nb, n, out):
    k = 0
    for i in range(na):
        out[k] = a[(ha + i) % n]; k += 1
    out[k] = -1; k += 1
    for i in range(nb):
        out[k] = b[(hb + i) % n]; k += 1
    return k


@njit(parallel=True, cache=True)
def loop_rate(suits, ranks, war_down, order, n_games, seed, cap):
    n = suits * ranks
    base = np.empty(n, np.int8)
    for r in range(ranks):
        for s in range(suits):
            base[r * suits + s] = r + 1
    loops = np.zeros(n_games, np.int8)
    lengths = np.zeros(n_games, np.int32)
    for g in prange(n_games):
        st = (seed * 1000003 + g * 2654435761 + 12345) & 0xFFFFFFFFFFFFFFFF
        deck = base.copy()
        for i in range(n - 1, 0, -1):
            st = (st * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
            j = (st >> 33) % (i + 1)
            tmp = deck[i]; deck[i] = deck[j]; deck[j] = tmp
        r = play(deck, war_down, order, cap)
        if r == -2:
            loops[g] = 2  # passou do limite sem fechar ciclo (indeterminado)
        elif r < 0:
            loops[g] = 1
            lengths[g] = -r  # comprimento do ciclo
        else:
            lengths[g] = r
    return loops, lengths


def find_cycle(deck, war_down=3, order=0, cap=1_000_000):
    """Simulação exata em Python: devolve (rodada em que o ciclo começa, comprimento do ciclo) ou None."""
    deck = list(deck)
    a = deck[0::2]; b = deck[1::2]
    seen = {}
    rounds = 0
    while a and b and rounds < cap:
        key = (tuple(a), tuple(b))
        if key in seen:
            return seen[key], rounds - seen[key]
        seen[key] = rounds
        rounds += 1
        pa = [a.pop(0)]; pb = [b.pop(0)]
        while pa[-1] == pb[-1]:
            if len(a) < war_down + 1 or len(b) < war_down + 1:
                return None
            for _ in range(war_down + 1):
                pa.append(a.pop(0)); pb.append(b.pop(0))
        a_wins = pa[-1] > pb[-1]
        won = (pa + pb) if ((order == 0) == a_wins) else (pb + pa)
        (a if a_wins else b).extend(won)
    return None
