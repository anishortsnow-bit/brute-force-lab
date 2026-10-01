"""Conta cadeias de guerras (guerra dupla, tripla...) no War real: 52 cartas, guerra com 3 para baixo, devolução aleatória."""
import numpy as np
from numba import njit, prange


@njit(cache=True)
def _play_chains(deck, war_down, rng_state, hist):
    a = np.empty(52, np.int8); b = np.empty(52, np.int8)
    ha = 0; na = 26; hb = 0; nb = 26
    for i in range(26):
        a[i] = deck[2 * i]; b[i] = deck[2 * i + 1]
    pot = np.empty(52, np.int8)
    s = rng_state; battles = 0
    while na > 0 and nb > 0:
        battles += 1
        if battles > 100000:
            return battles
        k = 0
        ca = a[ha]; ha = (ha + 1) % 52; na -= 1
        cb = b[hb]; hb = (hb + 1) % 52; nb -= 1
        pot[k] = ca; pot[k + 1] = cb; k += 2
        chain = 0
        while ca == cb:
            chain += 1
            if na < war_down + 1 or nb < war_down + 1:
                hist[min(chain, 19)] += 1
                return battles
            for _ in range(war_down):
                pot[k] = a[ha]; ha = (ha + 1) % 52; na -= 1; k += 1
                pot[k] = b[hb]; hb = (hb + 1) % 52; nb -= 1; k += 1
            ca = a[ha]; ha = (ha + 1) % 52; na -= 1
            cb = b[hb]; hb = (hb + 1) % 52; nb -= 1
            pot[k] = ca; pot[k + 1] = cb; k += 2
        hist[min(chain, 19)] += 1
        for i in range(k - 1, 0, -1):
            s = (s * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
            j = (s >> 33) % (i + 1)
            t = pot[i]; pot[i] = pot[j]; pot[j] = t
        if ca > cb:
            for i in range(k):
                a[(ha + na) % 52] = pot[i]; na += 1
        else:
            for i in range(k):
                b[(hb + nb) % 52] = pot[i]; nb += 1
    return battles


@njit(parallel=True, cache=True)
def chains(n_games, war_down, seed, n_threads):
    hists = np.zeros((n_threads, 20), np.int64)
    maxchain = np.zeros(n_games, np.int8)
    base = np.empty(52, np.int8)
    for v in range(13):
        for s_ in range(4):
            base[v * 4 + s_] = v + 2
    per = n_games // n_threads
    for t in prange(n_threads):
        h = np.zeros(20, np.int64)
        for g in range(t * per, (t + 1) * per):
            st = (seed * 1000003 + g * 2654435761 + 12345) & 0xFFFFFFFFFFFFFFFF
            deck = base.copy()
            for i in range(51, 0, -1):
                st = (st * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
                j = (st >> 33) % (i + 1)
                tmp = deck[i]; deck[i] = deck[j]; deck[j] = tmp
            before = h.copy()
            _play_chains(deck, war_down, st ^ 0x9E3779B97F4A7C15, h)
            m = 0
            for c in range(19, 0, -1):
                if h[c] > before[c]:
                    m = c; break
            maxchain[g] = m
        hists[t] = h
    return hists.sum(axis=0), maxchain
