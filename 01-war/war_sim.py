"""Simulação do jogo de cartas War (Guerra) com numba.

Regras (parametrizáveis):
- 52 cartas, valores 2..14, 4 naipes. Cada jogador recebe 26.
- Cada rodada: os dois viram a carta do topo; a maior leva as duas para o fundo do monte.
- Empate = "guerra": cada um põe WAR_DOWN cartas viradas para baixo e 1 para cima; repete enquanto empatar.
- Quem não tem cartas para completar a guerra perde.
- Ordem de devolução das cartas ganhas (ORDER):
    0 = fixa: cartas do vencedor primeiro, depois as do perdedor (na ordem em que foram jogadas)
    1 = fixa: cartas do perdedor primeiro
    2 = aleatória (embaralha o monte ganho)
Saídas por partida: número de rodadas (batalhas), número de guerras, vencedor. Partidas que passam de
MAX_ROUNDS são marcadas como -1 (candidatas a ciclo infinito; confirmadas depois com detecção exata de ciclo).
"""
import numpy as np
from numba import njit, prange

MAX_ROUNDS = 100_000


@njit(cache=True)
def _play(deck, war_down, order, max_rounds, rng_state):
    # filas circulares de 52 posições para cada jogador
    a = np.empty(52, np.int8); b = np.empty(52, np.int8)
    ha = 0; na = 26; hb = 0; nb = 26
    for i in range(26):
        a[i] = deck[2 * i]; b[i] = deck[2 * i + 1]
    pot_a = np.empty(52, np.int8); pot_b = np.empty(52, np.int8)
    rounds = 0; wars = 0
    s = rng_state
    while na > 0 and nb > 0:
        rounds += 1
        if rounds > max_rounds:
            return -1, wars, 0
        ka = 0; kb = 0
        # cartas da batalha
        ca = a[ha]; ha = (ha + 1) % 52; na -= 1
        cb = b[hb]; hb = (hb + 1) % 52; nb -= 1
        pot_a[ka] = ca; ka += 1; pot_b[kb] = cb; kb += 1
        while ca == cb:
            wars += 1
            need = war_down + 1
            if na < need or nb < need:
                # quem não consegue completar a guerra perde
                if na < need and nb < need:
                    return rounds, wars, (1 if na >= nb else 2)  # raro: decide por quem tem mais
                return rounds, wars, (2 if na < need else 1)
            for _ in range(war_down):
                pot_a[ka] = a[ha]; ha = (ha + 1) % 52; na -= 1; ka += 1
                pot_b[kb] = b[hb]; hb = (hb + 1) % 52; nb -= 1; kb += 1
            ca = a[ha]; ha = (ha + 1) % 52; na -= 1
            cb = b[hb]; hb = (hb + 1) % 52; nb -= 1
            pot_a[ka] = ca; ka += 1; pot_b[kb] = cb; kb += 1
        # monta a ordem de devolução
        tot = ka + kb
        won = np.empty(tot, np.int8)
        a_wins = ca > cb
        if order == 0 or order == 2:
            first = pot_a if a_wins else pot_b; kf = ka if a_wins else kb
            second = pot_b if a_wins else pot_a; ks = kb if a_wins else ka
        else:
            first = pot_b if a_wins else pot_a; kf = kb if a_wins else ka
            second = pot_a if a_wins else pot_b; ks = ka if a_wins else kb
        for i in range(kf):
            won[i] = first[i]
        for i in range(ks):
            won[kf + i] = second[i]
        if order == 2:
            for i in range(tot - 1, 0, -1):
                s = (s * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
                j = (s >> 33) % (i + 1)
                t = won[i]; won[i] = won[j]; won[j] = t
        if a_wins:
            for i in range(tot):
                a[(ha + na) % 52] = won[i]; na += 1
        else:
            for i in range(tot):
                b[(hb + nb) % 52] = won[i]; nb += 1
    return rounds, wars, (1 if na > 0 else 2)


@njit(parallel=True, cache=True)
def simulate(n_games, war_down, order, seed, max_rounds):
    rounds = np.empty(n_games, np.int32)
    wars = np.empty(n_games, np.int32)
    winner = np.empty(n_games, np.int8)
    aces_a = np.empty(n_games, np.int8)
    base = np.empty(52, np.int8)
    for v in range(13):
        for s in range(4):
            base[v * 4 + s] = v + 2
    for g in prange(n_games):
        st = (seed * 1000003 + g * 2654435761 + 12345) & 0xFFFFFFFFFFFFFFFF
        deck = base.copy()
        for i in range(51, 0, -1):
            st = (st * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
            j = (st >> 33) % (i + 1)
            t = deck[i]; deck[i] = deck[j]; deck[j] = t
        c = 0
        for i in range(26):
            if deck[2 * i] == 14:
                c += 1
        aces_a[g] = c
        r, w, win = _play(deck, war_down, order, max_rounds, st ^ 0x9E3779B97F4A7C15)
        rounds[g] = r; wars[g] = w; winner[g] = win
    return rounds, wars, winner, aces_a
