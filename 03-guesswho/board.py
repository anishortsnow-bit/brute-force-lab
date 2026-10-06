"""The board used in the video: 24 original faces, one name per letter from A to X.

It follows the design rule of the classic game: (almost) every feature belongs to exactly
five faces. Nine yes/no features have five faces each, and the five hair colours have
5, 4, 5, 5 and 5. Every pair of faces differs in at least two single-feature questions.
The table was found by random search (seed 2026) and then frozen here.
"""
import numpy as np

NAMES = ['Ada', 'Ben', 'Cal', 'Dot', 'Eli', 'Fay', 'Gus', 'Hal', 'Ivy', 'Jon', 'Kit', 'Lou',
         'Mae', 'Ned', 'Ola', 'Pip', 'Quin', 'Rex', 'Sue', 'Ted', 'Una', 'Vic', 'Wes', 'Xan']

HAIR = ['red', 'black', 'white', 'brown', 'white', 'black', 'blond', 'black', 'brown', 'red', 'blond', 'brown',
        'white', 'red', 'white', 'blond', 'red', 'blond', 'red', 'black', 'blond', 'white', 'brown', 'black']

FEATURES = {
    'hat':       [7, 12, 17, 18, 22],
    'glasses':   [1, 7, 13, 15, 20],
    'beard':     [1, 4, 13, 15, 16],
    'mustache':  [4, 10, 11, 19, 21],
    'bald':      [4, 6, 9, 13, 19],
    'earrings':  [6, 8, 12, 14, 18],
    'bowtie':    [0, 11, 12, 15, 23],
    'long_hair': [2, 8, 11, 14, 22],
    'rosy':      [2, 5, 10, 20, 21],
}
HAIR_COLOURS = ['black', 'brown', 'blond', 'red', 'white']

N = len(NAMES)
FULL = (1 << N) - 1


def has(i, feature):
    return i in FEATURES[feature]


def questions():
    """Every single-feature question as (label, bitmask of the faces that answer yes)."""
    out = []
    for f, idx in FEATURES.items():
        out.append((f, sum(1 << i for i in idx)))
    for colr in HAIR_COLOURS:
        out.append((colr + ' hair', sum(1 << i for i, h in enumerate(HAIR) if h == colr)))
    return out


QUESTIONS = questions()
MASKS = np.array([m for _, m in QUESTIONS], dtype=np.int64)

if __name__ == '__main__':
    for label, m in QUESTIONS:
        print(f'{label:12s} {bin(m).count("1")} faces: ' + ', '.join(NAMES[i] for i in range(N) if m >> i & 1))
    vec = [tuple(m >> i & 1 for _, m in QUESTIONS) for i in range(N)]
    assert len(set(vec)) == N
    print('all 24 faces can be told apart by single-feature questions')
