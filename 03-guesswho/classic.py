"""Feature list of the classic 24-character board, used only as a cross-check.

The table is the one encoded in the simulator behind Mark Rober's 2015 video
(github.com/chadgra/GuessWho, GameBoard.cs): 15 single-feature questions, 13 of them
true for exactly five characters and two for four.
"""
import numpy as np

CLASSIC = {
    'Claire': ['Woman', 'Ginger Hair', 'Hat', 'Glasses'], 'Eric': ['Yellow Hair', 'Hat'],
    'Maria': ['Woman', 'Brown Hair', 'Hat'], 'George': ['White Hair', 'Hat'],
    'Bernard': ['Brown Hair', 'Hat', 'Bulbous Nose'], 'Sam': ['White Hair', 'Bald', 'Glasses'],
    'Tom': ['Black Hair', 'Bald', 'Blue Eyes', 'Glasses'], 'Paul': ['White Hair', 'Glasses'],
    'Joe': ['Yellow Hair', 'Glasses'], 'Frans': ['Ginger Hair'], 'Anne': ['Woman', 'Black Hair'],
    'Max': ['Black Hair', 'Moustache', 'Thick Lips', 'Bulbous Nose'], 'Alex': ['Black Hair', 'Moustache', 'Thick Lips'],
    'Philip': ['Black Hair', 'Beard', 'Rosy Cheeks'], 'Bill': ['Ginger Hair', 'Bald', 'Beard', 'Rosy Cheeks'],
    'Anita': ['Woman', 'Yellow Hair', 'Rosy Cheeks', 'Blue Eyes'], 'David': ['Yellow Hair', 'Beard'],
    'Charles': ['Yellow Hair', 'Moustache', 'Thick Lips'], 'Herman': ['Ginger Hair', 'Bald', 'Bulbous Nose'],
    'Peter': ['White Hair', 'Thick Lips', 'Blue Eyes', 'Bulbous Nose'],
    'Susan': ['Woman', 'White Hair', 'Rosy Cheeks', 'Thick Lips'],
    'Robert': ['Brown Hair', 'Rosy Cheeks', 'Blue Eyes', 'Bulbous Nose'],
    'Richard': ['Brown Hair', 'Bald', 'Moustache', 'Beard'], 'Alfred': ['Ginger Hair', 'Moustache', 'Blue Eyes'],
}
NAMES = list(CLASSIC)
FEATURES = sorted({f for v in CLASSIC.values() for f in v})
MASKS = np.array([sum(1 << i for i, n in enumerate(NAMES) if f in CLASSIC[n]) for f in FEATURES], dtype=np.int64)

if __name__ == '__main__':
    for f, m in zip(FEATURES, MASKS):
        print(f'{f:13s} {bin(int(m)).count("1")}')
