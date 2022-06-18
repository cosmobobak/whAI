from dataclasses import dataclass
from typing import List
import numpy as np

Suit = int

TRUMP_BONUS = 13
CARD_BASE_VALUE = 4

[SPADES, CLUBS, DIAMONDS, HEARTS] = SUITS = list(range(1, 5))
NULL_SUIT = 0
SUIT_NAMES = ["NULL", "SPADES", "CLUBS", "DIAMONDS", "HEARTS"]
[TWO, THREE, FOUR, FIVE, SIX, SEVEN, EIGHT, NINE, TEN, JACK, QUEEN, KING, ACE] = FACES = list(range(1, 14))
FACE_NAMES = ["NULL", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE", "TEN", "JACK", "QUEEN", "KING", "ACE"]

@dataclass
class Card:
    suit: Suit = SPADES
    face: int = ACE

    def __repr__(self) -> str:
        return f"{FACE_NAMES[self.face]} of {SUIT_NAMES[self.suit]}"

    def minimal_repr(self) -> str:
        face = ["NULL", "2", "3", "4", "5", "6", "7",
                "8", "9", "10", "J", "Q", "K", "A"][self.face]
        suit = ["NULL", "S", "C", "D", "H"][self.suit]
        return f"{face}-{suit}"

    def get_index(self) -> int:
        return CARDS.index(self)

    def vectorise(self) -> np.ndarray:
        out = np.zeros(52)
        out[self.get_index()] = 1
        return out


def value(card: Card, trump: Suit):
    mod = TRUMP_BONUS if card.suit == trump else 0
    return card.face + mod + CARD_BASE_VALUE

CARDS = tuple(Card(s, f) for s in SUITS for f in FACES)
assert len(CARDS) == 13 * 4

AVERAGE_HAND_QUALITY = sum(value(c, SPADES) for c in CARDS) / len(CARDS)
