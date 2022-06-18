from dataclasses import dataclass, field
from Card import Card, SPADES, SUIT_NAMES, Suit

@dataclass
class Observation:
    player_count: int = 4
    cards_seen: "list[Card]" = field(default_factory=list)
    trick_pile: "list[Card]" = field(default_factory=list)
    trump: Suit = SPADES
    leading_suit: Suit = SPADES

    def __repr__(self) -> str:
        cal = '\n | '
        return f"Observation on {self.player_count}-player game. \n{self.count_cards_in_pile()} plays into trick.\nTrump: {SUIT_NAMES[self.trump]}\nLeading Suit: {SUIT_NAMES[self.leading_suit]} \nCards in trick so far:\n | {cal.join(map(repr, self.trick_pile))} \nCards played prior over the whole game: {list(map(lambda x: x.minimal_repr(), self.cards_seen))}"

    def count_cards_in_pile(self) -> int:
        return len(self.trick_pile)
