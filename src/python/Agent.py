from typing import Callable
from Observation import Observation
from Card import Card
from Policies import HumanTrickPolicy, HumanTrumpPolicy

class Agent:
    def __init__(self, id: str = "NULL", trick_policy: Callable = HumanTrickPolicy, trump_policy: Callable = HumanTrumpPolicy) -> None:
        self.id = id
        self.hand: list[Card] = []
        self.trick_policy = trick_policy
        self.trump_policy = trump_policy

    def play(self, observation: Observation) -> Card:
        idx = self.trick_policy(self.hand, observation)
        choice = self.hand.pop(idx)
        return choice

    def add_to_hand(self, card: Card):
        self.hand.append(card)

    def choose_trump(self):
        return self.trump_policy(self.hand)

