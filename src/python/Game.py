from Observation import Observation
from Card import CARDS, Card, SUIT_NAMES, value, SUITS, FACES, NULL_SUIT
from dataclasses import dataclass, field
from Agent import Agent
from random import shuffle
from copy import deepcopy

PLAYER_COUNT = 4


class Game:
    def __init__(self) -> None:
        self.deck = list(CARDS)
        shuffle(self.deck)

        self.players = [Agent(f"Player {i + 1}") for i in range(PLAYER_COUNT)]
        cards_for_each_player = 52 // PLAYER_COUNT
        for p in self.players:
            for _ in range(cards_for_each_player):
                p.add_to_hand(self.deck.pop())
            p.hand = sorted(sorted(p.hand, key=lambda c: c.face), key=lambda c: c.suit)

        self.starting_player_idx = 0
        self.readout = True
        self.played_cards: "list[Card]" = []

        self.observations: list[Observation] = []
        self.actions: list[Card] = []
        self.values: list[bool] = []

    def record_trick_result(self, observations: "list[Observation]", pile: "list[Card]", winner: int):
        for i in range(PLAYER_COUNT):
            self.observations.append(observations[i])
            self.actions.append(pile[i])
            self.values.append(winner == i)

    def reset(self):
        self.deck = list(CARDS)
        shuffle(self.deck)

        for p in self.players:
            p.hand.clear()

        cards_for_each_player = 52 // PLAYER_COUNT
        for p in self.players:
            for _ in range(cards_for_each_player):
                p.add_to_hand(self.deck.pop())

        self.starting_player_idx = 0
        self.played_cards = []

    def set_player(self, i: int, p: Agent):
        '''
        Sets a new Agent object in one of the Game's player slots.

        Hand is maintained from the old player.
        '''
        transfer_hand = [c for c in self.players[i].hand]
        p.hand = transfer_hand
        self.players[i] = p

    def play_trick(self) -> int:
        current_player = self.starting_player_idx
        starting_player = self.players[current_player]
        trump = starting_player.choose_trump()
        if self.readout:
            print("Trick starting!")
            print(f"Trump is: {SUIT_NAMES[trump]}")
        trick_observations: "list[Observation]" = []
        pile: "list[Card]" = []
        observation = Observation(
            player_count=PLAYER_COUNT, 
            trick_pile=pile, 
            trump=trump, 
            leading_suit=NULL_SUIT,
            cards_seen=self.played_cards)
        while len(pile) < PLAYER_COUNT:
            player_to_move = self.players[current_player]
            # print(f"{observation}\n")
            trick_observations.append(deepcopy(observation))
            played_card = player_to_move.play(observation)
            if self.readout:
                print(f"{player_to_move.id} plays {played_card}!")

            # this should update observation bcos pass-by-ref bullshit
            pile.append(played_card)
            self.played_cards.append(played_card)

            if observation.leading_suit == NULL_SUIT:
                observation.leading_suit = pile[0].suit

            current_player += 1
            if current_player == PLAYER_COUNT:
                current_player = 0

        def contextual_value(c): return value(c, trump)
        best_card = pile.index(max(pile, key=contextual_value))
        winning_player = (best_card + self.starting_player_idx) % PLAYER_COUNT
        self.starting_player_idx = winning_player
        self.record_trick_result(trick_observations, pile, winning_player)
        return winning_player

    def run_game(self) -> int:
        scores = [0 for _ in self.players]
        max_tricks = 52 // PLAYER_COUNT
        trick = 1
        while sum(scores) < max_tricks:
            winner = self.play_trick()
            scores[winner] += 1
            if self.readout:
                print(f"{self.players[winner].id} wins trick {trick}!")
            trick += 1

        self.reset()
        return scores.index(max(scores))

    def dump_data(self):
        return zip(self.observations, self.actions, self.values)
