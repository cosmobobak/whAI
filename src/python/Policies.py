import numpy as np
import tensorflow as tf
from hyperparams import FINAL_SAVE_PATH
from ValueNetwork import CardValueNet, vectorise_obs_action_pair
from random import choice, randint
from typing import Callable
from Observation import Observation
from Card import ACE, AVERAGE_HAND_QUALITY, CARDS, Card, DIAMONDS, HEARTS, JACK, SPADES, SUITS, SUIT_NAMES, Suit, TWO, value

thresholds_tracker = []

def is_legal(card: Card, hand: "list[Card]", obs: Observation) -> bool:
    if card.suit == obs.leading_suit:
        return True
    return all(c.suit != obs.leading_suit for c in hand)


def worst(card_set: "list[Card]", trump: Suit):
    def contextual_value(c): return value(c, trump)
    return min(card_set, key=contextual_value)


def is_winning(card: Card, obs: Observation) -> bool:
    if obs.count_cards_in_pile() == 0:
        return True
    def contextual_value(c): return value(c, obs.trump)
    best_so_far = max(obs.trick_pile, key=contextual_value)
    return contextual_value(card) > contextual_value(best_so_far)

def average_card_value(card_set: "list[Card]", trump: Suit):
    if len(card_set) == 0:
        return AVERAGE_HAND_QUALITY
    return sum(value(c, trump) for c in card_set) / len(card_set)

def show_cards(cards, msg, consultation_list = []):
    if consultation_list == []:
        consultation_list = cards
    calator = '\n | '
    card_string = calator.join(f"{consultation_list.index(c) + 1}: {c}"
                               for c in cards)
    print(f"{msg} \n | {card_string}")

def HumanTrickPolicy(hand: "list[Card]", obs: Observation) -> int:
    # NO MUTATING DATA OR ELSE

    show_cards(hand, "Your cards are:")

    legal_choices = [c for c in hand if is_legal(c, hand, obs)]

    show_cards(legal_choices, "Your legal choices are:", hand)

    idx = int(input("Choose a card \n==> "))
    while hand[idx - 1] not in legal_choices:
        idx = int(input("==> "))
    return idx - 1

def HumanTrumpPolicy(hand: "list[Card]") -> int:
    # NO MUTATING DATA OR ELSE

    show_cards(hand, "Your cards are:")

    suit_string = ' | '.join(f"{p + 1}: {SUIT_NAMES[suit]}" for p, suit in enumerate(SUITS))
    suit = Suit(input(f"Choose a suit to be trump [{suit_string}] \n==> "))
    while suit not in SUITS:
        suit = Suit(input("==> "))
    return suit


def complex_threshold(greed: float, hand_quality: float, pile_quality: float, trick_progress: float) -> bool:
    global thresholds_tracker
    THRESHOLD = -100
    if pile_quality == 0:
        return greed == 1
    card_ratio = hand_quality - pile_quality
    chances = card_ratio * trick_progress * greed * 10
    # print(chances)
    thresholds_tracker.append(chances)
    return chances > THRESHOLD


def random_threshold(greed: float, hand_quality: float, pile_quality: float, trick_progress: float) -> bool:
    return choice([True, False])


def greedy_threshold(greed: float, hand_quality: float, pile_quality: float, trick_progress: float) -> bool:
    return True


def scared_threshold(greed: float, hand_quality: float, pile_quality: float, trick_progress: float) -> bool:
    return False


def AITrickPolicy(hand: "list[Card]", obs: Observation, threshold_function: Callable) -> int:
    # value from 0 (never try to win before the last turn) to 1 (blow your best card on round 1)
    greed = 1.5

    trump = obs.trump
    legal_choices = [c for c in hand if is_legal(c, hand, obs)]
    hand_quality = average_card_value(hand, trump)
    pile_quality = average_card_value(obs.trick_pile, trump)

    trick_progress = obs.count_cards_in_pile() / obs.player_count
    is_last_play: bool = (obs.player_count == obs.count_cards_in_pile() + 1)
    is_first_play = obs.count_cards_in_pile() == 0
    try_to_win_hand: bool = not is_first_play and (is_last_play or threshold_function(
        greed, hand_quality, pile_quality, trick_progress))

    if try_to_win_hand:
        winning_cards = [c for c in legal_choices if is_winning(c, obs)]
        if len(winning_cards) > 0:
            return hand.index(worst(winning_cards, trump))

    return hand.index(worst(legal_choices, trump))

def AITrumpPolicy(hand: "list[Card]") -> int:
    suits = [0, 0, 0, 0]
    for card in hand:
        suits[card.suit - 1] += 1
    return suits.index(max(suits)) + 1

def trick_pol_gen(threshold_function: Callable) -> Callable:
    return lambda hand, obs: AITrickPolicy(hand, obs, threshold_function)


def RandomPolicy(hand: "list[Card]", obs: Observation) -> int:
    legal_choices = [c for c in hand if is_legal(c, hand, obs)]
    return hand.index(choice(legal_choices))

class NeuralPolicy():
    def __init__(self) -> None:
        self.model = tf.keras.models.load_model(
            FINAL_SAVE_PATH + "_" + "10000")

        # weights = self.loaded_model.get_weights()
        # self.model = CardValueNet(
        #     xbatch_size=1
        # )()

        # self.model.set_weights(weights)
        # # single_item_model.load_weights(CHECKPOINT_PATH)
        # self.model.compile(
        #     optimizer=tf.keras.optimizers.SGD(),
        #     loss=tf.keras.losses.CategoricalCrossentropy(),
        #     metrics=[tf.keras.metrics.CategoricalAccuracy()],
        # )

    def __call__(self, hand: "list[Card]", obs: Observation) -> int:
        trump = obs.trump
        legal_choices = [c for c in hand if is_legal(c, hand, obs)]
        options = [vectorise_obs_action_pair(obs, card) for card in legal_choices]
        evals = self.model.predict(np.stack(options, axis=0))
        winning_cards = [o for i, o in enumerate(legal_choices) if np.argmax(evals[i]) == 1]
        if len(winning_cards) > 0:
            return hand.index(worst(winning_cards, trump))
        return hand.index(worst(hand, trump))

