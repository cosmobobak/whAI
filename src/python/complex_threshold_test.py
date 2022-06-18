from Card import ACE, CARDS, Card, DIAMONDS, HEARTS, JACK, SPADES, SUIT_NAMES, TWO
from Observation import Observation
from random import choice
from Policies import average_card_value, complex_threshold, is_legal, is_winning

if __name__ == "__main__":
    obs1 = Observation(player_count=4, cards_seen=[Card(SPADES, ACE), Card(HEARTS, ACE)], trump=SPADES, leading_suit=SPADES)
    obs2 = Observation(player_count=4, cards_seen=[Card(SPADES, TWO), Card(HEARTS, TWO)], trump=DIAMONDS, leading_suit=SPADES)
    obs3 = Observation(player_count=4, cards_seen=[Card(SPADES, JACK), Card(DIAMONDS, JACK)], trump=HEARTS, leading_suit=SPADES)
    obss = [obs1, obs2, obs3]
    h = [choice(CARDS) for _ in range(7)]
    
    for i, o in enumerate(obss):
        l = list(filter(lambda c: is_legal(c, h, o), h))
        w = list(filter(lambda c: is_winning(c, o), l))


        print(f"OBSERVATION {i}: {o.cards_seen} TRUMP: {SUIT_NAMES[o.trump]}")
        print(f"HAND: {h}")
        print(f"legal cards for observation 1: {l}")
        print(f"winning cards for observation 1: {w}")


        greed = 1.5
        hand_value = average_card_value(h, o.trump)
        trick_value = average_card_value(o.cards_seen, o.trump)
        prog = len(o.cards_seen) / o.player_count

        print(f"{hand_value = }")
        print(f"{trick_value = }")
        print(f"{hand_value / trick_value * prog * greed = }")

        ds = ["TRY TO WIN", "SAVE GOOD CARDS"]
        print(f"DECISION: {ds[0 if complex_threshold(greed, hand_value, trick_value, prog) else 1]}\n")