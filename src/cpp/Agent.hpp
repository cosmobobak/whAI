#pragma once

#include <string>
#include <vector>
#include "Observation.hpp"
#include "Card.hpp"

using Observation::Observation;
using Card::Card;

template <typename TrickPolicy, typename TrumpPolicy>
class Agent {
    std::string id;
    std::vector<Card> hand;
    TrickPolicy trick_policy;
    TrumpPolicy trump_policy;

    Agent(std::string id = "NULL", TrickPolicy trick_policy, TrumpPolicy trump_policy) {
        this->id = id;
        this->trick_policy = trick_policy;
        this->trump_policy = trump_policy;
    }

    auto play(Observation observation) -> Card {
        auto idx = self.trick_policy(self.hand, observation);
        auto choice = self.hand.pop(idx);
        return choice;
    }

    auto add_to_hand(Card card) {
        self.hand.append(card);
    }

    auto choose_trump() {
        return self.trump_policy(self.hand);
    }
}