import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tqdm import tqdm
from Policies import AITrickPolicy, AITrumpPolicy, NeuralPolicy, RandomPolicy, trick_pol_gen, complex_threshold, greedy_threshold, random_threshold, scared_threshold, thresholds_tracker
from Agent import Agent
from Game import Game
import matplotlib.pyplot as plt

RUNS = 100

def main():
    game = Game()
    game.set_player(0, Agent(id="greedy", trick_policy=trick_pol_gen(greedy_threshold), trump_policy=AITrumpPolicy))
    game.set_player(1, Agent(id="scared", trick_policy=trick_pol_gen(scared_threshold), trump_policy=AITrumpPolicy))
    game.set_player(2, Agent(id="complex", trick_policy=trick_pol_gen(complex_threshold), trump_policy=AITrumpPolicy))
    # game.set_player(3, Agent(id="random", trick_policy=trick_pol_gen(random_threshold), trump_policy=AITrumpPolicy))
    game.set_player(3, Agent(id="neural", trick_policy=NeuralPolicy(), trump_policy=AITrumpPolicy))
    game.readout = False

    results = [game.run_game() for _ in tqdm(range(RUNS))]

    print(f"{game.players[0].id}: {results.count(0) * 100 / RUNS:.1f}%")
    print(f"{game.players[1].id}: {results.count(1) * 100 / RUNS:.1f}%")
    print(f"{game.players[2].id}: {results.count(2) * 100 / RUNS:.1f}%")
    print(f"{game.players[3].id}: {results.count(3) * 100 / RUNS:.1f}%")

    print(
        f"average threshold: {sum(thresholds_tracker) / len(thresholds_tracker)}")
    print(
        f"median threshold: {sorted(thresholds_tracker)[len(thresholds_tracker) // 2]}")
    print(
        f"maxmin: {max(thresholds_tracker)} {min(thresholds_tracker)}")

    fig, ax = plt.subplots()  # Create a figure containing a single axes.
    ax.plot(list(range(1, len(thresholds_tracker) + 1)), sorted(thresholds_tracker))  # Plot some data on the axes.
    plt.savefig('/mnt/c/github/whAI/plot.png', bbox_inches='tight')

    for state, action, value in game.dump_data():
        print(f"Observation: {repr(state.trick_pile): <55}, Action: {action}, Winning? {value}")

main()
