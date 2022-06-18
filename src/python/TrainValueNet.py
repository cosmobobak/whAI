import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib.pyplot as plt
from Game import Game
from Agent import Agent
from Policies import AITrickPolicy, AITrumpPolicy, RandomPolicy, trick_pol_gen, complex_threshold, greedy_threshold, random_threshold, scared_threshold, thresholds_tracker
import tensorflow as tf
import numpy as np
from hyperparams import *
from ValueNetwork import CardValueNet, vectorise_obs_action_pair
from tqdm import tqdm

def generate_data(runs):
    game = Game()
    game.set_player(0, Agent(id="greedy", trick_policy=trick_pol_gen(
        greedy_threshold), trump_policy=AITrumpPolicy))
    game.set_player(1, Agent(id="scared", trick_policy=trick_pol_gen(
        scared_threshold), trump_policy=AITrumpPolicy))
    game.set_player(2, Agent(id="complex", trick_policy=trick_pol_gen(
        complex_threshold), trump_policy=AITrumpPolicy))
    # game.set_player(3, Agent(id="random", trick_policy=trick_pol_gen(random_threshold), trump_policy=AITrumpPolicy))
    game.set_player(
        3, Agent(id="random", trick_policy=RandomPolicy, trump_policy=AITrumpPolicy))
    game.readout = False

    for _ in tqdm(range(runs)):
        game.run_game()

    print(f"Done generating data, vectorising data.")
    return game.dump_data()

def bool_to_vec(b):
    out = np.zeros(2)
    idx = 1 if b else 0
    out[idx] = 1
    return out

def main():
    print(f"Generating data on {RUNS} games!")
    data_dump = generate_data(RUNS)

    # convert to multiples of BATCH_SIZE
    datalen = RUNS * 52
    resize_endpoint = datalen // (BATCH_SIZE *
                                  BATCH_SCALING) * (BATCH_SIZE * BATCH_SCALING)
    xy_train = [(vectorise_obs_action_pair(obs, act), bool_to_vec(v))
                for obs, act, v in data_dump][:resize_endpoint]
    print(f"Done vectorising data, processing data.")
    x_train = np.stack([x for x, _ in tqdm(xy_train)], axis=0)
    y_train = np.stack([y for _, y in tqdm(xy_train)], axis=0)
    print(
        f"trimmed data to chunks of {BATCH_SCALING}x batch_size and converted to ndarrays.")
    print(f"{len(xy_train)=}")
    print(f"{len(x_train)=}")
    print(f"{len(y_train)=}")
    print(f"{np.ndim(x_train)=}")
    print(f"{np.ndim(y_train)=}")
    assert len(x_train) == len(y_train)
    # y_val = pd.read_csv(VALIDATION_DATA_FILENAME, usecols=[49])

    # get model
    model = CardValueNet()()

    model_id = input("Enter current training run ID: \n--> valuenet_")

    # checkpoint stuff
    checkpoint_dir = os.path.dirname(CHECKPOINT_PATH + "_" + model_id)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=CHECKPOINT_PATH, save_weights_only=True, verbose=1, save_best_only=True)

    es_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=8)

    logdir = os.path.join(os.path.curdir, "logs")
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    model.fit(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[cp_callback, es_callback, tb_callback],
        validation_split=VALIDATION_SPLIT
    )

    save_path = FINAL_SAVE_PATH + "_" + model_id
    model.save(save_path)
    print(f"Saved model as {save_path}")


if __name__ == "__main__":
    main()
