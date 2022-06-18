from Card import Card
from Observation import Observation
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.layers import Conv2D, Flatten, Dense, BatchNormalization, Input, concatenate, Reshape, MaxPooling2D, Add
from hyperparams import BATCH_SIZE, DIMENSIONS
from tensorflow.keras.models import Model
from tensorflow import keras
import numpy as np

def vectorise_obs_action_pair(obs: Observation, action: Card) -> np.ndarray:
    SUIT_VECTOR = 0
    HISTORY_VECTOR = 1
    PILE_VECTOR = 2
    ACTION_VECTOR = 5
    out = np.zeros(DIMENSIONS)
    # setting row 1
    trump_offset = [0, 13, 26, 39][obs.trump - 1]
    for i in range(0 + trump_offset, 13 + trump_offset):
        out[SUIT_VECTOR][i] = 1
    # setting row 2
    for card in obs.cards_seen:
        out[HISTORY_VECTOR][card.get_index()] = 1
    pile_size = obs.count_cards_in_pile()
    for i in range(pile_size):
        card = obs.trick_pile[i]
        idx = card.get_index()
        out[PILE_VECTOR + i][idx] = 1
    # setting row 6
    out[ACTION_VECTOR] = action.vectorise()
    return out

class CardValueNet:
    # input is a 52x6 matrix, first row is trump, second row is card history, next three rows are the pile, last is the considered card.
    def __init__(self, xbatch_size=BATCH_SIZE) -> None:
        print(f"{xbatch_size=}")
        input_layer = Input(shape=DIMENSIONS, batch_size=xbatch_size, name="input")
        x = Flatten()(input_layer)
        x = Dense(512, activation="relu", name="Dense1")(x)
        x = Dense(256, activation="relu", name="Dense2")(x)
        x = Dense(128, activation="relu", name="Dense3")(x)
        # x = Dense(64, activation="relu", name="Dense4")(x)
        # x = Dense(32, activation="relu", name="Dense5")(x)

        output_layer = Dense(2, name="eval", activation="softmax")(x)
        self.evalModel = Model(inputs=input_layer, outputs=output_layer)

        self.evalModel.compile(
            optimizer=keras.optimizers.SGD(),
            loss=keras.losses.CategoricalCrossentropy(),
            metrics=[keras.metrics.CategoricalAccuracy()],
        )

        self.evalModel.summary()

    def __call__(self) -> Model:
        return self.evalModel
