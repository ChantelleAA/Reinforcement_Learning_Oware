import random
from collections import deque
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard
import time
import datetime

import sys
sys.path.append("C:/Users/chant/OneDrive/Desktop/thesis_code")

from env.Board import Board
from env.GameState import GameState

# Setup TensorFlow environment settings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings('ignore')

# Set the log directory for TensorBoard logs
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)


class DQNAgent:
    def __init__(self, state_size, action_size, player_id):
        self.state_size = state_size
        self.action_size = action_size
        self.player_id = player_id  # Identifier for the player (1 or 2)
        self.memory = deque(maxlen=2000)  # Memory buffer for storing experiences
        self.gamma = 0.80  # Discount factor for future rewards
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.0995
        self.learning_rate = 0.004
        self.model = self._build_model()
        self.tensorboard = TensorBoard(log_dir=f"logs/{player_id}-{int(time.time())}")


    def _build_model(self):
        """ Neural Net for Deep-Q learning Model. """
        model = tf.keras.Sequential([
            layers.Dense(24, input_dim=self.state_size, activation='relu'),
            layers.Dense(24, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randint(0, self.action_size - 1)  # Assuming action space is discrete and zero-indexed
        act_values = self.model.predict(np.array([state]))
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = np.squeeze(state)
            next_state = np.squeeze(next_state)
            target = reward
            if not done:

                target += self.gamma * np.max(self.model.predict(np.array([next_state]))[0])
            target_f = self.model.predict(np.array([state]))
            target_f[0][action] = target
            state = np.squeeze(state)
            self.model.fit(np.array([state]), target_f, epochs=1, verbose=0, callbacks=[self.tensorboard])
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, name):
        self.model.save_weights(name)

    def load(self, name):
        self.model.load_weights(name)

    def save_model(self, name):
        model_json = self.model.to_json()
        with open(name, "w") as json_file:
            json_file.write(model_json)

    def load_model(self, name):
        json_file = open(name, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        return loaded_model_json

            