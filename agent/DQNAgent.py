import random
from collections import deque
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from utils.utils import *
from utils.config import *

class DQNAgent:
    def __init__(self, state_size, action_size, player_id):
        self.state_size = state_size
        self.action_size = action_size
        self.player_id = player_id  # Identifier for the player (1 or 2)
        self.memory = deque(maxlen=2000)  # Memory buffer for storing experiences
        self.gamma = 0.8  # Discount factor for future rewards
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.995
        self.learning_rate = 0.01
        self.model = self._build_model()

    def _build_model(self):
        """ Neural Net for Deep-Q learning Model. """
        # initializer = tf.keras.initializers.HeNormal()
        initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.03, seed=None)

        model = tf.keras.Sequential([
            layers.Dense(128, input_dim=self.state_size, activation='relu', kernel_initializer=initializer, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.Dense(128, activation='relu', kernel_initializer=initializer, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.Dense(128, activation='relu', kernel_initializer=initializer, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.Dense(128, activation='relu', kernel_initializer=initializer, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.Dense(self.action_size, activation='linear', kernel_initializer=initializer)
        ])
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, clipnorm=0.0001)
        model.compile(loss='mse', optimizer=optimizer)
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def dqn_act(self, state, agent, board):
        valid_actions = possible_states(agent.player_id, state, board)
        if not valid_actions:
            return None, INVALID_ACTION_PENALTY  # Return None for action and a penalty for invalid state

        if np.random.rand() <= agent.epsilon:
            return random.choice(valid_actions), 0
        else:
            act_values = agent.model.predict(state)
            act_values[0][[action for action in range(agent.action_size) if action not in valid_actions]] = float('-inf')
            return np.argmax(act_values[0]), 0

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        epsilon = 1e-8

        losses = []
        for state, action, reward, next_state, done in minibatch:
            state = np.squeeze(state)
            state = (state - np.min(state)) / (np.max(state) - np.min(state) + epsilon)
            next_state = np.squeeze(next_state)
            next_state = (next_state - np.min(next_state)) / (np.max(next_state) - np.min(next_state) + epsilon)

            target = self.model.predict(np.array([state]))
            if done:
                target[0][action] = reward
            else:
                t = self.model.predict(np.array([next_state]))[0]
                target[0][action] = reward + self.gamma * np.amax(t)

            history = self.model.fit(np.array([state]), target, epochs=1, verbose=0)
            losses.append(history.history['loss'][0])

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return np.mean(losses)

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
