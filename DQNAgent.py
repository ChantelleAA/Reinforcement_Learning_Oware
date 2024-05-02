import random
from collections import deque
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import sys
sys.path.append("C:/Users/chant/OneDrive/Desktop/thesis_code")

from env.Board import Board
from env.GameState import GameState



TF_ENABLE_ONEDNN_OPTS=0
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)  # Memory buffer for storing experiences
        self.gamma = 0.95  # Discount factor for future rewards
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

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

    # def possible_states(self, state):
    #     valid_actions = []
    #     for action in range(self.action_size):
    #         pit_index = self.board.board_indices[action]
    #         if self.environment.current_board_state[pit_index] > 0:
    #             valid_actions.append(action)
    #             if not valid_actions:
    #                 raise ValueError("No valid actions available for this current state")
    #     return valid_actions


    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.choice(self.possible_states(state))
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    # def act(self, state):
    #     valid_actions = self.possible_states(state)
    #     if not valid_actions:
    #         raise ValueError("No valid actions available for the current state.")
        
    #     if np.random.rand() <= self.epsilon:
`       `    #     else:
    #         act_values = self.model.predict(state)
    #         return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.max(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

