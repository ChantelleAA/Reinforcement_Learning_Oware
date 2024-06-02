import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from env import RandomAgent, DQNAgent, DDQNAgent, GameController
import datetime
from tensorflow.keras.callbacks import TensorBoard
import random
from collections import deque

class AlphaZeroAgent:
    def __init__(self, state_size, action_size, player_id):
        self.state_size = state_size
        self.action_size = action_size
        self.player_id = player_id
        self.model = self.build_model()
        self.memory = deque(maxlen=2000)
        self.gamma = 0.8
        self.epsilon = 1.0
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.999
        self.learning_rate = 0.005

    def build_model(self):
        initializer = tf.keras.initializers.HeNormal()
        inputs = layers.Input(shape=(self.state_size,))
        shared = layers.Dense(128, activation='relu', kernel_initializer=initializer)(inputs)
        shared = layers.Dense(128, activation='relu', kernel_initializer=initializer)(shared)
        shared = layers.Dense(128, activation='relu', kernel_initializer=initializer)(shared)
        policy = layers.Dense(self.action_size, activation='softmax', kernel_initializer=initializer)(shared)
        value = layers.Dense(1, kernel_initializer=initializer)(shared)
        model = tf.keras.Model(inputs=inputs, outputs=[policy, value])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), 
                      loss=['categorical_crossentropy', 'mean_squared_error'])
        return model

    def remember(self, state, policy, value):
        self.memory.append((state, policy, value))

    def act(self, state):
        state = np.reshape(state, [1, self.state_size])
        policy, value = self.model.predict(state)
        action = np.random.choice(self.action_size, p=policy[0])
        return action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        states = np.zeros((batch_size, self.state_size))
        policies = np.zeros((batch_size, self.action_size))
        values = np.zeros((batch_size, 1))
        for i in range(batch_size):
            states[i], policies[i], values[i] = minibatch[i]

        self.model.fit(states, [policies, values], epochs=1, verbose=2)

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

def mcts_policy(state, agent, simulations=100):
    """ A simplified MCTS policy """
    valid_actions = [i for i in range(agent.action_size)]  # Assuming all actions are valid for simplicity
    action_probs = np.zeros(agent.action_size)
    for _ in range(simulations):
        action = np.random.choice(valid_actions)
        action_probs[action] += 1
    action_probs /= simulations
    return action_probs