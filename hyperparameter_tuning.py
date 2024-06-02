# from env import DQNAgent, RandomAgent, GameController, DDQNAgent
from agent.DQNAgent import DQNAgent
from agent.DDQNAgent import DDQNAgent
from agent.RandomAgent import RandomAgent
from env.GameController import GameController

import numpy as np
import random
import matplotlib.pyplot as plt
import warnings
import datetime
warnings.filterwarnings('ignore')
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled")
    except RuntimeError as e:
        print(e)

INVALID_ACTION_PENALTY = -1

from sklearn.model_selection import ParameterGrid
import numpy as np

def train_agents_with_params(params, episodes=100):
    learning_rate = params['learning_rate']
    gamma = params['gamma']
    clipping_ratio = params['clipping_ratio']
    epsilon = params['epsilon']
    epsilon_decay = params['epsilon_decay']
    batch_size = params['batch_size']

    game_controller = GameController()

    action_size = game_controller.action_space_size
    state_size = game_controller.state_space_size

    rid = random.sample([1, 2], 1)[0]
    aid = 1 if rid == 2 else 2

    learning_agent = DDQNAgent(state_size, action_size, aid, 
                               learning_rate=learning_rate, 
                               gamma=gamma, 
                               epsilon=epsilon, 
                               epsilon_decay=epsilon_decay)
    learning_agent.optimizer.clipnorm = clipping_ratio

    random_agent = RandomAgent(action_size, game_controller.board, rid)
    rewards_learner = []

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        for e in range(episodes):
            state = game_controller.reset_game()
            state = np.reshape(state, [1, state_size])
            done = False
            episode_reward = 0

            while not done:
                if params['agent_type'] == 'ddqn':
                    action1, penalty1 = learning_agent.ddqn_act(state, learning_agent, game_controller.board)
                else:
                    action1, penalty1 = learning_agent.dqn_act(state, learning_agent, game_controller.board)

                next_state, reward1, done, info = game_controller.step(action1, player=1)
                next_state = np.reshape(next_state, [1, state_size])
                learning_agent.remember(state, action1, reward1 + penalty1, next_state, done)
                state = next_state
                episode_reward += reward1

                if done:
                    break

                action2, penalty2 = random_agent.act(state, game_controller.board)
                next_state, reward2, done, info = game_controller.step(action2, player=2)
                next_state = np.reshape(next_state, [1, state_size])
                state = next_state
                episode_reward += reward2

                if len(learning_agent.memory) > batch_size:
                    learning_agent.replay(batch_size)

            rewards_learner.append(episode_reward)

    cumulative_reward = np.sum(rewards_learner)
    return cumulative_reward

param_grid = {
    'learning_rate': [0.001, 0.005, 0.01],
    'gamma': [0.8, 0.9, 0.99],
    'clipping_ratio': [0.001, 0.01, 0.1],
    'epsilon': [1.0, 0.5],
    'epsilon_decay': [0.995, 0.999],
    'batch_size': [32, 64, 128],
    'agent_type': ['ddqn']
}

best_params = None
best_reward = -np.inf
for params in ParameterGrid(param_grid):
    reward = train_agents_with_params(params)
    if reward > best_reward:
        best_reward = reward
        best_params = params

print("Best Hyperparameters:")
print(best_params)
print("Best Reward:")
print(best_reward)
