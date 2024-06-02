# from env import DQNAgent, RandomAgent, GameController
from env.GameController import GameController
from agent.RandomAgent import RandomAgent
from agent.DQNAgent import DQNAgent

import numpy as np
import random
import matplotlib.pyplot as plt
import datetime
import tensorflow as tf

INVALID_ACTION_PENALTY = -1

log_dir = "logs/dqn_vs_random_train/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, update_freq='batch')
file_writer = tf.summary.create_file_writer(log_dir)

# def zero_row_exists(state, board):
#     for i in range(board.nrows):
#         if np.sum(state[i,:], axis = None)==0 :
#             return True
#     return False

# def valid_moves(state, board):
#     actions = board.actions
#     valid=[]
#     for i in actions:
#         pit_index = board.action2pit(i)
#         if state[pit_index] != 0:
#             if zero_row_exists(state, board) and state[pit_index]> 6-i%6:
#                 valid.append(i)
#             elif not zero_row_exists(state, board):
#                 valid.append(i)
#     return valid

# def possible_states(player, state, board):
#     actions = board.actions
#     player_id = player-1
#     open_moves = []
#     state = np.reshape(state, (2, -1))
#     for i in actions:
#         pit_index = board.action2pit(i)
#         if pit_index in board.player_territories[player_id]:
#             if i in valid_moves(state, board) :
#                 open_moves.append(i)
#     return open_moves

# def dqn_act(state, agent, board):
#     valid_actions = possible_states(agent.player_id, state, board)
#     if not valid_actions:
#         return None, INVALID_ACTION_PENALTY  # Return None for action and a penalty for invalid state

#     if np.random.rand() <= agent.epsilon:
#         return random.choice(valid_actions), 0
#     else:
#         act_values = agent.model.predict(state)
#         act_values[0][[action for action in range(agent.action_size) if action not in valid_actions]] = float('-inf')
#         return np.argmax(act_values[0]), 0

def train_agents(episodes, game_controller, save_weights_path, save_model_path):
    rid = random.sample([1, 2], 1)[0]
    aid = 1 if rid==2 else 2
    action_size = game_controller.action_space_size
    state_size = game_controller.state_space_size
    learning_agent = DQNAgent(state_size, action_size, aid)
    random_agent = RandomAgent(action_size, game_controller.board, rid)  # Initialize the random agent
    rewards_learner = []
    rewards_random = []
    epsilon_values = []
    average_rewards = []
    episode_lengths = []

    # Training loop
    batch_size = 74

    for e in range(episodes):
        state = game_controller.reset_game()
        state = np.reshape(state, [1, state_size])
        done = False
        episode_reward = 0
        reward_random = 0
        episode_length = 0

        # Adjust learning rate manually based on epoch
        if e > 10:
            learning_agent.adjust_learning_rate(0.001)
        if e > 20:
            learning_agent.adjust_learning_rate(0.0005)
        if e > 30:
            learning_agent.adjust_learning_rate(0.0001)

        while not done:
            episode_length += 1
            # Learning agent's turn (DQN Agent)
            action1, penalty1 = learning_agent.dqn_act(state, learning_agent, game_controller.board)

            next_state, reward1, done, info = game_controller.step(action1, player=1)
            next_state = np.reshape(next_state, [1, state_size])
            learning_agent.remember(state, action1, reward1 + penalty1, next_state, done)
            state = next_state
            episode_reward += reward1
            rewards_learner.append(reward1)

            if done:
                print(f"Episode: {e+1}/{episodes}, Epsilon: {learning_agent.epsilon:.2f}")
                break

            action2, penalty2 = random_agent.act(state, game_controller.board)
            next_state, reward2, done, info = game_controller.step(action2, player=2)
            next_state = np.reshape(next_state, [1, state_size])
            state = next_state  # No memory or learning for the random agent
            reward_random += reward2
            rewards_random.append(reward2)

            # Only train the learning agent
            if len(learning_agent.memory) > batch_size:
                loss = learning_agent.replay(batch_size, tensorboard_callback)
                with file_writer.as_default():
                    tf.summary.scalar('Loss', loss, step=e)

            if done:
                print(f"Episode: {e+1}/{episodes}, Epsilon: {learning_agent.epsilon:.2f}")
                break
        
        epsilon_values.append(learning_agent.epsilon)
        average_rewards.append(episode_reward / episode_length)
        episode_lengths.append(episode_length)

        # Log the episode reward
        with file_writer.as_default():
            tf.summary.scalar('Episode Reward DQN Agent', episode_reward, step=e)
            tf.summary.scalar('Episode Reward Random Agent', reward_random, step=e)
            tf.summary.scalar("Episode Length", episode_length, step=e)
            tf.summary.scalar("Epsilon", learning_agent.epsilon, step=e)
            tf.summary.flush()

    np.savetxt("rewards_learner.csv", rewards_learner)
    np.savetxt("rewards_random.csv", rewards_random)    

    print("Training completed and model weights saved for the learning agent.")

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(np.cumsum(rewards_learner), "r", label="DQN Agent Rewards")
    plt.xlabel("Time Steps")
    plt.ylabel("Cumulative Reward")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(epsilon_values, "g", label="Epsilon Decay")
    plt.xlabel("Episodes")
    plt.ylabel("Epsilon")
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(np.cumsum(average_rewards), "b", label="Average Reward per Episode")
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative Average Reward")
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(episode_lengths, "m", label="Episode Length")
    plt.xlabel("Episodes")
    plt.ylabel("Length")
    plt.legend()

    plt.tight_layout()
    plt.show()

game_controller = GameController()
train_agents(30, game_controller, './saved_weights1/E{}.weights.h5', './saved_weights1/model_params{}.json')
