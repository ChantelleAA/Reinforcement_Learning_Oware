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
import csv

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

log_dir = "logs/ddqn_dqn_random_train/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer = tf.compat.v1.summary.FileWriter(log_dir)

def train_agents(episodes, game_controller, save_weights_path, save_model_path, agent_type='ddqn'):
    action_size = game_controller.action_space_size
    state_size = game_controller.state_space_size

    rid = random.sample([1, 2], 1)[0]
    aid = 1 if rid == 2 else 2

    if agent_type == 'ddqn':
        learning_agent = DDQNAgent(state_size, action_size, aid)
    else:
        learning_agent = DQNAgent(state_size, action_size, aid)

    random_agent = RandomAgent(action_size, game_controller.board, rid)
    rewards_learner = []
    rewards_random = []
    epsilon_values = []
    average_rewards = []
    episode_lengths = []
    loss_list = []

    # Additional data lists for CSV
    training_losses = []
    average_rewards_per_episode = []
    winners = []
    epsilon_over_episodes = []
    cumulative_rewards_end_game = []

    batch_size = 64

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        for e in range(episodes):
            state = game_controller.reset_game()
            state = np.reshape(state, [1, state_size])
            done = False
            episode_reward_ddqn = 0
            episode_reward_dqn = 0
            episode_reward_random = 0
            episode_reward = 0
            episode_length = 0

            while not done:
                episode_length += 1
                # Learning agent's turn (DDQN or DQN Agent)
                if agent_type == 'ddqn':
                    action1, penalty1 = learning_agent.ddqn_act(state, learning_agent, game_controller.board)
                else:
                    action1, penalty1 = learning_agent.dqn_act(state, learning_agent, game_controller.board)

                next_state, reward1, done, info = game_controller.step(action1, player=1)
                next_state = np.reshape(next_state, [1, state_size])
                learning_agent.remember(state, action1, reward1 + penalty1, next_state, done)
                state = next_state
                episode_reward += reward1
                if agent_type == 'ddqn':
                    episode_reward_ddqn += reward1
                else:
                    episode_reward_dqn += reward1

                rewards_learner.append(reward1)

                if done:
                    winners.append(1)  # Learning agent wins
                    print(f"Episode: {e+1}/{episodes}, Epsilon: {learning_agent.epsilon:.2f}")
                    break

                action2, penalty2 = random_agent.act(state, game_controller.board)
                next_state, reward2, done, info = game_controller.step(action2, player=2)
                next_state = np.reshape(next_state, [1, state_size])
                state = next_state  # No memory or learning for the random agent
                episode_reward += reward2
                episode_reward_random += reward2
                rewards_random.append(reward2)

                # Only train the learning agent
                if len(learning_agent.memory) > batch_size:
                    loss = learning_agent.replay(batch_size)
                    loss_list.append(loss)
                    training_losses.append(loss)  # Store loss for CSV
                    if loss is not None:
                        summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag='Loss', simple_value=loss)])
                        file_writer.add_summary(summary, e)

                if done:
                    winners.append(2)  # Random agent wins
                    print(f"Episode: {e+1}/{episodes}, Epsilon: {learning_agent.epsilon:.2f}")
                    break
            
            epsilon_values.append(learning_agent.epsilon)
            average_rewards.append(episode_reward / episode_length)
            average_rewards_per_episode.append(episode_reward / episode_length)  # Store average reward for CSV
            episode_lengths.append(episode_length)
            epsilon_over_episodes.append(learning_agent.epsilon)  # Store epsilon for CSV
            cumulative_rewards_end_game.append(episode_reward)  # Store cumulative reward for CSV

            # Log the episode reward
            summary = tf.compat.v1.Summary(value=[
                tf.compat.v1.Summary.Value(tag='Episode Reward', simple_value=episode_reward),
                tf.compat.v1.Summary.Value(tag='Episode Length', simple_value=episode_length),
                tf.compat.v1.Summary.Value(tag='Epsilon', simple_value=learning_agent.epsilon),
            ])
            if agent_type == 'ddqn':
                summary.value.add(tag='Episode Reward DDQN Agent', simple_value=episode_reward_ddqn)
            else:
                summary.value.add(tag='Episode Reward DQN Agent', simple_value=episode_reward_dqn)
            summary.value.add(tag='Episode Reward Random Agent', simple_value=episode_reward_random)
            file_writer.add_summary(summary, e)
            file_writer.flush()

        # Save model weights and structure
        learning_agent.save(save_weights_path.format(episodes))
        learning_agent.save_model(save_model_path.format(episodes))

    np.savetxt("rewards_learner.csv", rewards_learner)
    np.savetxt("rewards_random.csv", rewards_random)    

    # Save collected data to CSV
    if agent_type=='ddqn':
        file = 'training_data_ddqn.csv'
    elif agent_type=='dqn':
        file= 'training_data_dqn.csv'
    with open(file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Episode", "Loss", "Average Reward", "Winner", "Epsilon", "Cumulative Reward End Game"])
        for i in range(episodes):
            writer.writerow([i+1, training_losses[i] if i < len(training_losses) else 'N/A', 
                             average_rewards_per_episode[i], winners[i], epsilon_over_episodes[i], cumulative_rewards_end_game[i]])

    print("Training completed and model weights saved for the learning agent.")
    print(loss)
    print(loss_list)
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 3, 1)
    plt.plot(cumulative_rewards_end_game, "r", label="Cumulative Reward at End of Game")
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative Reward")
    plt.legend()

    plt.subplot(2, 3, 2)
    plt.plot(epsilon_values, "g", label="Epsilon Decay")
    plt.xlabel("Episodes")
    plt.ylabel("Epsilon")
    plt.legend()

    plt.subplot(2, 3, 3)
    plt.plot(np.cumsum(average_rewards), "b", label="Average Reward per Episode")
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative Average Reward")
    plt.legend()

    plt.subplot(2, 3, 4)
    plt.plot(episode_lengths, "m", label="Episode Length")
    plt.xlabel("Episodes")
    plt.ylabel("Length")
    plt.legend()

    plt.subplot(2, 3, 5)
    plt.plot(rewards_learner, "k--", label="Agent Rewards")
    plt.plot(rewards_random, "p--", label="Random Agent Rewards")
    plt.xlabel("Time Steps")
    plt.ylabel("Reward")
    plt.legend()   

    plt.subplot(2, 3, 6)
    plt.plot(loss_list, "m", label="Agent")
    plt.xlabel("Loss")
    plt.ylabel("Length")
    plt.legend() 

    plt.tight_layout()
    plt.show()

game_controller = GameController()
train_agents(100, game_controller, './saved_weights_ddqn/E{}.weights.h5', './saved_weights_dqn/model_params{}.json', agent_type='ddqn')
train_agents(100, game_controller, './saved_weights_dqn/E{}.weights.h5', './saved_weights_dqn/model_params{}.json', agent_type='dqn')