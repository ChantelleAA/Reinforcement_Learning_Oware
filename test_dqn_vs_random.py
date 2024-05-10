# from env import DQNAgent
# from env import GameController
# import numpy as np

# def test_dqn(agent, game_controller, test_episodes):
#     total_rewards = []  

#     for episode in range(test_episodes):
#         state = game_controller.reset_game()
#         state = np.reshape(state, [1, agent.state_size])
#         done = False
#         total_reward = 0

#         while not done:
            
#             act_values = agent.model.predict(state)
#             action = np.argmax(act_values[0])

#             next_state, reward, done, info = game_controller.step(action)
#             next_state = np.reshape(next_state, [1, agent.state_size])

#             total_reward += reward
#             state = next_state

#         total_rewards.append(total_reward)
#         print(f"Test Episode {episode+1}/{test_episodes}: Reward: {total_reward}")

#     return total_rewards

# def load_weights(agent, weights_path):
#     agent.model.load_weights(weights_path)
#     print("Weights loaded successfully.")

# test_agent = DQNAgent(state_size=12, action_size=12)
# load_weights(test_agent, 'saved_weights_epoch_5.h5')

# game_controller = GameController()  
# trained_agent = DQNAgent(state_size=12, action_size=12)  
# trained_agent.model.load_weights('saved_weights_epoch_5.h5') 

# test_rewards = test_dqn(trained_agent, game_controller, test_episodes=5)
# print("Average reward:", np.mean(test_rewards))

from env import DQNAgent, RandomAgent, GameController
import numpy as np
import matplotlib.pyplot as plt
import random


def test_dqn_vs_random(dqn_agent, random_agent, game_controller, test_episodes):
    total_rewards = []  
    wins = 0
    losses = 0
    draws = 0

    for episode in range(test_episodes):
        state = game_controller.reset_game()
        board = game_controller.board.board_indices
        state = np.reshape(state, [1, dqn_agent.state_size])
        done = False
        total_reward = 0
        step = 0
        rewards = []
        actions = []

        while not done:
            if step % 2 == 0:  # DQN agent's turn
                act_values = dqn_agent.model.predict(state)
                action = np.argmax(act_values[0])
            else:  # Random agent's turn
                action, _ = random_agent.act(state, board_indices=board)

            next_state, reward, done, info = game_controller.step(action, 1)
            next_state = np.reshape(next_state, [1, dqn_agent.state_size])
            
            if step % 2 == 0:
                total_reward += reward  # Only accumulate rewards for the DQN agent

            if np.sum(next_state, axis=None)==0:
                done=True

            state = next_state
            rewards.append(reward)
            actions.append(action)
            step += 1

        total_rewards.append(total_reward)
        print(f"Test Episode {episode+1}/{test_episodes}: Reward: {total_reward}")

        # Determine the winner
        winner = game_controller.environment.game_winner()
        if winner == 1:  # Assuming 1 is the ID for the DQN agent
            wins += 1
        elif winner == 2:
            losses +=1
        elif winner == 0:
            draws +=1        

        # Visualization per episode
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(rewards)
        plt.title('Rewards per Step')
        plt.xlabel('Step')
        plt.ylabel('Reward')
        
        plt.subplot(1, 2, 2)
        plt.hist(actions, bins=np.arange(dqn_agent.action_size+1) - 0.5, rwidth=0.8)
        plt.title('Action Distribution')
        plt.xlabel('Action')
        plt.ylabel('Frequency')

        plt.suptitle(f'Episode {episode+1} Analysis')
        plt.show()

    win_rate = wins / test_episodes  # Calculate the win rate of the DQN agent
    print(f"Total Episodes: {test_episodes}, Wins: {wins}")
    print(f"Win Rate of DQN Agent: {win_rate:.2%}")

    return total_rewards

# Initialize agents and game controller
game_controller = GameController()  
trained_agent = DQNAgent(state_size=12, action_size=12, player_id=0)  
trained_agent.model.load_weights('saved_weights_epoch_4.h5') 
random_agent = RandomAgent(action_size=12)

# Run test
test_rewards = test_dqn_vs_random(trained_agent, random_agent, game_controller, test_episodes=100)
print("Average reward:", np.mean(test_rewards))
