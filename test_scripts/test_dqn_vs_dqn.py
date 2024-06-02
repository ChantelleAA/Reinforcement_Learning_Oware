from env import DQNAgent, GameController
import numpy as np
import matplotlib.pyplot as plt
import random

def test_two_dqn_agents(agent1, agent2, game_controller,test_episodes):
    agent1_rewards = []
    agent2_rewards = []
    wins = [0, 0]  # Wins for agent1 and agent2 respectively
    draws = 0

    for episode in range(test_episodes):
        state = game_controller.reset_game()
        state = np.reshape(state, [1, agent1.state_size])
        done = False
        agent1_total_reward = 0
        agent2_total_reward = 0
        step = 0

        while not done:
            current_agent = agent1 if step % 2 == 0 else agent2
            opponent_agent = agent2 if step % 2 == 0 else agent1
            
            # Current agent's turn
            act_values = current_agent.model.predict(state)
            action = np.argmax(act_values[0])

            player_id = 1 if current_agent == agent1 else 2
            next_state, reward, done, info = game_controller.step(action, player_id)  # Assuming step() method correctly alternates players internally
            next_state = np.reshape(next_state, [1, current_agent.state_size])
            
            # Accumulate rewards for the appropriate agent
            if step % 2 == 0:
                agent1_total_reward += reward
            else:
                agent2_total_reward += reward

            # Both agents learn from the experience, regardless of whose turn it is
            if not done:
                current_agent.remember(state, action, reward, next_state, done)
                if len(current_agent.memory) > 32:
                    current_agent.replay(32)  # Assuming a batch size of 32

            state = next_state
            step += 1

        agent1_rewards.append(agent1_total_reward)
        agent2_rewards.append(agent2_total_reward)

        # Determine the winner, assuming game_controller can decide the winner
        winner = game_controller.environment.game_winner()
        if winner == 1:
            wins[0] += 1
        elif winner == 2:
            wins[1] += 1
        else:
            draws += 1

        print(f"Episode {episode+1}: Agent 1 Reward = {agent1_total_reward}, Agent 2 Reward = {agent2_total_reward}")

    print(f"Total Episodes: {test_episodes}, Agent 1 Wins: {wins[0]}, Agent 2 Wins: {wins[1]}, Draws: {draws}")
    print(f"Agent 1 Win Rate: {wins[0] / test_episodes:.2%}, Agent 2 Win Rate: {wins[1] / test_episodes:.2%}")
    
    return agent1_rewards, agent2_rewards

# Initialize agents and game controller
game_controller = GameController()  
agent1 = DQNAgent(state_size=12, action_size=12, player_id=1)
agent2 = DQNAgent(state_size=12, action_size=12, player_id=2)

agent1.model.load_weights('./saved_weights/saved_weights_epoch_1.h5') 
agent2.model.load_weights('./saved_weights/saved_weights_epoch_2.h5') 

# Run test
agent1_rewards, agent2_rewards = test_two_dqn_agents(agent1, agent2, game_controller, test_episodes=1)
print("Agent 1 Average Reward:", np.mean(agent1_rewards))
print("Agent 2 Average Reward:", np.mean(agent2_rewards))
