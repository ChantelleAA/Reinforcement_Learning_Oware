from env import DQNAgent, RandomAgent, GameController
import numpy as np
import matplotlib.pyplot as plt

def test_dqn_vs_random(dqn_agent, random_agent, game_controller, test_episodes):
    total_rewards = []
    wins = 0
    losses = 0
    draws = 0

    for episode in range(test_episodes):
        state = game_controller.reset_game()
        board = game_controller.board
        state = np.reshape(state, [1, dqn_agent.state_size])
        done = False
        total_reward = 0
        step = 0
        rewards = []
        actions = []

        while not done:
            if step % 2 == 0:  # DQN agent's turn
                pid = 1
                act_values = dqn_agent.model.predict(state)
                action = np.argmax(act_values[0])
            else:  # Random agent's turn
                pid = 2
                action, _ = random_agent.act(state, board)

            next_state, reward, done, info = game_controller.step(action, pid)
            next_state = np.reshape(next_state, [1, dqn_agent.state_size])

            if step % 2 == 0:
                total_reward += reward  # Only accumulate rewards for the DQN agent

            if np.sum(next_state) == 0:
                done = True

            if action is not None:
                state = next_state
                rewards.append(reward)
                actions.append(action)
                step += 1

        total_rewards.append(total_reward)
        print(f"Test Episode {episode + 1}/{test_episodes}: Reward: {total_reward}")

        # Determine the winner
        winner = info["winner"]
        print("winner: ", winner)
        if winner == 1:  # Assuming 1 is the ID for the DQN agent
            wins += 1
        elif winner == 2:
            losses += 1
        elif winner == 0:
            draws += 1
        else:
            print(winner)
    return total_rewards, wins, losses, draws

# Initialize agents and game controller
game_controller = GameController()
trained_agent = DQNAgent(state_size=12, action_size=12, player_id=1)
trained_agent.model.load_weights('./saved_weights1/saved_weights_epoch_15.h5')
random_agent = RandomAgent(action_size=12, board=game_controller.board, player_id=2)

# Run test
test_episodes=25
test_rewards, wins, losses, draws = test_dqn_vs_random(trained_agent, random_agent, game_controller, test_episodes=test_episodes)
print("Average reward:", np.mean(test_rewards))
print(f"Total Episodes: {test_episodes}, Wins: {wins}, Draws: {draws}, Losses: {losses}")
win_rate = wins/test_episodes
print(f"Win Rate of DQN Agent: {win_rate:.2%}")
