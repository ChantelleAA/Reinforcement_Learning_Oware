from env import DQNAgent, GameController
import numpy as np
import matplotlib.pyplot as plt

def human_action(state, valid_actions):
    """
    Allows a human player to select an action based on the current game state.
    Assumes that the state is printed or otherwise visible to the human player.
    """
    print("Current game state:")
    print(state)  # Modify this line if you need a more complex representation
    print("Valid actions:", valid_actions)
    state = np.reshape(state, (2, -1))
    action = None
    while action not in valid_actions:
        try:
            action = int(input("Enter your action choice: "))
            if action not in valid_actions:
                print(f"Invalid action. Please choose from {valid_actions}.")
        except ValueError:
            print("Please enter a valid integer representing the action.")
    return action

def test_dqn_vs_human(dqn_agent, game_controller, test_episodes):
    total_rewards = []

    for episode in range(test_episodes):
        state = game_controller.reset_game()
        state = np.reshape(state, [1, dqn_agent.state_size])
        done = False
        total_reward = 0
        step = 0

        while not done:
            if step % 2 == 0:  # DQN agent's turn
                act_values = dqn_agent.model.predict(state)
                action = np.argmax(act_values[0])
                print(f"DQN Agent chooses action: {action}")
            else:  # Human player's turn
                if np.sum(state, axis=None)==0:
                    done=True  
                    continue              
                action = human_action(state, game_controller.environment.valid_moves((np.reshape(state, (2, -1)))))

            next_state, reward, done, info = game_controller.step(action, game_controller.get_player_turn())
            next_state = np.reshape(next_state, [1, dqn_agent.state_size])

            if step % 2 == 0:
                total_reward += reward  # Accumulate rewards only for the DQN agent

            if np.sum(state, axis=None)==0:
                done=True
            state = next_state
            step += 1

        total_rewards.append(total_reward)
        print(f"Test Episode {episode+1}/{test_episodes}: Reward: {total_reward}")

    print(f"Average reward for DQN Agent: {np.mean(total_rewards):.2f}")
    return total_rewards

# Initialize the game controller and agents
game_controller = GameController()  
trained_agent = DQNAgent(state_size=12, action_size=12, player_id=1)  
trained_agent.model.load_weights('saved_weights_epoch_5.h5')

# Run the test
test_rewards = test_dqn_vs_human(trained_agent, game_controller, test_episodes=5)
