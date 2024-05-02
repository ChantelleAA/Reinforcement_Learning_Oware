from env import DQNAgent
from env import GameController
import numpy as np
import random

import warnings
warnings.filterwarnings('ignore')


def possible_states(state, board, game_state, agent):
    valid_actions = []
    for action in range(agent.action_size):
        pit_index = board.board_indices[action]
        if game_state.current_board_state[pit_index] > 0:
            valid_actions.append(action)
            if not valid_actions:
                raise ValueError("No valid actions available for this current state")
    return valid_actions

def act(state, board, game_state, agent):
    valid_actions = possible_states(state, board, game_state, agent)
    if not valid_actions:
        raise ValueError("No valid actions available for the current state.")
    
    if np.random.rand() <= agent.epsilon:
        return random.choice(valid_actions)
    else:
        act_values = agent.model.predict(state)
        return np.argmax(act_values[0])
    


def train_dqn(episodes, game_controller):
    action_size = game_controller.action_space_size
    state_size = game_controller.state_space_size
    board = game_controller.board
    game_state = game_controller.environment

    agent = DQNAgent(state_size=12, action_size=12) 
    count = 0
    for e in range(episodes):
        state = game_controller.reset_game()
        print(state)
        state = np.reshape(state, [1, state_size])
        while count<1000:
            action = act(state, board, game_state, agent)
            next_state, reward, done, info = game_controller.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            count = count+1
            if done:
                print(f"episode: {e+1}/{episodes}, score: {game_controller.get_score()}, e: {agent.epsilon:.2}")
                break
            if len(agent.memory) > 32:
                agent.replay(32)

# Example usage
game_controller = GameController()  # Your game controller with a compatible interface
train_dqn(10, game_controller)
