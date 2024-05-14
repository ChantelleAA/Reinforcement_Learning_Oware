from env import GameController
import numpy as np
import scipy.stats as sp
import random
from agent import DQNAgent
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
agent = DQNAgent.DQNAgent(12, 12)
num_of_simulations = 10
winners = []
epsilon = 0.1
for simulation in range(1, num_of_simulations+1):
    try:
        game = GameController()
        state = game.environment.current_board_state
        print(state, "\n")
        valid_actions = game.environment.valid_moves(state)
        print(valid_actions, "\n")
        if not valid_actions:
            raise ValueError("No valid actions available for the current state.")
        
        if np.random.rand() <= epsilon:
            action_c = random.choice(valid_actions)
        else:
            flattened_state = np.reshape(state, (-1, 12))
            print(flattened_state, "\n")
            act_values = agent.model.predict(flattened_state)
            action_c = np.argmax(act_values[0])
            print(f"Player chose action: {action_c}\n")

        game.step(action_c)
        obs = game.environment.all_states_np[:, 1:].astype(np.int32)
        print(game.environment.current_board_state)
        print(game.environment.current_store_state)

        action1 = np.array(game.environment.player_1_actions, dtype=np.int32)
        action2 = np.array(game.environment.player_2_actions, dtype=np.int32)
        print(obs, obs.shape)

        if game.rules.stop_round():
            game_winner = sp.mode(np.array(game.environment.game_winner_list))[0]
            winners.append(int(game_winner))
        # np.savetxt(f"./data/game_state_{simulation}.txt", obs, fmt='%d', delimiter=",")
        # np.savetxt(f"./actions/player_1/action_state_{simulation}.txt", action1, fmt='%d', delimiter=",")
        # np.savetxt(f"./actions/player_2/action_state_{simulation}.txt", action2, fmt='%d', delimiter=",")

    except Exception as e:
        print(f"An error occured in simulation {simulation}: {e}")
        raise

np.savetxt(f"./labels/game_state.txt", np.array(winners, dtype=np.int32), fmt='%d', delimiter=",")

