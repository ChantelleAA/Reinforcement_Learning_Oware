from env import GameController
import numpy as np
import scipy.stats as sp

import warnings
warnings.filterwarnings('ignore')
TF_ENABLE_ONEDNN_OPTS=0

# num_of_simulations = 200
# winners = []
# for simulation in range(1, num_of_simulations+1):
#     try:
#         game = GameController()
#         game.game(num_of_rounds=6)
#         obs = game.environment.all_states_np[:, 1:].astype(np.int32)
#         print(game.environment.current_board_state)
#         print(game.environment.current_store_state)

#         action1 = np.array(game.environment.player_1_actions, dtype=np.int32)
#         action2 = np.array(game.environment.player_2_actions, dtype=np.int32)
#         print(obs, obs.shape)
#         game_winner = sp.mode(np.array(game.environment.game_winner_list))[0]
#         winners.append(int(game_winner))
#         np.savetxt(f"./data/game_state_{simulation}.txt", obs, fmt='%d', delimiter=",")
#         np.savetxt(f"./actions/player_1/action_state_{simulation}.txt", action1, fmt='%d', delimiter=",")
#         np.savetxt(f"./actions/player_2/action_state_{simulation}.txt", action2, fmt='%d', delimiter=",")

#     except Exception as e:
#         print(f"An error occured in simulation {simulation}: {e}")
#         continue
        

# np.savetxt(f"./labels/game_state.txt", np.array(winners, dtype=np.int32), fmt='%d', delimiter=",")

num_of_simulations = 2000
winners = []
exc=[]
for simulation in range(1, num_of_simulations+1):
    try:
        game = GameController()
        game.game(num_of_rounds=6)

        obs = game.environment.all_states_np[:, 1:].astype(np.int32)
        game_winner = sp.mode(np.array(game.environment.game_winner_list))[0]
        winners.append(int(game_winner))
        action1 = np.array(game.environment.player_1_actions, dtype=np.int32)
        action2 = np.array(game.environment.player_2_actions, dtype=np.int32)



        # Save features and labels only if the simulation completes successfully
        np.savetxt(f"./data/game_state_{simulation}.txt", obs, fmt='%d', delimiter=",")
        np.savetxt(f"./actions/player_1/action_state_{simulation}.txt", action1, fmt='%d', delimiter=",")
        np.savetxt(f"./actions/player_2/action_state_{simulation}.txt", action2, fmt='%d', delimiter=",")
    except Exception as e:
        print(f"An error occurred in simulation {simulation}: {e}")
        exc.append(simulation)
        # break
        continue  # Skip saving any data for this simulation

# Save labels after all simulations have completed
np.savetxt(f"./labels/game_state.txt", np.array(winners, dtype=np.int32), fmt='%d', delimiter=",")
print(exc)
