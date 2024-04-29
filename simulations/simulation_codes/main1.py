from env import Board, GameState, Player, RuleEngine, GameController
import numpy as np
import scipy.stats as sp
# num_of_rounds = 6
# game = GameController()
# game.game(num_of_rounds)
# print(game.environment.game_state)
# print(f"{game.environment.current_store_state=}")
# print(f"{game.environment.current_territory_count=}")
# arr = game.environment.game_state
# arr.tofile('./data/environment_1.csv', sep = ',')

# num_of_rounds = 6
# game = GameController()
# game.game(num_of_rounds)
# arr = np.array(game.environment.game_state).reshape(2, -1)
# turns =  len(game.environment.all_states)
# # arr2 = np.array(game.environment.all_states).reshape(-1, turns)
# arr2 = game.environment.all_states_np[:, 1:]
# simulation = 1
# k = 6
# winners = []
# game = GameController()
# game.game(num_of_rounds=k)

# obs = game.environment.all_states_np[:, 1:].astype(np.int32)
# print(obs, obs.shape)
# game_winner = sp.mode(np.array(game.environment.game_winner_list))[0]
# winners.append(int(game_winner))
# np.savetxt(f"./data/game_state_{simulation}.txt", obs)


# print(game.environment.game_winner_list)
# print(f"Game winner is Player {sp.mode(np.array(game.environment.game_winner_list))[0]}")
# # arr.tofile('./data/environment_1.csv', sep = ',')
# with np.printoptions(threshold=np.inf):
#     print(obs)


num_of_simulations = 500
winners = []
for simulation in range(1, num_of_simulations+1):
    try:
        game = GameController()
        game.game(num_of_rounds=6)
        obs = game.environment.all_states_np[:, 1:].astype(np.int32)
        print(game.environment.current_board_state)
        print(game.environment.current_store_state)

        action1 = np.array(game.environment.player_1_actions, dtype=np.int32)
        action2 = np.array(game.environment.player_2_actions, dtype=np.int32)
        print(obs, obs.shape)
        game_winner = sp.mode(np.array(game.environment.game_winner_list))[0]
        winners.append(int(game_winner))
        np.savetxt(f"./data/game_state_{simulation}.txt", obs, fmt='%d', delimiter=",")
        np.savetxt(f"./actions/player_1/action_state_{simulation}.txt", action1, fmt='%d', delimiter=",")
        np.savetxt(f"./actions/player_2/action_state_{simulation}.txt", action2, fmt='%d', delimiter=",")

    except Exception as e:
        print(f"An error occured in simulation {simulation}: {e}")

np.savetxt(f"./labels/game_state.txt", np.array(winners, dtype=np.int32), fmt='%d', delimiter=",")
