import numpy as np
import scipy.stats as sp
import concurrent.futures
from env import Board, GameState, Player, RuleEngine, GameController

# Define a function to run the game simulation
def run_game_simulation(simulation):
    game = GameController()
    game.game(num_of_rounds=6)
    return game

num_of_simulations = 500
winners = []

# Start the simulation from the last successful one
for simulation in range(num_of_simulations):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit the simulation to the executor and run it in a separate thread
        future = executor.submit(run_game_simulation, simulation)
        try:
            # Set a timeout for the simulation (e.g., 10 seconds)
            game = future.result(timeout=10)
            obs = game.environment.all_states_np[:, 1:].astype(np.int32)
            print(obs, obs.shape)
            game_winner = sp.mode(np.array(game.environment.game_winner_list))[0]
            winners.append(game_winner)
            np.savetxt(f"./data/game_state_{simulation}.txt", obs, fmt='%d', delimiter=",")
        except concurrent.futures.TimeoutError:
            print(f"Simulation {simulation} timed out.")
        except Exception as e:
            print(f"An error occurred in simulation {simulation}: {e}")

# Save the winners after all simulations
np.savetxt(f"./labels/game_state_{simulation}.txt", np.array(winners, dtype=np.int32))
