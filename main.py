from env import Board, GameState, Player, RuleEngine, GameController
import numpy as np

num_of_rounds = 6
game = GameController()
game.game(num_of_rounds)
arr = np.array(game.environment.game_state).reshape()
print(arr.shape)
print(arr)

# arr.tofile('./data/environment_1.csv', sep = ',')
