import numpy as np
import random
from further_refined import GameController

num_of_rounds = 20
game = GameController()
game.game(num_of_rounds)
print(game.environment.game_states)
print(f"{game.environment.current_store_state=}")
print(f"{game.environment.current_territory_count=}")
arr = game.environment.game_states
arr.tofile('./data/environment_1.csv', sep = ',')


def save_game_state(self):
    """
    Saves the game state to the game_states matrix after each turn.
    Each row is padded or truncated to length 'k'.
    Each row format: [board state, store state, territory count, ...] for each round.
    """
    # Flatten current state components into a single array
    current_state = np.concatenate([
        self.B.board.flatten(),
        self.B.stores,
        self.B.territory_count
    ])
    num_elements = len(current_state)
    total_required_per_round = self.max_turns * num_elements  # Total elements needed per round

    # Check if current round storage is initialized, if not, initialize
    if self.rounds_completed >= self.game_states.shape[1]:
        # Expand game_states array to accommodate more rounds if necessary
        new_columns = self.rounds_completed + 1 - self.game_states.shape[1]
        additional_cols = np.zeros((self.max_turns, num_elements * new_columns), dtype=int)
        self.game_states = np.hstack((self.game_states, additional_cols))

    # Compute the index for the current round
    round_index = self.rounds_completed % self.max_rounds

    # Save the current state into the appropriate slice of the game_states array
    current_turn_index = self.turns_completed % self.max_turns  # Current turn within the round
    start_idx = round_index * num_elements
    end_idx = start_idx + num_elements

    # Handle potential overflow if this is the last turn that can be recorded
    if current_turn_index < self.max_turns:
        self.game_states[current_turn_index, start_idx:end_idx] = current_state

