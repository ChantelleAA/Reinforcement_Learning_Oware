import numpy as np

class Board():
    """
    Manages the Oware board layout, actions, and score tracking. This class is central to the Oware nam-Nam environment and interacts directly with the game logic.
    """

    def __init__(self):
        self.nrows = 2  # Number of rows on the board
        self.ncols = 6  # Number of columns on the board
        self.total_stores = 2  # Number of places where players' captured seeds are stored
        self.board = 4 * np.ones((self.nrows, self.ncols), dtype=int)  # Initializes the board with 4 seeds in each pit
        self.stores = np.zeros(self.total_stores, dtype=int)  # Seed storage for capturing
        self.board_indices = list(np.ndindex(self.nrows, self.ncols))  # List of board indices for navigation
        self.player_territories = [self.board_indices[5::-1], self.board_indices[6:]]  # Define territories for each player
        self.territory_count = np.array((self.ncols, self.ncols), dtype=int)  # Initial territory counts (incorrectly sized and should be reviewed)
        self.actions = np.arange(0, 12, dtype=int)
        self.n_players = 2  # Static number of players
        self.turns_completed = 0  # Counter for the number of turns completed  
        self.current_player = 1
        self.other_player = 2

    @property
    def total_seeds(self):
        """Returns the total count of seeds currently on the board."""
        return np.sum(self.board, axis=None)

    @property
    def board_format(self):
        return self.board_indices[5::-1] + self.board_indices[6:]

    def zero_rows(self):
        """Check if any row on the board is empty and returns a boolean array indicating such rows."""
        return np.all(self.board == 0, axis=1)

    def reset_board(self):
        """Resets the board to the initial state with 4 seeds in each pit."""
        print("Enter a new board.")
        self.board = 4 * np.ones((self.nrows, self.ncols))
        return self.board

    def action2pit(self, action):
        """Converts an action index into a board index."""
        return self.board_format[action]

    def get_seeds(self, action):
        """Returns the number of seeds at the specified action's pit."""
        pit_index = self.action2pit(action=action)
        return self.board[pit_index]

    def set_seeds(self, action, new_value):
        """Sets the number of seeds at the specified action's pit to new_value."""
        pit_index = self.action2pit(action=action)
        self.board[pit_index] = new_value

    def update_seeds(self, action, new_value):
        """Updates the number of seeds at the specified action's pit by adding new_value."""
        pit_index = self.action2pit(action=action)
        self.board[pit_index] += new_value