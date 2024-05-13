import numpy as np

class RuleEngine:
    """
    Manages and applies the rules of the game, determining the validity of actions and the conditions for stopping rounds and the game.

    This class is responsible for checking if actions taken by players are valid, determining when a round or the game should end based on the state of the board and game rules.

    Attributes:
        B (Board): An instance of the Board class, which represents the game board and its state.
        state (GameState): An instance of the GameState class, used to track and manage the current state of the game.
        round (int): The current round number in the game.
        turn (int): The current turn number within the round.
        actions (np.ndarray): An array of possible action indices, usually representing pits on the board.

    Methods:
        __init__(board, state): Initializes the rule engine with references to the game board and state.
        is_valid_action(action, player): Determines if a given action is valid for a player at the current state.
        stop_round(): Checks if the current round should stop based on the game board's state.
        stop_game(): Determines if the game should be stopped, typically based on a win condition or other rule.
        check_round_end(): Verifies if the round should end, generally due to a specific board state.
    """

    def __init__(self, board, state):
        """
        Initializes the RuleEngine with necessary components of the game.

        Parameters:
            board (Board): The game board which holds the state and configuration of seeds and pits.
            state (GameState): The object managing the overall state of the game, including scores and turns.
        """
        self.B = board
        self.state = state
        self.round = 1
        self.turn = 1
        self.actions = np.arange(12)

    def is_valid_action(self, action, player):
        """
        Checks if a specific action is valid for a given player based on the current game state.

        Parameters:
            action (int): The action to check, typically an index representing a pit.
            player (int): The player number performing the action.

        Returns:
            bool: True if the action is valid, False otherwise.
        """
        # check if the action is valid for a player
        return action in self.state.possible_moves(player)

    def stop_round(self):
        """
        Determines if the current round should be stopped, typically when no seeds are left to play.

        Returns:
            bool: True if the board is empty and the round should stop, False otherwise.
        """
        return np.sum(self.B.board, axis=None) == 0

    def stop_game(self, num_rounds):
        """
        Checks if the game should end, usually based on a significant victory condition or rule, such as a player achieving a specific territory count.

        Returns:
            bool: True if the game should end, for example, if a player's territory count reaches a set threshold.
        """
        return self.B.territory_count[1] == 11 or self.B.territory_count[0] == 11
        # return self.B.territory_count[1] == 12 or self.B.territory_count[0] == 12 or self.round == num_rounds