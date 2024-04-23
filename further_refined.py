import numpy as np
import random

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
        self.territory_count = np.array((self.ncols, self.ncols), dtype=int)  # Initial territory counts (incorrectly sized and should be reviewed)
        self.board_indices = list(np.ndindex(self.nrows, self.ncols))  # List of board indices for navigation
        self.player_territories = [self.board_indices[5::-1], self.board_indices[6:]]  # Define territories for each player
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
        self.board = 4 * np.ones((self.nrows, self.ncols))

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

    # def distribute_seeds(self, action:int):
    #     """
    #     Distributes seeds from a selected pit and captures seeds according to the game rules.
    #     """
    #     pit_index = self.action2pit(action)
    #     seeds = self.board[pit_index]  # pick seeds from current pit
    #     self.board[pit_index] = 0  # set seeds from current pit to zero
        
    #     while seeds > 0 : # iterate over number of seeds picked 
    #         global next_index
    #         action = (action + 1) % 12 # navigate to the next pit
    #         next_index = self.board_format[action] 
    #         self.board[next_index] += 1 # drop seed at the point found
    #         seeds -= 1 # update number of seeds in hand
    #         # print(self.turns_completed)
    #         if self.turns_completed > 1:
    #             self.capture_seeds(action)

    #             if np.sum(self.board) < 4:
    #                 self.board = np.zeros((self.nrows, self.ncols))
    #                 break

    #         if seeds == 0:    
    #             self.capture_seeds(action, during_game=False)

    #         # update current index
    #         pit_index = next_index
    #     self.turns_completed +=1 
    #     print(f"End distribution, last seed is {self.board[next_index]}")
    #     return next_index


    # def capture_seeds(self, action:int, during_game = True):

    #     """ This function is crafted to execute the capture process in the game"""

    #     pit_index = self.action2pit(action)  # for the given action value here we want to check the viability of a seed capture
    #     player_idx = self.current_player - 1  # Get the player's id
    #     print(f"{player_idx=}")
    #     print(f"{self.current_player=}")
    #     if self.get_seeds(action) == 4 and np.sum(self.board, axis=None) > 8: # Condition checking if there is 4 in the pit where the player just dropped his seed
    #         if during_game: # if this capture is happening during the course of 

    #             if pit_index in self.player_territories[player_idx]: # if the pit is the player's territory
    #                 self.board[pit_index] = 0  # remove seeds from the board
    #                 self.stores[player_idx] += 4 # add the seeds to the store of the player

    #             else: # otherwise if it is in the territory of the opponent
    #                 self.board[pit_index] = 0 # remove seeds from the board 
    #                 self.stores[1 - player_idx] += 4 # the opponent gets the seeds

    #         else: # if its not during the course but at the end

    #             if self.board[pit_index] == 4: 
    #                 self.board[pit_index] = 0
    #                 self.stores[pit_index] += 4

    #     elif self.get_seeds(action) == 4 and np.sum(self.board, axis=None) <= 8:
    #         print(f"Now the board has just 8 seeds left and player {self.current_player} has captured the first 4, so he gets the rest.")
    #         self.stores[player_idx] += 8
    #         self.board[self.board > 0] = 0
    #         print(f"Let's start a new round, taking note of our winner for this round!!!")


#######################################################################################

########################################################################################


class Player():
    def __init__(self, board, state):
        self.B = board
        self.state = GameState(self.B)
        self.stores = self.B.stores
        self.territories = self.B.board_indices

    def territory_count(self):
        return self.B.territory_count

    def territory_indices(self, player):
        player_id = player - 1
        if player == 1:
            return self.territories[:6]
        elif player == 2:
            return self.territories[6:]

    def is_round_winner(self, player):
        player_id = player - 1
        opponent_id = 1 - player_id
        if self.stores[player_id] > self.stores[opponent_id]:
            return True
        return False

    def player_step(self, start_action):
        final_idx = self.state.distribute_seeds(start_action)
        print(final_idx)
        print(self.B.board)
        seeds = self.B.board[final_idx]
        action = start_action
        while self.B.board[final_idx] > 1:
            final_idx = self.state.distribute_seeds(action)
            print(self.B.board)
            action = self.B.board_format.index(final_idx)
        return self.B.board
    



#######################################################################################

########################################################################################


class GameState:
    """
    Manages the state of the game, including tracking the current board, stores, territory counts,
    and the history of actions taken throughout the game.

    This class is essential for maintaining a record of the game's progress and status, allowing
    for state retrieval at any point, which is critical for both game logic and potential UI rendering
    or game replays.

    Attributes:
        B (Board): An instance of the Board class which represents the current state of the game board.
        total_games_played (int): The total number of games played during this session.
        games_won (np.ndarray): Array tracking the number of games won by each player.
        current_board_state (np.ndarray): Snapshot of the current state of the game board.
        current_store_state (np.ndarray): Snapshot of the current seeds stored by each player.
        current_territory_count (np.ndarray): Snapshot of the current territories held by each player.
        rounds_completed (int): The number of rounds completed in the current game.
        turns_completed (int): The number of turns completed in the current round.
        win_list (list): A list of players who have won each round.
        round_winner (int): The player who won the most recent round.
        current_player (int): The player who is currently taking their turn.
        other_player (int): The player who is not currently taking their turn.
        actions (np.ndarray): Array of possible actions players can take (typically pit indices).
        game_actions (list): A comprehensive list of all actions taken in the game.
        player_1_actions (list): A list of actions specifically taken by player 1.
        player_2_actions (list): A list of actions specifically taken by player 2.
        max_turns (int): The maximum allowed turns per game to prevent infinite loops.
        max_rounds (int): The maximum allowed rounds per game.
        game_states (np.ndarray): A matrix to store detailed game state after each turn for analysis or replay.

    Methods:
        __init__(board, max_turns=2000, max_rounds=7): Initializes a new game state with a reference to the game board.
        update_win_list(player): Adds the winning player of a round to the win list.
        possible_moves(player): Returns a list of possible moves for the specified player based on the current board state.
        save_actions(player, action): Records an action taken by a player for historical tracking.
        save_game_state(): Stores the detailed current state of the game in the game_states matrix for later retrieval or analysis.
    """

    def __init__(self, board, max_turns=2000, max_rounds=7):
        """
        Initializes the game state with a reference to the board and sets up initial tracking variables.

        Parameters:
            board (Board): The game board instance.
            max_turns (int): Maximum number of turns to prevent infinite game loops.
            max_rounds (int): Maximum number of rounds in a game session.
        """   
        self.B = board
        self.total_games_played = 0
        self.games_won = np.array([0, 0])
        self.current_board_state = self.B.board
        self.current_store_state = self.B.stores
        self.current_territory_count = self.B.territory_count
        self.rounds_completed = 0
        self.turns_completed = 0
        self.win_list = []
        self.round_winner = 0
        self.current_player = self.B.current_player
        self.other_player = self.B.current_player
        self.actions = np.arange(12)
        self.game_actions = []
        self.player_1_actions = []
        self.player_2_actions = []      
        self.max_turns = max_turns
        self.max_rounds = max_rounds
        self.game_states = np.zeros((max_turns, max_rounds * (self.B.nrows * self.B.ncols + 2 * self.B.total_stores + 2)), dtype=int)

    def update_win_list(self, player):
        if player == self.round_winner:
            self.win_list += [player]

    def possible_moves(self, player):
        player_id = player-1
        open_moves = []
        for i in self.actions:
            pit_index = self.B.action2pit(i)
            if pit_index in self.B.player_territories[player_id]:
                if self.B.board[pit_index] != 0 :
                    open_moves.append(i)
        return open_moves

    def save_actions(self, player, action):
        if player == 1:
            self.player_1_actions.append(action)
            self.game_actions.append(action)
        elif player == 2:
            self.player_2_actions.append(action)
            self.game_actions.append(action)

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

    def switch_player(self):
        current_player = self.current_player
        other_player = self.other_player
        print(f"Initial state: {self.current_player=} and {self.other_player=}")
        self.current_player = other_player
        self.other_player = current_player
        print(f"Final state: {self.current_player=} and {self.other_player=}")


    def distribute_seeds(self, action:int):
        """
        Distributes seeds from a selected pit and captures seeds according to the game rules.
        """
        pit_index = self.B.action2pit(action)
        seeds = self.B.board[pit_index]  # pick seeds from current pit
        self.B.board[pit_index] = 0  # set seeds from current pit to zero
        
        while seeds > 0 : # iterate over number of seeds picked 
            global next_index
            action = (action + 1) % 12 # navigate to the next pit
            next_index = self.B.board_format[action] 
            self.B.board[next_index] += 1 # drop seed at the point found
            seeds -= 1 # update number of seeds in hand
            # print(self.B.turns_completed)
            if self.B.turns_completed > 1:
                self.capture_seeds(action)

                if np.sum(self.B.board) < 4:
                    self.B.board = np.zeros((self.B.nrows, self.B.ncols))
                    break

            if seeds == 0:    
                self.capture_seeds(action, during_game=False)

            # update current index
            pit_index = next_index
        self.turns_completed +=1 
        print(f"End distribution, last seed is {self.B.board[next_index]}")
        return next_index


    def capture_seeds(self, action:int, during_game = True):

        """ This function is crafted to execute the capture process in the game"""

        pit_index = self.B.action2pit(action)  # for the given action value here we want to check the viability of a seed capture
        player_idx = self.current_player - 1  # Get the player's id
        print(f"{player_idx=}")
        print(f"{self.current_player=}")
        if self.B.get_seeds(action) == 4 and np.sum(self.B.board, axis=None) > 8: # Condition checking if there is 4 in the pit where the player just dropped his seed
            if during_game: # if this capture is happening during the course of a game,
                # print(f"{self.B.get_seeds(action) == 4 and np.sum(self.B.board, axis=None) > 8 = }")

                if pit_index in self.B.player_territories[player_idx]: # if the pit is the player's territory
                    self.B.board[pit_index] = 0  # remove seeds from the board
                    self.B.stores[player_idx] += 4 # add the seeds to the store of the player

                else: # otherwise if it is in the territory of the opponent
                    self.B.board[pit_index] = 0 # remove seeds from the board 
                    self.B.stores[1 - player_idx] += 4 # the opponent gets the seeds

            else: # if its not during the course but at the end

                if self.B.board[pit_index] == 4: 
                    self.B.board[pit_index] = 0
                    self.B.stores[player_idx] += 4

        elif self.B.get_seeds(action) == 4 and np.sum(self.B.board, axis=None) <= 8:
            print(f"Now the board has just 8 seeds left and player {self.current_player} has captured the first 4, so he gets the rest.")
            self.B.stores[player_idx] += 8
            self.B.board[self.B.board > 0] = 0
            print(f"Let's start a new round, taking note of our winner for this round!!!")


#######################################################################################

########################################################################################


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
        if action in self.state.possible_moves(player):
            return True
        return False

    def stop_round(self):
        """
        Determines if the current round should be stopped, typically when no seeds are left to play.

        Returns:
            bool: True if the board is empty and the round should stop, False otherwise.
        """
        if np.sum(self.B.board, axis=None) == 0:
            return True

    def stop_game(self):
        """
        Checks if the game should end, usually based on a significant victory condition or rule, such as a player achieving a specific territory count.

        Returns:
            bool: True if the game should end, for example, if a player's territory count reaches a set threshold.
        """
        if self.B.territory_count[1] == 12 or self.B.territory_count[0] == 12:
            return True
        return False

    def check_round_end(self):
        """
        Verifies whether the current round should end based on the board's state, typically when no seeds are left to distribute.

        Returns:
            bool: True if no seeds are left on the board, indicating the end of the round.
        """        
        if np.sum(self.B.board, axis = None)==0:
            return True
        

#######################################################################################

########################################################################################

class GameController:
    """
    Manages the overall flow of an Oware game, controlling game rounds, player actions, and the game state.

    This controller sets up the game, decides the starting player, manages the sequence of turns within each round,
    and checks for end-of-game conditions. It interacts with the `Board`, `Player`, and `GameState` classes to
    execute the game logic and maintain the state of the game.

    Attributes:
        n_players (int): Number of players in the game, typically two.
        board (Board): An instance of the Board class representing the game board.
        player (Player): An instance of the Player class to manage player actions.
        environment (GameState): An instance of the GameState class to track and store the game's state.
        rules (RuleEngine): An instance of the RuleEngine class to enforce game rules.
        max_turns (int): The maximum number of turns allowed to prevent infinite loops.

    Methods:
        __init__(num_of_rounds=7): Initializes the game controller with a specified number of rounds.
        starting_player(how="random"): Determines which player starts a round.
        choose_action_player(player): Randomly selects a valid action for the given player.
        game(): Executes the main game loop, handling the progression of rounds and managing game state updates.

    The game() method is the core function that runs the game loop, orchestrating the game by managing rounds,
    processing player actions, updating the state, and determining when the game ends based on the rules defined
    in the RuleEngine.

    Example:
        num_of_rounds = 7
        game = GameController(num_of_rounds)
        game.game()  # Start the game loop
    """
    def __init__(self, num_of_rounds=7):
        self.n_players = 2
        self.board = Board()
        self.environment = GameState(self.board)
        self.player = Player(self.board, self.environment)
        self.rules = RuleEngine(self.board, self.environment)
        self.max_turns = 2000

    def starting_player(self, how="random"):
        """
        Determines the starting player of a round based on the specified method.

        Parameters:
            how (str): Method to determine the starting player. Options are "random" or "last_winner".

        Returns:
            int: The player number who will start the round.
        """
        if how == "random":
            starter = random.sample([1, 2], 1)
            return starter[0]
        elif how == "last_winner" and len(self.environment.win_list)!=0:
            starter = self.environment.win_list[-1]
            return starter
        else:
            starter = 1
            return starter


    def choose_action_player(self, player):
        """
        Selects a valid action for the player randomly from the possible moves.

        Parameters:
            player (int): The player number for whom to select the action.

        Returns:
            int: The action index chosen for the player.
        """
        return random.sample(self.environment.possible_moves(player), 1)[0]



    def game(self):

        """
        Executes the game loop, managing rounds and player turns until game completion conditions are met.

        This loop orchestrates the game flow by:
        1. Determining which player starts each round, alternating starting players based on random choice for the first round and the winner of the previous round for subsequent rounds.
        2. Incrementing the round count and processing individual turns within each round until there are no seeds left on the board or a stopping condition is triggered.
        3. Each player's turn involves choosing a valid action (pit from which to distribute seeds), distributing seeds from the selected pit, and potentially capturing seeds from the board.
        4. After each action, the game state is saved to record the actions and board state.
        5. The loop checks if the round should stop (based on the board state or other game rules), and if so, breaks out of the round loop to start a new round.
        6. After the completion of each round, the loop checks for game end conditions such as reaching a maximum number of rounds or a specific condition that defines the end of the game.
        7. Updates the game statistics, including the number of games won by each player and updates to territory counts based on the results of the round.
        8. Resets the board to its initial state at the end of each round and prepares for the next round if the game has not reached its conclusion.

        Parameters:
        - num_of_rounds (int): Maximum number of rounds to be played.
        - self.max_turns (int): Maximum number of turns allowed, to prevent potentially infinite games.

        Outputs:
        - Prints the current round and turn, the state of the board after each turn, and messages at the end of each round and game.
        - Updates internal state to track rounds, player scores, and other game metrics.

        Side effects:
        - Modifies the internal state of the board, players, and game controller to reflect the progression of the game.
        """

        
        while self.rules.round < num_of_rounds and self.board.turns_completed < self.max_turns:
            print(f" Start round {self.rules.round}")
            # Decision on who starts the round
            if self.rules.round == 1:
                self.environment.current_player = self.starting_player("random")
                self.environment.other_player = 1 if self.environment.current_player == 2 else 2
                print(self.environment.current_player)
            else:
                self.environment.current_player = self.starting_player("last_winner")
                self.environment.other_player = 1 if self.environment.current_player == 2 else 2
            
            # increment the number of rounds after the choice of who starts the round
            self.rules.round +=1
            
            # Implement a round
            while np.sum(self.board.board, axis = None) > 0:
                print("Total seeds on board", np.sum(self.board.board, axis = None) )
                action_c = self.choose_action_player(self.environment.current_player)
                self.player.player_step(action_c)
                self.environment.save_actions(self.environment.current_player, action_c)

                self.environment.save_game_state()
                print(f"End of turn for player {self.environment.current_player}")
                # print(f"GameController Board = {self.board.board}")
                self.environment.switch_player()

                if self.rules.stop_round() == True:             
                    break

                action_o = self.choose_action_player(self.environment.other_player)
                self.player.player_step(action_o)
                self.environment.save_actions(self.environment.other_player, action_o)
                self.environment.save_game_state()
                print(f"End of turn for player {self.environment.other_player}")
                # print(f"GameController Board = {self.board.board}")
                print(self.board.turns_completed)
                
                if self.board.turns_completed == 2000:
                    print(self.board.turns_completed,"2000")
                    break

            print(f"End round")
            # print(f" Final GameController Board = {self.board.board}")
            print(self.board.turns_completed)

            if self.board.turns_completed == 2000:
                print(self.board.turns_completed ,"2000")
                break

            if self.board.stores[0] > self.board.stores[1]:
                self.environment.games_won[0] +=1
                self.environment.win_list += [1]
                self.board.territory_count[0] +=1
                self.board.territory_count[1] -=1
                # write function to update territories

            elif self.board.stores[0] < self.board.stores[1]:
                self.environment.games_won[1] +=1
                self.environment.win_list += [2]
                self.board.territory_count[1] +=1
                self.board.territory_count[0] -=1

            self.environment.rounds_completed += 1
            print(self.environment.rounds_completed)
            print(f"Territory_status: {self.board.territory_count}")
            self.board.reset_board()
            
            self.board.stores = np.array([0, 0])

            if self.rules.stop_game() == True:
                break


num_of_rounds = 20
game = GameController(num_of_rounds)
game.game()
print(game.environment.game_states)
print(f"{game.environment.current_store_state=}")
print(f"{game.environment.current_territory_count=}")

