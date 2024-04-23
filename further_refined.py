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
        self.current_player = 1  # Index of the current player
        self.other_player = 2  # Index of the opposing player
        self.turns_completed = 0  # Counter for the number of turns completed

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

    def distribute_seeds(self, action:int):
        """
        Distributes seeds from a selected pit and captures seeds according to the game rules.
        """
        pit_index = self.action2pit(action)
        seeds = self.board[pit_index]  # pick seeds from current pit
        self.board[pit_index] = 0  # set seeds from current pit to zero
        while seeds > 0 : # iterate over number of seeds picked 
            global next_index
            action = (action + 1) % 12 # navigate to the next pit
            next_index = self.board_format[action] 
            self.board[next_index] += 1 # drop seed at the point found
            # self.capture_seeds(action)
            seeds -= 1 # update number of seeds in hand
            print(self.turns_completed)
            if self.turns_completed > 1:
                print("Enter seed capture")
                self.capture_seeds(action)
                if np.sum(self.board) < 4:
                    self.board = np.zeros((self.nrows, self.ncols))
                    break
            if seeds == 0:    
                self.capture_seeds(action, during_game=False)

        self.turns_completed +=1 
        
        return next_index


    def capture_seeds(self, action:int, during_game = True):

        """ This function is crafted to execute the capture process in the game"""

        pit_index = self.action2pit(action)  # for the given action value here we want to check the viability of a seed capture

        player_idx = self.current_player - 1  # Get the player's id
        # print(f"{self.get_seeds(action) == 4 = }")
        # print(f"{np.sum(self.board, axis=None) > 8 = }")
        # print(f"{np.sum(self.board, axis=None) == 8 = } but {np.sum(self.board, axis=None)=} evidence: {self.board}")

        if self.get_seeds(action) == 4 and np.sum(self.board, axis=None) > 8: # Condition checking if there is 4 in the pit where the player just dropped his seed
            if during_game: # if this capture is happening during the course of a game,
                print(f"{self.get_seeds(action) == 4 and np.sum(self.board, axis=None) > 8 = }")
                # print(f"Seed capture during game")

                if pit_index in self.player_territories[player_idx]: # if the pit is the player's territory

                    # print(f" Home capture for player {self.current_player}")
                    self.board[pit_index] = 0  # remove seeds from the board
                    self.stores[player_idx] += 4 # add the seeds to the store of the player

                    

                    # print(f"Board after capture: \n {self.board} \n Player stores after capture: \n {self.stores}")

                else: # otherwise if it is in the territory of the opponent

                    # print(f"Opponent, player {self.other_player} has captured during player {self.current_player}'s turn")
                    self.board[pit_index] = 0 # remove seeds from the board 
                    self.stores[1 - player_idx] += 4 # the opponent gets the seeds



                    # print(f"Board after capture: \n {self.board} \n Player stores after capture: \n {self.stores}")

            else: # if its not during the course but at the end

                # print(f"Seed capture at the end of a turn\n"*8)

                if self.board[pit_index] == 4: 
                    self.board[pit_index] = 0
                    self.stores[pit_index] += 4



                    # print(f"Board after capture: \n {self.board} \n Player stores after capture: \n {self.stores}")
        # elif self.get_seeds(action) == 4 and np.sum(self.board, axis=None) == 8:

        elif self.get_seeds(action) == 4 and np.sum(self.board, axis=None) <= 8:

            print(f"Now the board has just 8 seeds left and player {self.current_player} has captured the first 4, so he gets the rest.")

            self.stores[player_idx] += 8

            # print(f"Board before: \n {self.board}")

            # print(f"Stores status update: {self.stores}")

            self.board[self.board > 0] = 0
            
            # print(f"Board after that final capture. Also, round ended.")
            print(f"Let's start a new round, taking note of our winner for this round!!!")



#######################################################################################

########################################################################################
class Player():
    def __init__(self, board):
        self.B = board
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
        # print(f'{start_action=}')
        # print(f'{type(start_action) = }')
        final_idx = self.B.distribute_seeds(start_action)
        # print(self.B.board)
        seeds = self.B.board[final_idx]
        # print(seeds)
        action = start_action
        # print(action)
        while self.B.board[final_idx] != 0:
            final_idx = self.B.distribute_seeds(action)
            action = self.B.board_format.index(final_idx)
            # if self.B.turns_completed == 2000:
            #     print(self.B.turns_completed)
            #     break
            print(self.B.board)
            # print(final_idx)
        print(f"Player step ends here")
        return self.B.board

#######################################################################################

########################################################################################

class GameState:
    def __init__(self, board, max_turns=2000, max_rounds=7):
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
        self.current_player = 0
        self.other_player = 0
        self.actions = np.arange(12)
        self.game_actions = []
        self.player_1_actions = []
        self.player_2_actions = []      
        self.max_turns = max_turns
        self.max_rounds = max_rounds
        self.game_states = np.zeros((max_turns, max_rounds * (self.B.nrows * self.B.ncols + 2 * self.B.total_stores + 2)), dtype=int)

    # def __init__(self, board, max_turns=2000, max_rounds=7):
    #     self.B = board
    #     self.total_games_played = 0
    #     self.games_won = np.zeros(2, dtype=int)
    #     self.current_board_state = []
    #     self.current_store_state = []
    #     self.current_territory_count = []
    #     self.rounds_completed = 0
    #     self.win_list = []
    #     self.current_player = 0
    #     self.other_player = 1
    #     self.actions = np.arange(12)
    #     self.max_turns = max_turns
    #     self.max_rounds = max_rounds
    #     self.game_states = np.zeros((max_turns, max_rounds * (self.B.nrows * self.B.ncols + 2 * self.B.total_stores + 2)), dtype=int)  # Example setup


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



#######################################################################################

########################################################################################
class RuleEngine:
    def __init__(self, board, state):
        self.B = board
        self.state = state
        self.round = 1
        self.turn = 1
        self.actions = np.arange(12)

    def is_valid_action(self, action, player):
        # check if the action is valid for a player
        if action in self.state.possible_moves(player):
            return True
        return False

    def stop_round(self):
        if np.sum(self.B.board, axis=None) == 0:
            return True

    def stop_game(self):
        if self.B.territory_count[0] == 12 :
            return True
        return False

    def check_round_end(self):
        if np.sum(self.B.board, axis = None)==0:
            return True
        

#######################################################################################

########################################################################################

class GameController:
    def __init__(self, num_of_rounds=7):
        self.n_players = 2
        self.board = Board()
        self.player = Player(self.board)
        self.environment = GameState(self.board)
        self.rules = RuleEngine(self.board, self.environment)
        self.max_turns = 2000

    def starting_player(self, how="random"):
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
        return random.sample(self.environment.possible_moves(player), 1)[0]



    def game(self):

        """ Game implementation here attempts to implement a full game consisting of 6 or more rounds.
            A game consists of several rounds, after hich the following variables in the game are updated.
            - The board is reset.
            - The stores for each player are reset
            - The number of rounds completed are incremented by 1
            - A winner is take based on the number of seds in store and is documented
            - The territory count of both players is recalculated
            - The territory indices of both players is rearranged
            - The win_list is updated
            - The actions taken at each point in the game are kept in a array wwith 12 slots for each round.
            - A Game round is completed after several turns. Here is what happpens in the game after every turn is completed :
                - The number of turns completed is incremented by 1
                - add actions to list of actions in the game
                - add action of each player to his individual list of actions
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
            print(f"Territory_status: {self.board.territory_count}\n\n\n\n\n\n")
            self.board.reset_board()
            
            self.board.stores = np.array([0, 0])

            if self.rules.stop_game() == True:
                break


num_of_rounds = 20
game = GameController(num_of_rounds)
game.game()
print(game.environment.game_states)