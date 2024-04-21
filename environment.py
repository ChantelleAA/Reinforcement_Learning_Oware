import numpy as np
import gym
import random

class Board():
    def __init__(self):
        self.nrows = 2
        self.ncols = 6
        self.total_stores = 2
        self.board = 4 * np.ones((self.nrows, self.ncols))

        self.total_territories = self.nrows * self.ncols
        self.stores = np.zeros((self.total_stores,))
        self.territory_count = np.array((self.ncols, self.ncols))
        self.board_indices = list(np.ndindex(self.nrows, self.ncols))
        self.player_territories = [self.board_indices[5::-1], self.board_indices[6:]]

        self.n_players = 2
        self.current_player = 1
        self.other_player = 2

    @property
    def total_seeds(self):
        return np.sum(self.board, axis=None)

    @property
    def board_format(self):
        return self.board_indices[5::-1] + self.board_indices[6:]

    def zero_rows(self):
        rows = np.array([False, False])
        for i in range(self.nrows):
            if np.sum(self.board[i, :]) == 0:
                rows[i] = True
        return rows

    def reset_board(self):
        self.board = 4 * np.ones((self.nrows, self.ncols))

    def action2pit(self, action):
        return self.board_format[action]

    def get_seeds(self, action):
        pit_index = self.action2pit(action=action)
        return self.board[pit_index]

    def set_seeds(self, action, new_value):
        pit_index = self.action2pit(action=action)
        self.board[pit_index] = new_value

    def update_seeds(self, action, new_value):
        pit_index = self.action2pit(action=action)
        self.board[pit_index] += new_value

    def distribute_seeds(self, action):
        # should just change the state of the game but not return anything except maybe some print statements about the change of states, final_index
        # get pit index corresponding to action taken
        pit_index = self.action2pit(action=action)

        # pick seeds from current pit
        seeds = self.board[pit_index]

        # set seeds from current pit to zero
        self.board[pit_index] = 0

        # iterate over number of seeds picked
        while seeds > 0 :
            global next_index
            action += 1
            next_index = self.board_format[action]

            # drop seed at the point found
            self.board[next_index] += 1

            # update number of seeds in hand
            seeds -= 1

            # update current index
            pit_index = next_index

        return next_index

    def capture_seeds(self, action, during_game = True):

        pit_index = self.action2pit(action)
        player_idx = self.current_player - 1
        if self.get_seeds(action) == 4 and np.sum(self.board, axis=None) > 8:
            if during_game:
                if pit_index in self.board_format[player_idx]:
                    self.stores[player_idx] += 4
                    self.set_seeds(action, 0)
                else:
                    self.stores[1 - player_idx] += 4
                    self.board[pit_index] = 0
            else:
                if self.board[pit_index] == 4:
                    self.stores[pit_index] += 4
                    self.set_seeds(action, 0)
        elif self.get_seeds(action) == 4 and np.sum(self.board, axis=None) == 8:
            self.stores[player_idx] += 8
            self.board[self.board > 0] = 0
#######################################################################################

########################################################################################
class Player:
    def __init__(self, board):
        self.B = board
        self.player1 = 1
        self.player2 = 2
        self.stores = self.B.stores
        self.territories = self.B.board_indices
        self.env = GameState()

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

    def Player1_step(self, start_action):
        final_idx = self.B.distribute_seeds(start_action)
        print(self.B.board)
        seeds = self.B.board[final_idx]
        print(seeds)
        action = start_action
        print(action)
        while self.B.board[final_idx] != 1: 
            final_idx = self.B.distribute_seeds(action)
            action = b.board_format.index(final_idx)
            print(self.B.board)
            print(final_idx)
        return self.B.board

    def Player2_step(self, start_action):
        pass
#######################################################################################

########################################################################################
class GameState:
    def __init__(self, board):
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
        self.game_actions = np.array([])
        self.game_states = np.array([])

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

    def save_actions(self):
        pass          

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
        if self.B.territory_count[0] == 12  

    def last_4(self, player):
        if self.B.current_player == player and np.sum(self.B.board < 8):
            player_id = player - 1
            self.B.stores[player_id] +=4
            self.B.board = np.zeros((self.B.nrows, self.B.ncols))

num_of_rounds = 12
#######################################################################################

########################################################################################
class GameController:
    def __init__(self, num_of_rounds):
        self.n_players = 2
        self.board = Board(self.board)
        self.player = Player(self.board)
        self.environment = GameState(self.board)
        self.rules = RuleEngine(self.board, self.environment)

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
        return random.sample(self.environment.possible_moves(player), 1)



    def game(self):

        while self.rules.round < num_of_rounds:
            if self.rules.round ==1:
                self.environment.current_player = self.starting_player("random")
                self.environment.other_player = 1 if self.environment.current_player == 2 else 2
            else: 
                self.environment.current_player = self.starting_player("last_winner")
                self.environment.other_player = 1 if self.environment.current_player == 2 else 2
            
            self.rules.round +=1
            while np.sum(self.board.board, axis = None) > 4:
                
                action = self.choose_action_player(self.environment.current_player)
                

            self.environment.rounds_completed += 1
            self.rules.stop_round()
            



