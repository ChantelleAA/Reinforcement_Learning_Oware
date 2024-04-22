import numpy as np
import gym
import random

class Board():
    """
    This Board Class is a central class in the Oware nam-Nam environment. It describes the Oware board, and some important features associated with the board which are relaant for the game. It's attributes are used and manipulated by all other subsequent classes. It mainly manaes the board layout, the basic actions carried out in the playing of the game and the tracking of the scores.
    """

    def __init__(self):
        self.nrows = 2  # Number of rows on the board
        self.ncols = 6  # Number of columns on the board
        self.total_stores = 2  # Number of storage ptis here player's captured seeds are kept
        self.board = 4 * np.ones((self.nrows, self.ncols))  # The actual board for the game, defined as an array where the numbers in each position represent the number of seeds in the pit.

        self.total_territories = self.nrows * self.ncols # The number of territories is the same as the number of total pits in the board
        self.stores = np.zeros((self.total_stores,))  # The  array which keeps track of the seeds captured by each player
        self.territory_count = np.array((self.ncols, self.ncols)) # The number of territories possessed by each player at the start of the game
        self.board_indices = list(np.ndindex(self.nrows, self.ncols)) # The list of indicies in which order a player moves (anti-clockwise)
        self.player_territories = [self.board_indices[5::-1], self.board_indices[6:]] # Territory indicies for each player

        self.n_players = 2 # Define the number of players
        self.current_player = 1 # Define the curreent player
        self.other_player = 2 # Define the other player

        self.turns_completed = 0 # Define the number of turns completed

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

    def distribute_seeds(self, action:int):
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
            action = (action + 1)%12
            print(f'{action = }')
            next_index = self.board_format[action]

            # drop seed at the point found
            self.board[next_index] += 1
            print(f"{self.turns_completed = }")
            if self.turns_completed > 1:
                self.capture_seeds(action)

            # update number of seeds in hand
            seeds -= 1

            # update current index
            pit_index = next_index

        self.turns_completed +=1
        
        return next_index

    def capture_seeds(self, action:int, during_game = True):
        """ This function is crafted to execute the capture process in the game"""
        pit_index = self.action2pit(action)  # for the given action value here we want to check the viability of a seed capture
        player_idx = self.current_player - 1  # Get the player's id
        if self.get_seeds(action) == 4 and np.sum(self.board, axis=None) > 8: # Condition checking if there is 4 in the pit where the player just dropped his seed
            if during_game: # if this capture is happening during the course of a game,
                print(f"Seed capture during game")
                if pit_index in self.board_format[player_idx]: # if the pit is the player's territory
                    print(f" Home capture for player {self.current_player}")
                    self.stores[player_idx] += 4 # add the seeds to the store of the player
                    self.board[pit_index] = 0  # remove seeds from the board
                    print(f"Board after capture: \n {self.board} \n Player stores after capture: \n {self.stores}")
                else: # otherwise if it is in the territory of the opponent
                    print(f"Opponent, player {self.other_player} has captured during player {self.current_player}'s turn")
                    self.stores[1 - player_idx] += 4 # the opponent gets the seeds
                    self.board[pit_index] = 0 # remove seeds from the board 
                    print(f"Board after capture: \n {self.board} \n Player stores after capture: \n {self.stores}")
            else: # if its not during the course but at the end
                print(f"Seed capture at the end of a turn")
                if self.board[pit_index] == 4: 
                    self.stores[pit_index] += 4
                    self[pit_index] = 0
                    print(f"Board after capture: \n {self.board} \n Player stores after capture: \n {self.stores}")
        elif self.get_seeds(action) == 4 and np.sum(self.board, axis=None) == 8:
            print(f"Now the board has just 8 seeds left and player {self.current_player} has captured the first 4, so he gets the rest.")
            self.stores[player_idx] += 8
            print(f"Board before: \n {self.board}")
            print(f"Stores status update: {self.stores}")
            self.board[self.board > 0] = 0
            print(f"Board after that final capture. Also, round ended.")
            print(f"Let's start a new round, taking note of our winner for this round!!!")
#######################################################################################

########################################################################################
class Player():
    def __init__(self, board):
        self.B = board
        self.player1 = 1
        self.player2 = 2
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
        print(f'{start_action=}')
        print(f'{type(start_action) = }')
        final_idx = self.B.distribute_seeds(start_action)
        # print(self.B.board)
        seeds = self.B.board[final_idx]
        # print(seeds)
        action = start_action
        # print(action)
        while self.B.board[final_idx] != 0:
            final_idx = self.B.distribute_seeds(action)
            action = self.B.board_format.index(final_idx)
            print(self.B.board)
            print(final_idx)
        print(f"Player step ends here")
        return self.B.board

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
        # self.turns_completed = 0
        self.win_list = []
        self.round_winner = 0
        self.current_player = 0
        self.other_player = 0
        self.actions = np.arange(12)
        self.game_actions = []
        self.player_1_actions = []
        self.player_2_actions = []
        self.game_states = []

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
        # else:   
        #     pass

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
    def __init__(self, num_of_rounds):
        self.n_players = 2
        self.board = Board()
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
                - 

        """
        while self.rules.round < num_of_rounds:
            print(f" Start round {self.rules.round}")
            # Decision on who starts the round
            if self.rules.round == 1:
                self.environment.current_player = self.starting_player("random")
                self.environment.other_player = 1 if self.environment.current_player == 2 else 2
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
                print(f"End of turn for player {self.environment.current_player}")
                print(f"GameController Board = {self.board.board}")

                if self.rules.check_round_end:
                    break

                action_o = self.choose_action_player(self.environment.other_player)
                self.player.player_step(action_o)
                self.environment.save_actions(self.environment.other_player, action_o)
                print(f"End of turn for player {self.environment.other_player}")
                print(f"GameController Board = {self.board.board}")
            
            print(f"End round")
            print(f" Final GameController Board = {self.board.board}")

            if self.board.stores[0] > self.board.stores[1]:
                self.environment.games_won[0] +=1
                self.environment.win_list += [1]
            elif self.board.stores[0] < self.board.stores[1]:
                self.environment.games_won[1] +=1
                self.environment.win_list += [2]

            self.environment.rounds_completed += 1
            print(self.environment.rounds_completed)

            self.board.reset_board()
            print(f"New GameController Board = {self.board.board}")
            self.board.stores = np.array([0, 0])

            # self.rules.stop_round()
