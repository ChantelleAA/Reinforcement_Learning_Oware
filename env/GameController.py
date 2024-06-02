import numpy as np
import random

from .Board import Board
from .Player import Player
from .GameState import GameState
from .RuleEngine import RuleEngine


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
    global num_of_rounds
    num_of_rounds = 6

    def __init__(self):
        self.n_players = 2
        self.board = Board()
        self.environment = GameState(self.board)
        self.player = Player(self.board, self.environment)
        self.rules = RuleEngine(self.board, self.environment)
        self.max_turns = 2000
        self.action_space_size = 12
        self.state_space_size = 12
        self.cumulative_score = 0
        self.board_indicies = self.board.board_format

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

    def game(self, num_of_rounds= num_of_rounds):
        from ..agent import DQNAgent
        print(f"START GAME ...")

        agent = DQNAgent(state_size=12, action_size=12, player_id=0)

        while self.rules.round < num_of_rounds and self.board.turns_completed < self.max_turns:

            print(f" START ROUND {self.rules.round}\n")
            print(f"New Board: \n{self.environment.current_board_state}")

            # Decision on who starts the round
            if self.rules.round == 1:
                current_player = self.starting_player("random")
            else:
                current_player = self.starting_player("last_winner")

            other_player = 1 if current_player == 2 else 2
            print(f"Player {current_player} starts this round")

            # Implement a round
            while np.sum(self.board.board, axis = None) > 0 or not self.rules.stop_round():

                # record the state
                state = np.array(self.environment.current_board_state)

                valid_actions = self.environment.possible_moves(current_player, state)
                
                if not valid_actions:
                    done = True
                    print("No valid actions available for the current state.\n ", state)
                    remaining_seeds = np.sum(state, axis=None)
                    self.board.stores[other_player - 1] += remaining_seeds
                    state[state > 0] = 0
                    break

                self.environment.save_all_states()
                self.environment.save_game_state()  

                if np.random.rand() <= agent.epsilon:
                    action_c = random.choice(valid_actions)
                else:
                    act_values = agent.model.predict(state)
                    action_c = np.argmax(act_values[0])

                print(f"Current player: {current_player}")
                print(f"Player {current_player} action options {self.environment.possible_moves(current_player, state)}\n")
                print(f"Player {current_player} chooses action: {action_c}\n")

                # do a step for players
                next_state = self.player.player_step(action_c, current_player, other_player)
                self.environment.current_board_state = next_state
                # get reward for the player based on current states 
                reward = self.environment.calculate_reward(current_player)

                # is the round over
                done = self.rules.stop_round()

                # Store the transition in memory
                agent.remember(state.reshape(1, -1), action_c, reward, next_state.reshape(1, -1), done)  

                # save the actions taken by player 
                self.environment.save_actions(current_player, action_c)

                # save states of the game.
                print("GAME STATE SAVING ...")
                print(f"{self.environment.current_board_state}")
                self.environment.save_all_states()
                self.environment.save_game_state()
                print("GAME STATE SAVED ...")
                print(f"Total states saved ({len(self.environment.game_state)})\n")

                print(f"Current stores state:\n {self.board.stores}")
                print(f"Current territory count:\n {self.environment.current_territory_count}")
                # switch players
                print(f"Switch Players ...")
                self.environment.switch_player(current_player, other_player)
        
                if done:       
                    break 

                valid_actions = self.environment.possible_moves(other_player, state)

                if not valid_actions:
                    done = True
                    print("No valid actions available for the current state.\n ", state)

                    remaining_seeds = np.sum(state, axis=None)
                    self.board.stores[other_player - 1] += remaining_seeds
                    state[state > 0] = 0
                    break
                    # raise ValueError("No valid actions available for the current state.")

                if np.random.rand() <= agent.epsilon:
                    action_o = random.choice(valid_actions)
                else:
                    act_values = agent.model.predict(state)
                    action_o = np.argmax(act_values[0])

                print(f"Current player: {other_player}")
                print(f"Player {other_player} action options {self.environment.possible_moves(other_player, next_state)}\n")

                print(f"Player {other_player} chooses action: {action_o}\n")
                next_state = self.player.player_step(action_o,  other_player, current_player)
                self.environment.current_board_state = next_state

                self.environment.save_actions(other_player, action_o)
                print("GAME STATE SAVING ...")

                print(f"{next_state}")
                self.environment.save_game_state()
                self.environment.save_all_states()
                print("GAME STATE SAVED ...")
                
                if self.rules.stop_round() :
                    break

            self.update_territories()

            self.environment.rounds_completed += 1
            self.rules.round +=1
            winner = self.environment.game_winner()
            self.environment.save_winner() 

            print(f"ROUND ENDS\n")
            print(f"Stores state : {self.board.stores}")
            print(f"Rounds completed: {self.environment.rounds_completed }")

            if winner != 0:
                print(f"Player {winner} wins game!, {self.environment.current_territory_count}")
            else: 
                print(f" Game ends in draw ")

            if self.rules.stop_game(num_of_rounds) == True:
                print(f"STOP GAME")
                break

            print('RESET BOARD AND STORES FOR NEW ROUND ...')
            self.environment.current_board_state = self.board.reset_board()
            print(f"After previous round and board is reset:\n {self.environment.current_board_state}")
            self.board.stores = np.array([0, 0])    


    def step(self, action, player):
        """
        Executes a single game step given an action taken by the current player.
        
        Parameters:
            action (int): The action taken by the current player.

        Returns:
            next_state (np.array): The state of the game after the action.
            reward (float): The reward received after taking the action.
            done (bool): Whether the game or round has ended.
            info (dict): Additional information about the step for debugging.
        """
        
        self.environment.current_player = player
        self.environment.other_player = 2 if self.environment.current_player == 1 else 1

        # Save the current state
        current_state = np.copy(self.environment.current_board_state)
        print(f"Current board state: \n{current_state}, total seeds {np.sum(current_state)}")
        print(f"Current stores state: \n{self.board.stores}")

        # set players
        current_player = self.environment.current_player
        # print(f"{current_player=}")
        other_player = 2 if current_player == 1 else 1
        # print(f"{other_player=}")

        print(f"Player {current_player} chooses action: {action}\n")

        if action == None:
            print(f"Player {current_player} has no valid actions")
            print(f"End the round, and the one whose territory has the seeds takes all")
            round_done = True
            remaining_seeds = np.sum(current_state, axis=None)
            self.board.stores[other_player - 1] += remaining_seeds
            current_state[current_state > 0] = 0
            reward = self.environment.calculate_reward(current_player)
            done = self.rules.stop_game(num_of_rounds)

            if not done:
                self.update_territories()

                self.environment.rounds_completed += 1
                self.rules.round+=1

                print(f"ROUND ENDS\n")
                print(f"Rounds completed: {self.environment.rounds_completed }")

                print('RESET BOARD AND STORES FOR NEW ROUND ...')
                next_state = self.reset_round()

                current_player = self.starting_player("last_winner")
                other_player = 1 if current_player == 2 else 2

        else:
            next_state = self.player.player_step(action, current_player, other_player)
            self.environment.current_board_state = next_state
            self.environment.update_stores_list()
            self.environment.turns_completed+=1

            # Calculate the reward based on action
            reward = self.environment.calculate_reward(current_player)

            # Update the game state and check if the round/game has ended
            round_done = np.sum(next_state, axis=None) == 0
        
        done = self.rules.stop_game(num_of_rounds)

        # print(f"{round_done = }")
       
        # print(f"{self.rules.round=}")
        print(f"Territory count: {self.board.territory_count}")
        # print(f"{done=}")
        print(f"Territory indices: {self.board.player_territories}")

        info = {
            'current_player': current_player,
            'action_taken': action,
            'reward': reward,
            'winner': None
        }

        if not done and not round_done:
            print(f"round not over")
            current_player, other_player = self.environment.switch_player(current_player, other_player)
            print(f"Switch Player, current player is now {self.environment.current_player}")

        elif not done and round_done:
            print(f"Round over")
            self.update_territories()

            self.environment.rounds_completed += 1
            self.rules.round+=1
            winner = self.environment.game_winner()
            self.environment.save_winner() 

            print(f"ROUND ENDS\n")
            print(f"Rounds completed: {self.environment.rounds_completed }")
            print(f"Territory indices: {self.board.player_territories}")
            print(f"Territory count: {self.board.territory_count}")

            print('RESET BOARD AND STORES FOR NEW ROUND ...')
            next_state = self.reset_round()

            current_player = self.starting_player("last_winner")
            other_player = 1 if current_player == 2 else 2

        elif done and round_done:
            winner = self.environment.game_winner()
            info["winner"] = winner
            self.reset_game()
            print("Game is done")
            print(f"Territory count: {self.board.territory_count}")
            if winner == 0:
                print("Game ends in a draw")
            else:        
                print(f"Winner is player : {winner}")

        else:
            winner = self.environment.game_winner()
            info["winner"] = winner
        return next_state, reward, done, info

    def update_territories(self):
        print("Territories updating")
        if self.board.stores[0] > self.board.stores[1]:
            self.environment.games_won[0] +=1
            self.environment.win_list += [1]
            self.board.territory_count[0] +=1
            self.board.territory_count[1] -=1
            self.board.player_territories[0].append((self.board.player_territories[1][-1]))
            self.board.player_territories[1].pop(-1)

        elif self.board.stores[0] < self.board.stores[1]:
            self.environment.games_won[1] +=1
            self.environment.win_list += [2]
            self.board.territory_count[1] +=1
            self.board.territory_count[0] -=1
            self.board.player_territories[1].append((self.board.player_territories[0][-1]))
            self.board.player_territories[0].pop(-1)
        print(self.board.player_territories, self.board.territory_count)


    def reset_game(self):
        """
        Resets the game to a clean state at the start of a new game.
        
        Returns:
            np.array: The initial state of the game board.
        """
        # Reset the board to its initial configuration
        self.board.reset_board()
        self.environment.rounds_completed = 0
        self.rules.round = 0
        self.environment.total_games_played += 1
        self.environment.current_player = self.starting_player("random")
        self.environment.other_player = 1 if self.environment.current_player == 2 else 2
        self.environment.game_winner_list = []
        self.board.player_territories = [self.board.board_indices[5::-1], self.board.board_indices[6:]]
        self.board.territory_count = np.array([6, 6])
        self.environment.stores_list = []

        # Return the initial state of the board
        return np.array(self.board.board)

    def reset_round(self):
        """
        Resets the game to a clean state at the start of a new round.
        
        Returns:
            np.array: The initial state of the game board.
        """
        # Reset the board to its initial configuration
        self.board.board = self.board.reset_board() 
        self.environment.current_board_state = self.board.board
        self.board.stores = np.array([0, 0])
        self.environment.stores_list = []
        # Return the initial state of the board
        return np.array(self.board.board)

    def get_score(self):
        # Simply return the cumulative score
        return self.cumulative_score
    
    def sample_legal_move(self, state):
        legal_moves = []
        for i in range(self.action_space_size):
            if i in self.environment.valid_moves(state):
                legal_moves.append(i)
        return np.random.choice(legal_moves, 1)

    def get_player_turn(self):
        return self.environment.current_player
    
