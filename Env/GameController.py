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

    def game(self, num_of_rounds):
        from env import DQNAgent
        print(f"START GAME ...")

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
        agent = DQNAgent(state_size=12, action_size=12)

        while self.rules.round < num_of_rounds and self.board.turns_completed < self.max_turns:

            print(f" START ROUND {self.rules.round}")
            print("\n")
            
            # Decision on who starts the round
            if self.rules.round == 1:
                current_player = self.starting_player("random")
            else:
                current_player = self.starting_player("last_winner")

            other_player = 1 if current_player == 2 else 2


            print(f"Player {current_player} starts this round")

            # increment the number of rounds after the choice of who starts the round
            self.rules.round +=1
            
            # Implement a round
            while np.sum(self.board.board, axis = None) > 0 or not self.rules.stop_round():

                
                
                # record the state
                state = np.array(self.environment.current_board_state)
                

                valid_actions = self.environment.possible_moves(current_player)
                if not valid_actions:
                    raise ValueError("No valid actions available for the current state.")
                    # print("No valid actions available for the current state.")
                    # break

                print(self.board.board)
                self.environment.save_all_states()
                self.environment.save_game_state()  

                if np.random.rand() <= agent.epsilon:
                    action_c = random.choice(valid_actions)
                else:
                    act_values = agent.model.predict(state)
                    action_c = np.argmax(act_values[0])


                # get the action of the current player based on the current state
                # action_c = self.choose_action_player(current_player)

                print(f"Current player: {current_player}")
                print(f"Player {current_player} action options {self.environment.possible_moves(current_player)}\n")
                print(f"Player {current_player} chooses action: {action_c}\n")

                # do a step for players
                self.player.player_step(action_c, current_player, other_player)

                # get reward for the player based on current states 
                reward = self.environment.calculate_reward(current_player)

                # next_state
                next_state = np.array(self.environment.current_board_state)

                # is the round over
                done = self.rules.stop_round() or np.sum(self.environment.current_board_state, axis=None) == 0

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

                # switch players
                print(f"Switch Players ...")
                self.environment.switch_player(current_player, other_player)
                

                # if self.rules.stop_round() or np.sum(self.environment.current_board_state) == 0:      
                if done:       
                    break
                
                try: 
                    valid_actions = self.environment.possible_moves(other_player)
                    # if not valid_actions:
                        # raise ValueError("No valid actions available for the current state.")
                except Exception as e:
                    print(f"{e}")
                    break

                
                if np.random.rand() <= agent.epsilon:
                    action_o = random.choice(valid_actions)
                else:
                    act_values = agent.model.predict(state)
                    action_o = np.argmax(act_values[0])
                # action_o = self.choose_action_player(other_player)

                print(f"Current player: {other_player}")
                print(f"Player {other_player} action options {self.environment.possible_moves(other_player)}\n")

                print(f"Player {other_player} chooses action: {action_o}\n")
                self.player.player_step(action_o,  other_player, current_player)
                self.environment.save_actions(other_player, action_o)
                print("GAME STATE SAVING ...")
                print(f"{self.environment.current_board_state}")
                self.environment.save_game_state()
                self.environment.save_all_states()
                print("GAME STATE SAVED ...")
                
                if self.rules.stop_round() or np.sum(self.environment.current_board_state) == 0:             
                    break
                # if self.board.turns_completed == 2000:
                #     print("2000 turns in round reached")
                #     break

            # if self.board.turns_completed == 2000:
            #     break

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
            winner = self.environment.game_winner()
            self.environment.save_winner() 

            print(f"ROUND ENDS\n")
            print(f"Rounds completed: {self.environment.rounds_completed }")

            print('RESET BOARD AND STORES FOR NEW ROUND ...')
            self.board.reset_board()
            self.board.stores = np.array([0, 0])

            if self.rules.stop_game() == True:
                print(f"STOP GAME")
                break
            
            
            if winner != 0:
                print(f"Player {winner} wins!, {self.environment.current_territory_count}")
            else: 
                print(f" Game ends in draw ")

    # def step(self, action):
    #     """
    #     Executes a single game step given an action taken by the current player.
        
    #     Parameters:
    #         action (int): The action taken by the current player.

    #     Returns:
    #         next_state (np.array): The state of the game after the action.
    #         reward (float): The reward received after taking the action.
    #         done (bool): Whether the game or round has ended.
    #         info (dict): Additional information about the step for debugging.
    #     """
    #     # Decide on starting player
    #     print(f"{self.environment.turns_completed}")
    #     if self.environment.turns_completed == 0:

    #         print(f"First ever step with {self.environment.turns_completed} turn completed\n")
    #         print(f"Deciding the player who starts ...")

    #         if self.rules.round == 1:  
    #             self.environment.current_player = self.starting_player("random")
    #             self.environment.other_player = 1 if self.environment.current_player == 2 else 2

    #             print(f"Round 1 so starting player is chosen randomly: {self.environment.current_player}")

    #         else:
    #             self.environment.current_player = self.starting_player("last_winner")
    #             self.environment.other_player = 1 if self.environment.current_player == 2 else 2

    #             print(f"Round {self.rules.round} so starting player is {self.environment.current_player}")
        
        

    #     # Save the current state
    #     current_state = np.copy(self.environment.current_board_state)
    #     print(f"Current board state: \n{current_state}, total seeds {np.sum(current_state)}")
    #     print(f"Current stores state: \n{self.board.stores}")

    #     # set players
    #     current_player = self.environment.current_player
    #     print(f"{current_player=}")
    #     other_player = 2 if current_player == 1 else 1
    #     print(f"{other_player=}")

    #     # Execute action

    #     print(f"Player {current_player} chooses action: {action}\n")

    #     self.environment.current_board_state = self.player.player_step(action, current_player, other_player)
    #     # print(f"{self.environment.current_board_state=}")
    #     # update turns completed
    #     self.environment.turns_completed+=1

    #     # Calculate the reward based on action
    #     reward = self.environment.calculate_reward(current_player)
    #     # print(f"{reward = }")

    #     next_state = np.copy(self.environment.current_board_state)
    #     print(f"Next state:\n {next_state}, total seeds {np.sum(next_state)}")
    #     print(f"Current stores state: \n{self.board.stores}")
    #     # Update the game state and check if the round/game has ended
    #     # done = self.rules.stop_round() or np.sum(self.environment.current_board_state, axis=None) == 0

    #     round_done = np.sum(next_state, axis=None) == 0
    #     print(f"{round_done = }")
    #     done = self.rules.stop_game()
        
    #     print(f"{done=}")

    #     # Optionally, gather additional info for debugging or detailed logging
    #     info = {
    #         'current_player': current_player,
    #         'action_taken': action,
    #         'reward': reward,
    #     }
    #     # print(f"{info=}")

    #     # Switch players if the game continues
    #     if not done and not round_done:
    #         print(f"round not over")
    #         current_player, other_player = self.environment.switch_player(current_player, other_player)
    #         print(f"Switch Player , current player is now {self.environment.current_player}")
    #     elif not done and round_done:
    #         print(f"Round over")
    #         if self.board.stores[0] > self.board.stores[1]:
    #             self.environment.games_won[0] +=1
    #             self.environment.win_list += [1]
    #             self.board.territory_count[0] +=1
    #             self.board.territory_count[1] -=1
    #             # write function to update territories

    #         elif self.board.stores[0] < self.board.stores[1]:
    #             self.environment.games_won[1] +=1
    #             self.environment.win_list += [2]
    #             self.board.territory_count[1] +=1
    #             self.board.territory_count[0] -=1

    #         self.environment.rounds_completed += 1
    #         winner = self.environment.game_winner()
    #         self.environment.save_winner() 

    #         # print(f"ROUND ENDS\n")
    #         # print(f"Rounds completed: {self.environment.rounds_completed }")

    #         # print('RESET BOARD AND STORES FOR NEW ROUND ...')
    #         # self.board.reset_board()
    #         # self.board.stores = np.array([0, 0])

    #     #     current_player = self.starting_player("last_winner")
    #     #     other_player = 1 if current_player == 2 else 2

    #     # self.cumulative_score += reward
    #     return next_state, reward, done, info



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
        # Decide on starting player
        # print(f"{self.environment.turns_completed}")
        # if self.environment.turns_completed == 0:

        #     print(f"First ever step with {self.environment.turns_completed} turn completed\n")
        #     print(f"Deciding the player who starts ...")

        #     if self.rules.round == 1:  
        #         self.environment.current_player = self.starting_player("random")
        #         self.environment.other_player = 1 if self.environment.current_player == 2 else 2

        #         print(f"Round 1 so starting player is chosen randomly: {self.environment.current_player}")

        #     else:
        #         self.environment.current_player = self.starting_player("last_winner")
        #         self.environment.other_player = 1 if self.environment.current_player == 2 else 2

        #         print(f"Round {self.rules.round} so starting player is {self.environment.current_player}")
        
        self.environment.current_player = player
        self.environment.other_player = 2 if self.environment.current_player == 1 else 1

        # Save the current state
        current_state = np.copy(self.environment.current_board_state)
        print(f"Current board state: \n{current_state}, total seeds {np.sum(current_state)}")
        print(f"Current stores state: \n{self.board.stores}")

        # set players
        current_player = self.environment.current_player
        print(f"{current_player=}")
        other_player = 2 if current_player == 1 else 1
        print(f"{other_player=}")

        # Execute action

        print(f"Player {current_player} chooses action: {action}\n")

        self.environment.current_board_state = self.player.player_step(action, current_player, other_player)
        # print(f"{self.environment.current_board_state=}")
        # update turns completed
        self.environment.turns_completed+=1

        # Calculate the reward based on action
        reward = self.environment.calculate_reward(current_player)
        # print(f"{reward = }")

        next_state = np.copy(self.environment.current_board_state)
        print(f"Next state:\n {next_state}, total seeds {np.sum(next_state)}")
        print(f"Current stores state: \n{self.board.stores}")
        # Update the game state and check if the round/game has ended
        # done = self.rules.stop_round() or np.sum(self.environment.current_board_state, axis=None) == 0

        round_done = np.sum(next_state, axis=None) == 0
        print(f"{round_done = }")
        done = self.rules.stop_game()
        
        print(f"{done=}")

        # Optionally, gather additional info for debugging or detailed logging
        info = {
            'current_player': current_player,
            'action_taken': action,
            'reward': reward,
        }
        # print(f"{info=}")

        # Switch players if the game continues
        if not done and not round_done:
            print(f"round not over")
            current_player, other_player = self.environment.switch_player(current_player, other_player)
            print(f"Switch Player , current player is now {self.environment.current_player}")
        elif not done and round_done:
            print(f"Round over")
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
            winner = self.environment.game_winner()
            self.environment.save_winner() 

            print(f"ROUND ENDS\n")
            print(f"Rounds completed: {self.environment.rounds_completed }")

            print('RESET BOARD AND STORES FOR NEW ROUND ...')
            self.board.reset_board()
            self.board.stores = np.array([0, 0])

            current_player = self.starting_player("last_winner")
            other_player = 1 if current_player == 2 else 2

        return next_state, reward, done, info


    def reset_game(self):
        """
        Resets the game to a clean state at the start of a new game.
        
        Returns:
            np.array: The initial state of the game board.
        """
        # Reset the board to its initial configuration
        self.board.reset_board()  # Assuming there's a method in Board to reset it
        self.environment.rounds_completed = 0
        self.environment.total_games_played += 1
        self.environment.current_player = self.starting_player("random")
        self.environment.other_player = 1 if self.environment.current_player == 2 else 2
        self.environment.game_winner_list = []
        
        # Return the initial state of the board
        return np.array(self.board.board)

    def reset_round(self):
        """
        Resets the game to a clean state at the start of a new round.
        
        Returns:
            np.array: The initial state of the game board.
        """
        # Reset the board to its initial configuration
        self.board.reset_board()  # Assuming there's a method in Board to reset it
        self.board.stores = np.array([0, 0])
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
