import numpy as np
from Board import Board
from Player import Player
from GameState import GameState
from RuleEngine import RuleEngine

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



    def game(self, num_of_rounds):

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
                current_player = self.starting_player("random")
                other_player = 1 if current_player == 2 else 2
                print(current_player)
            else:
                current_player = self.starting_player("last_winner")
                other_player = 1 if current_player == 2 else 2
            
            # increment the number of rounds after the choice of who starts the round
            self.rules.round +=1
            
            # Implement a round
            while np.sum(self.board.board, axis = None) > 0:
                print("Total seeds on board", np.sum(self.board.board, axis = None) )
                action_c = self.choose_action_player(current_player)
                self.player.player_step(action_c, current_player, other_player)
                self.environment.save_actions(current_player, action_c)

                self.environment.save_game_state()
                print(f"End of turn for player {current_player}")
                # print(f"GameController Board = {self.board.board}")
                self.environment.switch_player(current_player, other_player)

                if self.rules.stop_round() == True:             
                    break

                action_o = self.choose_action_player(other_player)
                self.player.player_step(action_o,  other_player, current_player)
                self.environment.save_actions(other_player, action_o)
                self.environment.save_game_state()
                print(f"End of turn for player {current_player}")
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
