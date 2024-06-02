# Playing Oware Nam-nam with Reinforcement Learning
![image](https://github.com/ChantelleAA/Reinforcement_Learning_Oware/assets/115734837/1f9f3e49-01cd-49a9-becb-ab79b6ef00dd)
[Source](https://owarejam.wordpress.com/about/)
[Rules of the game: article](http://www.oware.org/nam_nam.asp)
[Rules of the game: video](https://www.youtube.com/watch?v=LQE2b5T5Cr4)
## Overview

This GitHub repository hosts an implementation of the traditional African board game Oware (also known as Mancala) developed to support reinforcement learning experiments. The implementation is structured using several Python classes that model the game's mechanics, and it is built atop popular libraries such as NumPy and gym.

Oware is a turn-based strategy game for two players, where the objective is to capture more seeds than the opponent. This implementation is particularly designed to facilitate the development and testing of reinforcement learning agents capable of learning optimal strategies through gameplay against other automated agents or predefined strategies.

## Code Structure

The codebase is organized into several classes, each handling specific aspects of the game environment:

### 1. Board Class

**Purpose:** Manages the physical state of the Oware board.

#### Attributes:
- `nrows`: Number of rows on the board (fixed at 2).
- `ncols`: Number of columns per row (fixed at 6).
- `total_stores`: Storage locations for captured seeds (fixed at 2, one for each player).
- `board`: A 2D NumPy array representing the current state of the game board.
- `total_territories`: Total number of pits on the board.
- `stores`: Array holding the count of seeds captured by each player.
- `territory_count`: Initial distribution of seeds across the board (12 seeds per player).
- `board_indices`: Indices representing the layout and traversal order of the board pits.
- `player_territories`: Specific territories controlled by each player.
- `n_players`: Total number of players (fixed at 2).
- `current_player`: The player whose turn is current.
- `other_player`: The opposing player.
- `turns_completed`: Counter for the number of turns that have been completed.

#### Methods:
- `total_seeds()`: Returns the sum of seeds on the board.
- `board_format()`: Returns the board configuration.
- `zero_rows()`: Checks for rows with zero seeds, indicating potential game-end conditions.
- `reset_board()`: Resets the board to the initial state with 4 seeds in each pit.
- `action2pit(action)`: Translates a game action (pit selection) into a board index.
- `get_seeds(action)`: Retrieves the number of seeds at a specific pit.
- `set_seeds(action, new_value)`, `update_seeds(action, new_value)`: Set or update the number of seeds in a pit.
- `distribute_seeds(action)`: Implements the seed sowing action of the game.
- `capture_seeds(action, during_game)`: Handles the capture of seeds according to game rules.

### 2. Player Class

**Purpose:** Manages player actions and interactions with the board.

#### Attributes:
- `B`: Instance of the Board class.
- `player1`, `player2`: Identifiers for the two players.
- `stores`: Reference to the seed storage array from the board.
- `territories`: Reference to the board indices from the board.

#### Methods:
- `territory_count()`: Returns the count of territories.
- `territory_indices(player)`: Returns the territories controlled by a specific player.
- `is_round_winner(player)`: Determines if the specified player has won the current round.
- `player_step(start_action)`: Executes a player's move starting from the specified action.

### 3. GameState Class

**Purpose:** Tracks the state of the game across rounds and games.

#### Attributes:
- `B`: Instance of the Board class.
- `total_games_played`, `games_won`: Trackers for games played and won by each player.
- `current_board_state`, `current_store_state`, `current_territory_count`: Current state snapshots.
- `rounds_completed`: Number of rounds completed in the current game.
- `win_list`, `game_actions`, `player_1_actions`, `player_2_actions`, `game_states`: Logs of various game metrics and actions.

#### Methods:
- `update_win_list(player)`: Updates the winner list.
- `possible_moves(player)`: Computes possible moves for a given player.
- `save_actions(player, action)`: Logs actions taken by the players.

### 4. RuleEngine Class

**Purpose:** Enforces the rules of the game.

#### Attributes:
- `B`: Instance of the Board class.
- `state`: Instance of the GameState class.
- `round`, `turn`: Current round and turn counters.

#### Methods:
- `is_valid_action(action, player)`: Checks if an action is valid for the given player.
- `stop_round()`, `stop_game()`, `check_round_end()`: Check and enforce end-of-round and end-of-game conditions.

### 5. GameController Class

**Purpose:** Orchestrates the gameplay, linking all components.

#### Attributes:
- `n_players`: Number of players.


- `board`: Board class instance.
- `player`: Player class instance.
- `environment`: GameState class instance.
- `rules`: RuleEngine class instance.

#### Methods:
- `starting_player(how)`: Determines who starts the game or round.
- `choose_action_player(player)`: Randomly picks a valid move for the player.
- `game()`: Implements the game logic over multiple rounds, handling setup and teardown of game states.

## Gameplay Logic

The game flow is managed by the `GameController` class, which initializes the components, selects the starting player, and cycles through the game rounds until a stop condition is met. During each round, players take turns making moves based on the current state of the board, with the game state being updated after each move. Rounds and games conclude based on the board state and seed counts, and the board is reset at the end of each game round.

## How to Use
To utilize this environment for reinforcement learning:
1. Instantiate the `GameController` with the desired number of rounds.
2. Use the `game()` method to start the game simulation.
3. Interact with the game using the player methods to make moves and capture seeds.

The detailed class structures and methods ensure that the game can be easily modified and extended for different reinforcement learning scenarios, providing a robust framework for experimenting with various strategies in the Oware game.
