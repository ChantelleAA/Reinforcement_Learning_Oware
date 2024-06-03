# Playing Oware Nam-nam with Reinforcement Learning
![image](https://github.com/ChantelleAA/Reinforcement_Learning_Oware/assets/115734837/1f9f3e49-01cd-49a9-becb-ab79b6ef00dd)

[Image Source](https://owarejam.wordpress.com/about/)

[Rules of the game: article](http://www.oware.org/nam_nam.asp)

[Rules of the game: video](https://www.youtube.com/watch?v=LQE2b5T5Cr4)

## Overview

Oware is a traditional board game that is believed to be about 15,000 years old [1](https://www.yvonneosafopsychotherapist.com/oware#:~:text=Oware%20is%20reputed%20to%20be,varies%20from%20country%20to%20country). Oware is a turn-based strategy game for two players, where the objective is to capture more seeds than the opponent. There are various versions of the game all around the world [2](http://www.oware.org/rules.asp), but the one that would be considered in this project is the Nam-nam version, commonly played in West Africa.

In this implementation, the following are presented:

- Creating a custom Python environment to play Oware nam-nam
- Simulating Oware games, where moves are taken randomly by players
- Training a DQN, DDQN, A3C, and Alpha Zero RL agents to play the game
- Testing scripts between DQN vs DDQN agents, DQN vs Random agents, and DQN vs Human player (interactive)
- Plots showing the training statistics for DQN Agents

## Code Structure

The codebase is organized into several folders. The file structure is as follows:

```
Reinforcement_Learning_Oware/
├── output.txt
├── oware.py
├── README.md
├── requirements.txt
│
├── agent/
│   ├── A3C.py
│   ├── AlphaZero.py
│   ├── DDQNAgent.py
│   ├── DQNAgent.py
│   └── RandomAgent.py
│
├── env/
│   ├── Board.py
│   ├── GameController.py
│   ├── GameState.py
│   ├── Player.py
│   └── RuleEngine.py
│
├── figures/
│   └── Figure_1.png
│
├── logs/
│   ├── ddqn_dqn_random_train
│   │   └── 1.txt
│   └── dqn_vs_random_train
│       └── 1.txt
│
├── output/
│   ├── training_data.csv
│   ├── training_data_ddqn.csv
│   ├── training_data_ddqn1.csv
│   ├── training_data_dqn.csv
│   ├── training_data_dqn1.csv
│   └── rewards
│       ├── rewards_learner.csv
│       └── rewards_random.csv
│
├── saved_weights_ddqn/
│   ├── E100.weights.h5
│   ├── E19.weights.h5
│   ├── E20.weights.h5
│   ├── E30.weights.h5
│   ├── E{}.weights.h5
│   └── model_params20.json
│
├── saved_weights_dqn/
│   ├── model_params100.json
│   └── model_params30.json
│
├── simulations/
│   ├── actions/
│   │   ├── action_state_0.txt
│   │   ├── action_state_1.txt
|   |   |         :
|   |   |         :   
|   |   |         :
│   │   ├── action_state_9.txt
│   │   └── README.md
│   │
│   ├── labels/
│   │   ├── game_state.txt
│   │   └── README.md
│   │
│   ├── simulation_codes/
│   │   ├── generate_all_states_.py
│   │   ├── main.py
│   │   └── simulate_game_env1.py
│   │
│   └── states/
│       ├── game_state_0.txt
│       ├── game_state_1.txt
|       |          :
|       |          :
|       |          :
│       ├── game_state_9.txt
│       └── README.md
│
├── test/
│   ├── test_dqn_vs_dqn.py
│   ├── test_dqn_vs_human.py
│   └── test_dqn_vs_random.py
│
├── train/
│   ├── hyperparameter_tuning.py
│   ├── train_a3c_vs_dqn_ddqn_random.py
│   ├── train_alpha0.py
│   ├── train_ddqn_vs_dqn_random.py
│   ├── train_dqn_vs_dqn.py
│   ├── train_dqn_vs_random.py
│   └── GPU/
│       ├── train_ddqn_gpu.py
│       └── train_dqn_gpu.py
│
└── utils/
    ├── config.py
    ├── utils.py
    └── __pycache__/
        └── replay_buffer.cpython-39.pyc
```

## Custom Oware Environment

### Board Class

The `Board` class is a fundamental component of the Oware nam-Nam environment, responsible for managing the board layout, actions, and score tracking. Here’s a summary of its key features and methods:

#### Key Features

- **Board Initialization**: The board is initialized with 2 rows and 6 columns, with each pit starting with 4 seeds.
- **Stores**: There are 2 stores for each player to keep their captured seeds.
- **Players**: The game is designed for 2 players.
- **Actions**: There are 12 possible actions corresponding to the pits on the board.
- **Turn Tracking**: Keeps track of the number of turns completed and manages the current and other player states.

#### Properties

- **`total_seeds`**: Returns the total number of seeds currently on the board.
- **`board_format`**: Provides a format of the board indices for navigation.

#### Methods

- **`zero_rows`**: Checks if any row on the board is empty and returns a boolean array indicating such rows.
- **`reset_board`**: Resets the board to its initial state with 4 seeds in each pit.
- **`action2pit`**: Converts an action index into a corresponding board index.
- **`get_seeds`**: Returns the number of seeds at the specified action's pit.
- **`set_seeds`**: Sets the number of seeds at the specified action's pit to a new value.
- **`update_seeds`**: Updates the number of seeds at the specified action's pit by adding a new value.

This class interacts directly with the game logic and is central to the functionality of the Oware game environment.

### Player Class

#### Key Features

- **Board Instance**: The `Player` class utilizes an instance of the `Board` class to manage the game board.
- **Game State Management**: The `Player` class interacts with the `GameState` class to manage the state of the game.
- **Seed Stores**: References the seed storage array from the board to keep track of captured seeds.
- **Player Territories**: Manages the indices of the board that represent the territories of each player.

#### Properties

- **`territory_count()`**: Returns the count of territories for each player.

#### Methods

- **`territory_indices(player)`**: Returns the indices of the territories controlled by a specific player.
  - **Parameters**: `player` (int) - The player number (1 or 2).
  - **Returns**: List of indices for the specified player's territories.
- **`is_round_winner(player)`**: Determines if the specified player has won the current round.
  - **Parameters**: `player` (int) - The player number (1 or 2).
  - **Returns**: `True` if the player has more seeds in their store than the opponent; otherwise, `False`.
- **`player_step(start_action, current_player, other_player)`**: Executes a player's turn starting from a given action, distributing seeds until the pit where the last seed lands has 1 or 0 seeds.
  - **Parameters**:
    - `start_action` (int) - The action index from which to start the turn.
    - `current_player` (int) - The number of the current player (1 or 2).
    - `other_player` (int) - The number of the other player (1 or 2).
  - **Returns**: The updated board state after the player's turn.

This class is crucial for managing the interactions of players with the board, facilitating the gameplay by handling moves, territory management, and determining round winners. The `player_step` method ensures the proper execution of a player's move according to the rules of the game.

### RuleEngine Class

#### Key Features

- **Game Board Management**: The `RuleEngine` class manages and applies the rules of the game by interacting with the game board and its state.
- **Action Validation**: Checks the validity of player actions based on the current game state.
- **Round and Game Stopping Conditions**: Determines when a round or the entire game should stop based on specific conditions and rules.

#### Attributes

- **`board`**: An instance of the `Board` class representing the game board and its state.
- **`state`**: An instance of the `GameState` class used to track and manage the current state of the game.
- **`round`**: The current round number in the game.
- **`turn`**: The current turn number within the round.
- **`actions`**: An array of possible action indices, usually representing pits on the board.

#### Methods

- **`__init__(board, state)`**: Initializes the rule engine with references to the game board and state.
  - **Parameters**:
    - `board` (Board): The game board which holds the state and configuration of seeds and pits.
    - `state` (GameState): The object managing the overall state of the game, including scores and turns.

- **`is_valid_action(action, player)`**: Checks if a specific action is valid for a given player based on the current game state.
  - **Parameters**:
    - `action` (int): The action to check, typically an index representing a pit.
    - `player` (int): The player number performing the action.
  - **Returns**: `True` if the action is valid, `False` otherwise.

- **`stop_round()`**: Determines if the current round should be stopped, typically when no seeds are left to play.
  - **Returns**: `True` if the board is empty and the round should stop, `False` otherwise.

- **`stop_game(num_rounds)`**: Checks if the game should end, usually based on a significant victory condition or rule, such as a player achieving a specific territory count.
  - **Parameters**:
    - `num_rounds` (int): The number of rounds to be played.
  - **Returns**: `True` if the game should end, for example, if a player's territory count reaches a set threshold.

This class is crucial for enforcing the rules of the game, validating player actions, and determining the conditions for ending rounds and the game. The `RuleEngine` ensures that the gameplay adheres to the defined rules and progresses correctly.

### GameState Class

#### Key Features

- **Game State Management**: The `GameState` class manages the state of the game, including the current board, stores, territory counts, and the history of actions taken.
- **State Tracking**: Maintains a detailed record of the game's progress and status, allowing for state retrieval at any point.
- **Round and Game Winners**: Determines the winners of rounds and the game based on the current state.

#### Attributes

- **`board`**: An instance of the `Board` class which represents the current state of the game board.
- **`total_games_played`**: The total number of games played during this session.
- **`games_won`**: Array tracking the number of games won by each player.
- **`current_board_state`**: Snapshot of the current state of the game board.
- **`current_store_state`**: Snapshot of the current seeds stored by each player.
- **`current_territory_count`**: Snapshot of the current territories held by each player.
- **`rounds_completed`**: The number of rounds completed in the current game.
- **`turns_completed`**: The number of turns completed in the current round.
- **`win_list`**: A list of players who have won each round.
- **`actions`**: Array of possible actions players can take (typically pit indices).
- **`game_actions`**: A comprehensive list of all actions taken in the game.
- **`player_1_actions`**: A list of actions specifically taken by player 1.
- **`player_2_actions`**: A list of actions specifically taken by player 2.
- **`max_turns`**: The maximum allowed turns per game to prevent infinite loops.
- **`max_rounds`**: The maximum allowed rounds per game.
- **`game_states`**: A matrix to store detailed game state after each turn for analysis or replay.

#### Methods

- **`__init__(board, max_turns=2000, max_rounds=7)`**: Initializes a new game state with a reference to the game board.
  - **Parameters**:
    - `board` (Board): The game board instance.
    - `max_turns` (int): Maximum number of turns to prevent infinite game loops.
    - `max_rounds` (int): Maximum number of rounds in a game session.

- **`update_win_list(player)`**: Adds the winning player of a round to the win list.
  - **Parameters**: `player` (int) - The player number.

- **`zero_row_exists(state)`**: Checks if any row on the board is empty.
  - **Parameters**: `state` (np.ndarray) - The current state of the board.
  - **Returns**: `True` if any row is empty, `False` otherwise.

- **`valid_moves(state)`**: Returns a list of valid moves based on the current board state.
  - **Parameters**: `state` (np.ndarray) - The current state of the board.
  - **Returns**: A list of valid moves (list).

- **`possible_moves(player, state)`**: Returns a list of possible moves for the specified player based on the current board state.
  - **Parameters**:
    - `player` (int) - The player number.
    - `state` (np.ndarray) - The current state of the board.
  - **Returns**: A list of possible moves (list).

- **`save_actions(player, action)`**: Records an action taken by a player for historical tracking.
  - **Parameters**:
    - `player` (int) - The player number.
    - `action` (int) - The action taken.

- **`save_stores()`**: Saves the current state of the stores.
  - **Returns**: The updated list of store states (list).

- **`save_actions_stores(action)`**: Saves the current state of actions and stores.
  - **Parameters**: `action` (int) - The action taken.
  - **Returns**: The combined list of actions and stores (list).

- **`save_game_state()`**: Stores the detailed current state of the game in the game_states matrix for later retrieval or analysis.

- **`save_all_states()`**: Saves all states of the game, including the board state, store state, and territory count.

- **`round_winner()`**: Determines the winner of the current round based on the state of the board.
  - **Returns**: The player number of the round winner (int).

- **`game_winner()`**: Determines the winner of the game based on the territory counts.
  - **Returns**: The player number of the game winner (int).

- **`save_winner()`**: Saves the game winner to the game_winner_list.

- **`switch_player(current_player, other_player)`**: Switches the current and other players.
  - **Parameters**:
    - `current_player` (int) - The current player number.
    - `other_player` (int) - The other player number.
  - **Returns**: The updated current and other players (tuple).

- **`distribute_seeds(action, current_player, other_player)`**: Distributes seeds from a selected pit and captures seeds according to the game rules.
  - **Parameters**:
    - `action` (int) - The action taken.
    - `current_player` (int) - The current player number.
    - `other_player` (int) - The other player number.
  - **Returns**: The index of the next pit (int).

- **`capture_seeds(action, current_player, other_player, during_game=True)`**: Executes the capture process in the game.
  - **Parameters**:
    - `action` (int) - The action taken.
    - `current_player` (int) - The current player number.
    - `other_player` (int) - The other player number.
    - `during_game` (bool) - Whether the capture is happening during the game (default is True).

- **`update_stores_list()`**: Updates the list of store states.

- **`calculate_reward(player)`**: Calculates the reward for a player based on the current state.
  - **Parameters**: `player` (int) - The player number.
  - **Returns**: The calculated reward (int).

### GameController Class

#### Key Features

- **Overall Game Management**: The `GameController` class manages the overall flow of an Oware game, including game rounds, player actions, and the game state.
- **Game Setup**: Initializes the game, decides the starting player, and sets up the necessary components.
- **Turn Management**: Manages the sequence of turns within each round, switching players, and ensuring valid actions are taken.
- **End-of-Game Conditions**: Checks for end-of-game conditions and determines when to stop the game based on predefined rules.

#### Attributes

- **`n_players`**: Number of players in the game, typically two.
- **`board`**: An instance of the `Board` class representing the game board.
- **`player`**: An instance of the `Player` class to manage player actions.
- **`environment`**: An instance of the `GameState` class to track and store the game's state.
- **`rules`**: An instance of the `RuleEngine` class to enforce game rules.
- **`max_turns`**: The maximum number of turns allowed to prevent infinite loops.
- **`action_space_size`**: The number of possible actions (12).
- **`state_space_size`**: The size of the game state space (12).
- **`cumulative_score`**: Tracks the cumulative score throughout the game.
- **`board_indices`**: Stores the format of the board indices.

#### Methods

- **`__init__()`**: Initializes the game controller with default settings.
  - **Parameters**: `num_of_rounds` (int) - The number of rounds to be played (default is 6).

- **`starting_player(how="random")`**: Determines which player starts a round.
  - **Parameters**: `how` (str) - Method to determine the starting player ("random" or "last_winner").
  - **Returns**: The player number who will start the round.

- **`game(num_of_rounds=num_of_rounds)`**: Executes the main game loop, handling the progression of rounds and managing game state updates.
  - **Parameters**: `num_of_rounds` (int) - The number of rounds to be played.

- **`step(action, player)`**: Executes a single game step given an action taken by the current player.
  - **Parameters**: 
    - `action` (int) - The action taken by the current player.
    - `player` (int) - The current player number.
  - **Returns**: 
    - `next_state` (np.array) - The state of the game after the action.
    - `reward` (float) - The reward received after taking the action.
    - `done` (bool) - Whether the game or round has ended.
    - `info` (dict) - Additional information about the step for debugging.

- **`update_territories()`**: Updates the territories based on the current state of the game.
- **`reset_game()`**: Resets the game to a clean state at the start of a new game.
  - **Returns**: The initial state of the game board (np.array).

- **`reset_round()`**: Resets the game to a clean state at the start of a new round.
  - **Returns**: The initial state of the game board (np.array).

- **`get_score()`**: Returns the cumulative score.
  - **Returns**: The cumulative score (int).

- **`sample_legal_move(state)`**: Samples a legal move from the given state.
  - **Parameters**: `state` (np.array) - The current state of the game.
  - **Returns**: A random legal move (int).

- **`get_player_turn()`**: Gets the current player's turn.
  - **Returns**: The current player number (int).

This class orchestrates the game by managing rounds, processing player actions, updating the state, and determining when the game ends based on the rules defined in the `RuleEngine`. The `game()` method is the core function that runs the game loop, ensuring that the game progresses correctly and adheres to the rules.

## Gameplay Logic

The game flow is managed by the `GameController` class, which initializes the components, selects the starting player, and cycles through the game rounds until a stop condition is met. During each round, players take turns making moves based on the current state of the board, with the game state being updated after each move. Rounds and games conclude based on the board state and seed counts, and the board is reset at the end of each game round.

## How to Use
To utilize this environment for reinforcement learning:
1. Instantiate the `GameController` with the desired number of rounds.
2. Use the `game()` method to start the game simulation.
3. Interact with the game using the player methods to make moves and capture seeds.

The detailed class structures and methods ensure that the game can be easily modified and extended for different reinforcement learning scenarios, providing a robust framework for experimenting with various strategies in the Oware game.
