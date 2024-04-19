import numpy as np
import gym
from gym import spaces
import logging

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class OwareEnv(gym.Env):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, verbose=False, manual=False):
        super(OwareEnv, self).__init__()
        self.name = 'Oware Nam-Nam'
        self.manual = manual
        self.verbose = verbose

        self.rows = 2
        self.cols = 6
        self.n_players = 2

        self.board = 4 * np.ones((self.rows, self.cols), dtype=int)
        self.board_indices = list(np.ndindex(self.board.shape))
        self.T = self.board_indices[5::-1] + self.board_indices[6:]
        self.action_space = spaces.Discrete(self.cols*self.rows)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(16,), dtype=np.float32)

        self.round = 1
        self.stores = np.array([0, 0], dtype=int)
        self.territories = [self.cols, self.cols]
        self.current_player_num = 1
        self.other_player_num = 2
        self.done = False
        self.previous_store = np.array([0, 0], dtype=int)
        self.previous_territories = np.array([self.cols, self.cols], dtype=int)

    def step(self, action):
        self.previous_store = np.copy(self.stores)
        self.previous_territories = np.copy(self.territories)
        
        if not self.is_legal(action):
            self.done = True
            reward = -10  # Penalty for illegal move
            logging.info("Illegal move attempted.")
            return self.observation, reward, self.done, {}

        self.player_move(action)
        reward = self.evaluate_reward()
        if self.check_round_over():
            self.capture_territory()

        self.done = self.check_game_over()
        if not self.done:
            self.switch_player()
        logging.info(f'Action taken: {action}, Board state: {self.board.tolist()}, Stores: {self.stores.tolist()}, Territories: {self.territories}, Current Player: {self.current_player_num}')
        return self.observation, reward, self.done, {}


    def reset(self):
        self.board = 4 * np.ones((self.rows, self.cols), dtype=int)
        self.stores = np.array([0, 0], dtype=int)
        self.done = False
        self.round += 1
        logging.info("Game reset.")
        return self.observation

    @property
    def observation(self):
        flat_board = self.board.flatten()
        max_seeds = 48
        normalized_board = 2 * (flat_board / max_seeds) - 1
        max_store_value = 48
        normalized_stores = 2 * (self.stores / max_store_value) - 1
        max_territories = self.cols
        normalized_territories = 2 * (np.array(self.territories) / max_territories) - 1
        observation_array = np.concatenate([normalized_board, normalized_stores, normalized_territories])
        return observation_array

    def player_move(self, action):
        self.distribute_seeds(action)

    def next_position(self, action_num):
        pit_index = self.T[action_num]
        row, col = pit_index
        rows, cols = self.board.shape
        if row == 0:
            if col == 0:
                idx = (row + 1, col)
                action_num = self.T.index(idx)
                return action_num
            idx = (row, col - 1)
            action_num = self.T.index(idx)
            return action_num
        else:
            if col == cols - 1:
                idx = (row - 1, col)
                action_num = self.T.index(idx)
                return action_num
            else:
                idx = (row, col + 1)
                action_num = self.T.index(idx)
                return action_num

    def distribute_seeds(self, action):
        row, col = self.T[action]
        seeds = self.board[row, col]
        self.board[row, col] = 0
        pos = action
        row, col = self.T[pos]
        while seeds > 0:
            pos = self.next_position(pos)  # Use the next_position method
            row, col = self.T[pos]
            self.board[row, col] += 1
            seeds -= 1
        self.capture_seeds(pos)

    def get_seeds(self, action):
        pit_index = self.T[action]
        return self.board[pit_index]

    def set_seeds(self, action, new_value):
        pit_index = self.T[action]
        self.board[pit_index] = new_value

    def capture_seeds(self, action, during_game = True):
        pit_index = self.T[action]
        player_idx = self.current_player_num - 1
        if self.get_seeds(action) == 4 and np.sum(self.board, axis=None) > 8:
            if during_game:
                if pit_index in self.territories[player_idx]:
                    self.stores[player_idx] += 4 
                    self.set_seeds(self, action, 0)
                else:
                    self.stores[1 - player_idx] += 4
                    self.board[pit_index] = 0
            else:
                if self.board[pit_index] == 4:
                    self.stores[pit_index] += 4
                    self.set_seeds(self, action, 0)
        elif self.get_seeds(action) == 4 and np.sum(self.board, axis=None) == 8:
            self.stores[player_idx] += 8
            self.board[self.board > 0] = 0

    def is_legal(self, action):
        row, col = self.T[action]
        row = self.current_player_num - 1
        return 0 <= action < self.cols and self.board[row, col] > 0

    def switch_player(self):
        self.current_player_num, self.other_player_num = self.other_player_num, self.current_player_num

    def check_game_over(self):
        if any(t == 0 for t in self.territories):
            return True
        return False

    def check_round_over(self):
        if np.sum(self.board) <= 6:  # Check if a round should end
            return True
        return False

    def capture_territory(self):
        player_idx = self.current_player_num - 1
        other_player_idx = self.other_player_num - 1
        if self.stores[player_idx] > self.stores[other_player_idx]:
            self.territories[player_idx] += 1
            self.territories[other_player_idx] -= 1

    def evaluate_reward(self):
        reward = 0
        player_idx = self.current_player_num - 1
        other_player_idx = self.other_player_num - 1
        
        # Reward for moving in the right direction (Assuming it means making a legal move)
        reward += 5
        
        # Reward for capturing seeds
        seeds_captured_this_turn = self.stores[player_idx] - self.previous_store[player_idx]
        if seeds_captured_this_turn > 0:
            reward += 4 * seeds_captured_this_turn
        
        # Check if player has the largest store value at the end of the round
        if self.check_round_over() and self.stores[player_idx] > self.stores[other_player_idx]:
            reward += 10

        # Reward for capturing an opponent's territory
        if self.territories[player_idx] > self.previous_territories[player_idx]:
            reward += 50 * (self.territories[player_idx] - self.previous_territories[player_idx])

        # Reward for winning the game
        if self.check_game_over() and self.stores[player_idx] > self.stores[other_player_idx]:
            reward += 100
        return reward

    def render(self, mode='human'):
        if mode == 'human':
            print("Current board state:")
            print(self.board)
            print(f"Stores: {self.stores}")
            print(f"Territories: {self.territories}")
            print(f"Current player: Player {self.current_player_num}")
