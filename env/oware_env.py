import gym
import numpy as np

import config

from stable_baselines import logger
class Player():
    def __init__(self, id):
        self.id = id      

class OwareEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, verbose = False, manual = False):
        super(OwareEnv, self).__init__()
        self.name = 'oware'
        self.manual = manual

        self.current_player_num = 1
        self.other_player_num = 2

        self.rows = 2
        self.cols = 6
        self.n_players = 2

        self.board = 4 * np.ones((self.rows, self.cols))
        self.action_space = gym.spaces.Discrete(self.cols)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape = (16,), dtype=int)
        self.round = 1
        self.turns_taken = 0
        self.stores = np.array([0,0])
        self.verbose = verbose
        
    @property
    def territory_count(self):
        territory_1 = len(self.territories[0])
        territory_2 = len(self.territories[1])
        ter_counts = [territory_1, territory_2]
        return ter_counts
        
    @property
    def observation(self):
        flat_board = self.board.flatten()
        max_seeds = 48  
        normalized_board = 2 * (flat_board / max_seeds)  - 1
        max_store_value = 48  
        normalized_stores = 2 * (self.stores / max_store_value) - 1
        max_territories = 12
        normalized_territories = 2 * (np.array(self.territory_count) / max_territories) - 1
        observation_array = np.concatenate((normalized_board, normalized_stores, normalized_territories))
        return observation_array
    
    @property
    def T(self):
        board_indices = list(np.ndindex(self.board.shape))
        initial_territory = board_indices[5::-1] + board_indices[6:]
        return initial_territory
    
    @property
    def territories(self):
        initial_territory = self.T
        ters = [initial_territory[:self.cols], initial_territory[self.cols:]]
        return ters

    @property
    def legal_actions(self):
        legal_actions = []                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
        for action_num in range(self.action_space.n):
            legal = self.is_legal(action_num)
            legal_actions.append(legal)
        return np.array(legal_actions)

    def is_within_range(self, action_num):
        return action_num in self.action_space.n
    
    def is_within_player_range(self, action_num):
        pit_index = self.T[action_num]
        return pit_index in self.territories[self.current_player_num - 1]
    
    def no_seeds_in_territory(self):
        num_seeds = 0
        for i in range(len(self.territories[self.current_player_num-1])):
            idx = self.territories[self.current_player_num-1][i]
            num_seeds += self.board[idx]
        return num_seeds == 0
    
    def contains_seeds(self, action_num):
        pit_index = self.T[action_num]
        return self.board[pit_index] > 0

    def update_round(self):
        if self.round_over():
            self.round +=1

    def is_legal(self, action_num):
        if self.is_within_range(action_num) and self.is_within_player_range(action_num) and self.contains_seeds(action_num):
            return 1
        return 0
    
    def next_position(self, action_num):
        print('next position')
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

    def get_seeds(self, action_num):
        pit_index = self.T[action_num]
        row, col = pit_index
        return self.board[pit_index]

    def set_seeds(self, action_num, new_value):
        pit_index = self.T[action_num]
        self.board[pit_index] = new_value

    def update_seeds(self, action_num, new_value):
        pit_index = self.T[action_num]
        self.board[pit_index] += new_value

    def distribute_seeds(self, action_num, first_dist=False):
        seeds = self.get_seeds(action_num)
        self.set_seeds(action_num, 0)
        while seeds > 0:
            global next_action
            next_action = self.next_position(action_num)
            self.update_seeds(next_action, 1)
            if not first_dist:
                self.capture_seeds(next_action)
            seeds -= 1
            action_num = next_action
        return next_action

    def capture_seeds(self, action_num, during_game = True):
        print('capture seeds')
        pit_index = self.T[action_num]
        player_idx = self.current_player_num - 1
        if self.get_seeds(action_num) == 4 and np.sum(self.board, axis=None) > 8:
            if during_game:
                if pit_index in self.territories[player_idx]:
                    self.stores[player_idx] += 4 
                    self.set_seeds(action_num, 0)
                else:
                    self.stores[1 - player_idx] += 4
                    self.board[pit_index] = 0
            else:
                if self.board[pit_index] == 4:
                    self.stores[pit_index] += 4
                    self.set_seeds(action_num, 0)
        elif self.get_seeds(action_num) == 4 and np.sum(self.board, axis=None) == 8:
            self.stores[player_idx] += 8
            self.board[self.board > 0] = 0

    def end_move(self, action_num):
        print('end move')
        index = self.T[action_num]
        return self.board[index] == 1 or self.board[index] == 0

    def player_move(self, action_num):
        final_pit = self.distribute_seeds(action_num, True)

        while True:
            final_pit = self.distribute_seeds(action_num, True)

            if self.end_move(final_pit):
                self.capture_seeds(final_pit)

                return self.board, action_num, self.stores
            action_num = final_pit

        return self.board, action_num, self.stores
    
    def switch_player(self):
        print('switch player')
        if self.current_player_num == 1 and self.other_player_num == 2:
            self.current_player_num = 2
            self.other_player_num = 1
        elif self.current_player_num == 2 and self.other_player_num == 1:
            self.current_player_num = 1
            self.other_player_num =2
        print('current player', self.current_player_num)

    def round_over(self):
        print('round_over')
        return np.sum(self.board, axis=None) == 4
    
    def turn(self, action_num):
        print('turn')
        while not self.round_over():
            
            print('player_move',self.player_move(action_num))
            board, action, stores = self.player_move(action_num)
            seeds = self.get_seeds(action_num)
            self.switch_player()
            self.board = board
            self.stores = stores
            action = int(input(f'Enter action for player {self.current_player_num}: '))
            action_num = action
        return self.board, action_num, self.stores

    def capture_territory(self):
        if self.round_over(self):
            if self.stores[0] > self.stores[1]:
                print('Player 1 wins round and captures a territory of player 2')
                self.territories[0].append(self.territories[1][-1])
                self.territories[1].pop(self.territories[1][-1])
            elif self.stores[1] > self.stores[0]:
                print('Player 2 wins round and captures a territory of player 1')
                self.territories[1].append(self.territories[0][-1])
                self.territories[0].pop(self.territories[0][-1])
            elif self.stores[0] == self.stores[1]:
                print('Round ended in a tie so game continues')

    def rules_move(self):
        WRONG_MOVE_PROB = 0.01
        player = self.current_player_num
        player_id = player - 1
        for action in range(self.action_space.n):
            if self.is_legal(action):
                self.board, action_num, self.stores = self.player_move(self, action)
                new_stores = self.stores
                if self.round_over():
                    if self.stores[player_id] == max(self.stores):
                        delta_stores = new_stores - self.stores
                        r = max(delta_stores)
                        idx = np.argmax(delta_stores)
                        self.capture_territory()
                        done = True
                        reward = [-r, -r]
                        reward[idx] = r
                        self.update_round()
                        self.reset_round()
                _, done = self.check_game_over(self.stores, self.board, player)
                if done:
                    action_probs = [WRONG_MOVE_PROB] * self.action_space.n
                    action_probs[action] = 1 - WRONG_MOVE_PROB * (self.action_space.n - 1)
                    return action_probs
        player = (self.current_player_num + 1) % 2
        for action in range(self.action_space.n):
            if self.is_legal(action):
                self.board, action_num, self.stores = self.player_move(self, action)
                new_stores = self.stores
                if self.round_over():
                    if self.stores[player_id] == max(self.stores):
                        delta_stores = new_stores - self.stores
                        r = max(delta_stores)
                        idx = np.argmax(delta_stores)
                        self.capture_territory()
                        done = True
                        reward = [-r, -r]
                        reward[idx] = r
                        self.update_round()
                        self.reset_round()
                _, done = self.check_game_over(self.stores, self.board, player)
                if done:
                    action_probs = [0] * self.action_space.n
                    action_probs[action] = 1 - WRONG_MOVE_PROB * (self.action_space.n - 1)
                    return action_probs
        action, masked_action_probs = self.sample_masked_action([1] * self.action_space.n)
        return masked_action_probs
            
    def reset_round(self):
        self.board = 4 * np.ones((2, 6))
        self.stores = np.array([0, 0])       
        
    def reset(self):
        self.board = 4 * np.ones((2, 6))
        self.stores = np.array([0, 0])
        self.current_player_num = 1
        self.turns_taken = 0
        self.done = False
        logger.debug(f'\n\n----- NEW GAME -----')
        return self.observation
        
    def check_game_over(self, board = None , player = None):
        if board is None:
            board = self.board
        if player is None:
            player = self.current_player_num
        player_id = player - 1
        if len(self.territories[player_id]) == 12 or (self.round == 200 and self.stores[player_id] > self.stores[1 - player_id]):
            return 1, True
        elif self.round == 200 and self.stores[player_id] > self.stores[1 - player_id]:
            return 0, True
        return 0, True  
    
    def step(self, action):
        reward = [0,0]
        player_id = self.current_player_num - 1
        # check move legality
        board = self.board
        old_stores = self.stores.copy()

        if not self.is_legal(action): 
            done = True
            reward = [1,1]
            reward[self.current_player_num] = -1
        else:
            self.board, action_num, self.stores = self.player_move(action)
            new_stores = self.stores
            if self.round_over():
                if self.stores[player_id] == max(self.stores):
                    r = new_stores - old_stores
                    self.capture_territory()
                    done = True
                    reward = [-r, -r]
                    reward[player_id] = r
                    self.update_round()
                    self.reset()
            self.turns_taken += 1
            r, done = self.check_game_over()
            reward = [-r,-r]
            reward[self.current_player_num] = r
        self.done = done
        if not done:
            self.current_player_num = (self.current_player_num + 1) % 2
        return self.observation, reward, done, {}
    
    def render(self, mode='human', close=False):
        logger.debug('')
        if close:
            return
        if self.done:
            logger.debug(f'GAME OVER')
        else:
            logger.debug(f"It is Player {self.current_player.id}'s turn to move")
        if self.verbose:
            logger.debug(f'\nObservation: \n{self.observation}')
        if not self.done:
            logger.debug(f'\nLegal actions: {[i for i,o in enumerate(self.legal_actions) if o != 0]}')
            
# class OwareEnv(gym.Env):
#     metadata = {'render.modes': ['human']}

#     def __init__(self, verbose = False, manual = False):
#         super(OwareEnv, self).__init__()
#         self.name = 'oware'
#         self.manual = manual

#         self.rows = 2
#         self.cols = 6
#         self.n_players = 2

#         self.board = 4 * np.ones((self.rows, self.cols))
#         self.board_indices = list(np.ndindex(self.board.shape))
#         self.T = self.board_indices[5::-1] + self.board_indices[6:]
#         self.territories = [self.T[:self.cols], self.T[self.cols:]]
#         self.action_space = gym.spaces.Discrete(self.cols)
#         self.observation_space = gym.spaces.Box(low=-1, high=1, shape = 16, dtype=int)
#         self.round = 1
#         self.stores = np.array([0,0])
#         self.verbose = verbose
        

#     @property
#     def observation(self):
#         flat_board = self.board.flatten()
#         max_seeds = 48  
#         normalized_board = 2 * (flat_board / max_seeds)  - 1
#         max_store_value = 48  
#         normalized_stores = 2 * (self.stores / max_store_value) - 1
#         max_territories = 12
#         normalized_territories = 2 * (self.territories / max_territories) - 1
#         observation_array = np.concatenate((normalized_board, normalized_stores, normalized_territories))
#         return observation_array

#     @property
#     def legal_actions(self):
#         legal_actions = []
#         for action_num in range(self.action_space.n):
#             legal = self.is_legal(action_num)
#             legal_actions.append(legal)
#         return np.array(legal_actions)

#     def is_within_range(self, action_num):
#         return action_num in self.action_space
    
#     def is_within_player_range(self, action_num):
#         pit_index = self.T[action_num]
#         return pit_index in self.territories[self.current_player_num - 1]
    
#     def no_seeds_in_territory(self):
#         num_seeds = 0
#         for i in range(len(self.territories[self.current_player_num-1])):
#             idx = self.territories[self.current_player_num-1][i]
#             num_seeds += self.board[idx]
#         return num_seeds == 0
    
#     def contains_seeds(self, action_num):
#         pit_index = self.T[action_num]
#         return self.board[pit_index] > 0

#     def update_round(self):
#         if self.round_over():
#             self.round +=1

#     def is_legal(self, action_num):
#         if self.is_within_range(self, action_num) and self.is_within_player_range(self, action_num) and self.contains_seeds(self, action_num):
#             return 1
#         return 0
    
#     def next_position(self, action_num):
#         pit_index = self.T[action_num]
#         row, col = pit_index
#         rows, cols = self.board.shape
#         if row == 0:
#             if col == 0:
#                 idx = (row + 1, col)
#                 action_num = self.T.index(idx)
#                 return action_num
#             idx = (row, col - 1)
#             action_num = self.T.index(idx)
#             return action_num
#         else:
#             if col == cols - 1:
#                 idx = (row - 1, col)
#                 action_num = self.T.index(idx)
#                 return action_num
#             else:
#                 idx = (row, col + 1)
#                 action_num = self.T.index(idx)
#                 return action_num

#     def get_seeds(self, action_num):
#         pit_index = self.T[action_num]
#         return self.board[pit_index]

#     def set_seeds(self, action_num, new_value):
#         pit_index = self.T[action_num]
#         self.board[pit_index] = new_value

#     def update_seeds(self, action_num, new_value):
#         pit_index = self.T[action_num]
#         self.board[pit_index] += new_value

#     def distribute_seeds(self, action_num, first_dist = False):
#         seeds = self.get_seeds(self, action_num)
#         self.set_seeds(self, action_num, 0)
#         while seeds > 0:
#             next_action = self.next_position(self, action_num)
#             self.update_seeds(self, action_num, 1)
#             if not first_dist:
#                 self.capture_seeds(self, action_num)
#             seeds -= 1
#             action_num = next_action
#         final_pit = action_num
#         return final_pit

#     def capture_seeds(self, action_num, during_game = True):
#         pit_index = self.T[action_num]
#         player_idx = self.current_player - 1
#         if self.get_seeds(self, action_num) == 4 and np.sum(self.board, axis=None) > 8:
#             if during_game:
#                 if pit_index in self.territories[player_idx]:
#                     self.stores[player_idx] += 4 
#                     self.set_seeds(self, action_num, 0)
#                 else:
#                     self.stores[1 - player_idx] += 4
#                     self.board[pit_index] = 0
#             else:
#                 if self.board[pit_index] == 4:
#                     self.stores[pit_index] += 4
#                     self.set_seeds(self, action_num, 0)
#         elif self.get_seeds(self, action_num) == 4 and np.sum(self.board, axis=None) == 8:
#             self.stores[player_idx] += 8
#             self.board[self.board > 0] = 0

#     def end_move(self, action_num):
#         index = self.T[action_num]
#         return self.board[index] == 1 or self.board[index] == 0

#     def player_move(self, action_num):
#         if self.is_legal(self, action_num):
#             final_pit = self.distribute_seeds(self, action_num, True)
#         else:
#             print('Choose from legal pit')
#             # first traverse around board is complete
#             while True:
#                 if not self.get_seeds(self, final_pit, 0):
#                     final_pit = self.distribute_seeds(self, action_num, True)

#                     if self.end_move(self, final_pit):
#                         self.capture_seeds(self, final_pit)
#                 else:
#                     print('Choose from legal pit')                      
#                 break
#         return self.board, action_num, self.stores 

#     def switch_player(self):
#         if self.current_player_num == 1 and self.other_player_num == 2:
#             self.current_player_num = 2
#             self.other_player_num = 1
#         elif self.current_player_num == 2 and self.other_player_num == 1:
#             self.current_player_num = 1
#             self.other_player_num =2

#     def turn(self, action_num):
#         while not self.round_over(self):
#             board, action, stores = player_move(self, action_num)
#             seeds = self.get_seeds(self, action_num)
#             self.switch_player(self)
#             self.board = board
#             self.stores = stores
#             action_num = action
#         return self.board, action_num, self.stores

#     def round_over(self):
#         return np.sum(self.board) == 4
    
#     def check_game_over(self):
#         if self.territories[0] == 12 or self.territories[1] == 12:
#             return True
#         else:
#             return False
        

#     def check_game_over(self, board = None , player = None):

#         if board is None:
#             board = self.board

#         if player is None:
#             player = self.current_player_num

#         player_id = player - 1

#         if len(self.territories[player_id]) == 12 or (self.round == 200 and self.stores[player_id] > self.stores[1 - player_id]):
#             return 1, True
        
#         elif self.round == 200 and self.stores[player_id] > self.stores[1 - player_id]:
#             return 0, True
        
#         return 0, True  

#     def capture_territory(self):
#         if self.round_over(self):
#             if self.stores[0] > self.stores[1]:
#                 print('Player 1 wins round and captures a territory of player 2')
#                 self.territories[0].append(self.territories[1][-1])
#                 self.territories[1].pop(self.territories[1][-1])
#             elif self.stores[1] > self.stores[0]:
#                 print('Player 2 wins round and captures a territory of player 1')
#                 self.territories[1].append(self.territories[0][-1])
#                 self.territories[0].pop(self.territories[0][-1])
#             elif self.stores[0] == self.stores[1]:
#                 print('Round ended in a tie so game continues')

#     def reset(self):
#         self.board = 4 * np.ones((2, 6))
#         self.stores = [0, 0]

#     @property
#     def current_player(self):
#         return self.players[self.current_player_num]

#     def step(self, action):
        
#         reward = [0,0]
        
#         # check move legality
#         board = self.board
        
#         if not self.is_legal(action): 
#             done = True
#             reward = [1,1]
#             reward[self.current_player_num] = -1
#         else:
#             square = self.get_square(board, action)
#             board[square] = self.current_player.token

#             self.turns_taken += 1
#             r, done = self.check_game_over()
#             reward = [-r,-r]
#             reward[self.current_player_num] = r

#         self.done = done

#         if not done:
#             self.current_player_num = (self.current_player_num + 1) % 2

#         return self.observation, reward, done, {}

    # def rules_move(self):
    #     WRONG_MOVE_PROB = 0.01
    #     player = self.current_player_num

    #     for action in range(self.action_space.n):
    #         if self.is_legal(action):
    #             new_board = self.board.copy()
    #             square = self.get_square(new_board, action)
    #             new_board[square] = self.players[player].token
    #             _, done = self.check_game_over(new_board, player)
    #             if done:
    #                 action_probs = [WRONG_MOVE_PROB] * self.action_space.n
    #                 action_probs[action] = 1 - WRONG_MOVE_PROB * (self.action_space.n - 1)
    #                 return action_probs

    #     player = (self.current_player_num + 1) % 2

    #     for action in range(self.action_space.n):
    #         if self.is_legal(action):
    #             new_board = self.board.copy()
    #             square = self.get_square(new_board, action)
    #             new_board[square] = self.players[player].token
    #             _, done = self.check_game_over(new_board, player)
    #             if done:
    #                 action_probs = [0] * self.action_space.n
    #                 action_probs[action] = 1 - WRONG_MOVE_PROB * (self.action_space.n - 1)
    #                 return action_probs

        
    #     action, masked_action_probs = self.sample_masked_action([1] * self.action_space.n)
    #     return masked_action_probs
