import random
import numpy as np
# from env import PLAYER_NUM_RANDOM, INVALID_ACTION_PENALTY
INVALID_ACTION_PENALTY = -10 
PLAYER_NUM_RANDOM = 2
class RandomAgent:
    def __init__(self, action_size, board, player_id = PLAYER_NUM_RANDOM):
        self.action_size = action_size
        self.board = board
        self.player_id = player_id

    def act(self, state, board):
        valid_actions = self.possible_states(self.player_id, state, board)
        if not valid_actions:
            return None, INVALID_ACTION_PENALTY  # Return None for action and a penalty for invalid state
        return random.choice(valid_actions), 0

    def zero_row_exists(self, state, board):
        for i in range(board.nrows):
            if np.sum(state[i,:], axis = None)==0 :
                return True
        return False

    def valid_moves(self, state, board):
        actions = board.actions
        valid=[]
        for i in actions:
            pit_index = board.action2pit(i)
            if state[pit_index] != 0:
                if self.zero_row_exists(state, board) and state[pit_index]> 6-i%6:
                    valid.append(i)
                elif not self.zero_row_exists(state, board):
                    valid.append(i)
        return valid

    def possible_states(self, player, state, board):
        actions = board.actions
        player_id = player-1
        open_moves = []
        state = np.reshape(state, (2, -1))
        for i in actions:
            pit_index = board.action2pit(i)
            if pit_index in board.player_territories[player_id]:
                if i in self.valid_moves(state, board) :
                    open_moves.append(i)
        return open_moves
