import random
import numpy as np

def zero_row_exists(state, board):
    for i in range(board.nrows):
        if np.sum(state[i,:], axis = None)==0 :
            return True
    return False

def valid_moves(state, board):
    actions = board.actions
    valid=[]
    for i in actions:
        pit_index = board.action2pit(i)
        if state[pit_index] != 0:
            if zero_row_exists(state, board) and state[pit_index]> 6-i%6:
                valid.append(i)
            elif not zero_row_exists(state, board):
                valid.append(i)
    return valid

def possible_states(player, state, board):
    actions = board.actions
    player_id = player-1
    open_moves = []
    state = np.reshape(state, (2, -1))
    for i in actions:
        pit_index = board.action2pit(i)
        if pit_index in board.player_territories[player_id]:
            if i in valid_moves(state, board) :
                open_moves.append(i)
    return open_moves
