from .GameState import GameState

class Player:
    def __init__(self, board, state):
        self.B = board
        self.state = GameState(self.B)
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

    def player_step(self, start_action,  current_player, other_player):
        final_idx = self.state.distribute_seeds(start_action,  current_player, other_player,)
        seeds = self.B.board[final_idx]
        action = start_action
        while self.B.board[final_idx] > 1:
            final_idx = self.state.distribute_seeds(action,  current_player, other_player,)
            action = self.B.board_format.index(final_idx)
        return self.B.board
