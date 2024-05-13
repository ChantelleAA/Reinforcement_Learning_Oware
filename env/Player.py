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

    def player_step(self, start_action, current_player, other_player):
        """
        Executes a player's turn starting from a given action, repeatedly distributing seeds
        until the pit where the last seed lands has 1 or 0 seeds.
        """
        action = start_action
        final_idx = self.state.distribute_seeds(action, current_player, other_player)
        max_iterations = 30  # Set a reasonable limit to prevent infinite loops

        for _ in range(max_iterations):
            if self.B.board[final_idx] <= 1:
                break  # Exit the loop if the last pit has 1 or 0 seeds
            action = self.B.board_format.index(final_idx)  # Update action based on the last index
            final_idx = self.state.distribute_seeds(action, current_player, other_player)

        return self.B.board

