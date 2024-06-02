import tkinter as tk
from tkinter import messagebox
import numpy as np

# Assuming the classes Board, GameState, Player, and RuleEngine are already defined
from env import Board, GameState, Player, RuleEngine  # Import your classes


class OwareGUI(tk.Tk):
    def __init__(self, board, state, player1, player2, rule_engine):
        super().__init__()
        self.board = board
        self.state = state
        self.player1 = player1
        self.player2 = player2
        self.rule_engine = rule_engine
        self.current_player = 1
        self.title("Oware Game")
        self.geometry("800x600")
        self.create_widgets()
        self.update_possible_actions()

    def create_widgets(self):
        self.main_frame = tk.Frame(self)
        self.main_frame.pack(expand=True)

        self.board_frame = tk.Frame(self.main_frame)
        self.board_frame.pack(pady=20)

        self.pits = []
        for r in range(2):
            row = []
            for c in range(6):
                btn = tk.Button(self.board_frame, text=f"{self.board.board[r, c]}", width=10, height=4,
                                command=lambda r=r, c=c: self.make_move(r, c))
                btn.grid(row=r, column=c)
                row.append(btn)
            self.pits.append(row)

        self.info_frame = tk.Frame(self.main_frame)
        self.info_frame.pack(pady=20)

        self.score_label = tk.Label(self.info_frame, text=f"Player 1: {self.board.stores[0]}  Player 2: {self.board.stores[1]}")
        self.score_label.grid(row=0, column=0, columnspan=2, pady=5)

        self.territory_label = tk.Label(self.info_frame, text=f"Player 1 Territory: {self.board.player_territories[0]}  Player 2 Territory: {self.board.player_territories[1]}")
        self.territory_label.grid(row=1, column=0, columnspan=2, pady=5)

        self.rounds_label = tk.Label(self.info_frame, text=f"Rounds Played: {self.state.rounds_completed}  Max Rounds: {self.state.max_rounds}")
        self.rounds_label.grid(row=2, column=0, pady=5)

        self.wins_label = tk.Label(self.info_frame, text=f"Player 1 Wins: {self.state.games_won[0]}  Player 2 Wins: {self.state.games_won[1]}")
        self.wins_label.grid(row=2, column=1, pady=5)

        self.possible_actions_label = tk.Label(self.info_frame, text="")
        self.possible_actions_label.grid(row=3, column=0, columnspan=2, pady=5)

    def update_board(self):
        for r in range(2):
            for c in range(6):
                self.pits[r][c].config(text=f"{self.board.board[r, c]}")
        self.score_label.config(text=f"Player 1: {self.board.stores[0]}  Player 2: {self.board.stores[1]}")
        self.territory_label.config(text=f"Player 1 Territory: {self.board.player_territories[0]}  Player 2 Territory: {self.board.player_territories[1]}")
        self.rounds_label.config(text=f"Rounds Played: {self.state.rounds_completed}  Max Rounds: {self.state.max_rounds}")
        self.wins_label.config(text=f"Player 1 Wins: {self.state.games_won[0]}  Player 2 Wins: {self.state.games_won[1]}")
        self.update_possible_actions()

    def make_move(self, r, c):
        action = r * 6 + c
        if self.rule_engine.is_valid_action(action, self.current_player):
            if self.current_player == 1:
                self.player1.player_step(action, self.current_player, 3 - self.current_player)
            else:
                self.player2.player_step(action, self.current_player, 3 - self.current_player)
            self.update_board()
            if self.rule_engine.stop_round():
                winner = self.state.round_winner()
                messagebox.showinfo("Round Over", f"Round over! Winner: Player {winner}")
                self.reset_board()
            elif self.rule_engine.stop_game(self.state.rounds_completed):
                winner = self.state.game_winner()
                messagebox.showinfo("Game Over", f"Game over! Winner: Player {winner}")
                self.quit()
            else:
                self.switch_player()
        else:
            messagebox.showwarning("Invalid Move", "This move is not valid. Choose another pit.")

    def update_possible_actions(self):
        possible_moves = self.state.possible_moves(self.current_player, self.board.board)
        self.possible_actions_label.config(text=f"Possible Actions: {possible_moves}")

    def switch_player(self):
        self.current_player = 3 - self.current_player
        self.update_possible_actions()

    def reset_board(self):
        self.board.reset_board()
        self.update_board()

if __name__ == "__main__":
    board = Board()
    state = GameState(board)
    player1 = Player(board, state)
    player2 = Player(board, state)
    rule_engine = RuleEngine(board, state)

    app = OwareGUI(board, state, player1, player2, rule_engine)
    app.mainloop()