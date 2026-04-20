import numpy as np

class UTTTGame:
    """
    Ultimate Tic-Tac-Toe game logic.
    
    Board representation:
        board: (3, 3, 3, 3) numpy array
            board[br, bc, lr, lc] = 0 (empty), 1 (X), -1 (O)
            br, bc = big board row, col
            lr, lc = local board row, col
        
        sub_board_winners: (3, 3) numpy array
            0 = unclaimed, 1 = X won, -1 = O won, 2 = draw
        
        current_player: 1 (X) or -1 (O)
        
        active_board: (br, bc) tuple indicating which sub-board
            must be played in, or None if free choice.
    """

    def __init__(self):
        self.board = np.zeros((3, 3, 3, 3), dtype=int)
        self.sub_board_winners = np.zeros((3, 3), dtype=int)
        self.current_player = 1       # X goes first
        self.active_board = None      # None = free choice (first move)
        self.done = False
        self.winner = 0               # 0 = no winner yet


    def get_legal_moves(self):
        """
        Returns a list of legal moves as (br, bc, lr, lc) tuples.
        """
        moves = []

        if self.active_board is not None:
            br, bc = self.active_board
            # Must play in the active sub-board
            for lr in range(3):
                for lc in range(3):
                    if self.board[br, bc, lr, lc] == 0:
                        moves.append((br, bc, lr, lc))
        else:
            # Free choice — can play in any open cell of any
            # unclaimed sub-board
            for br in range(3):
                for bc in range(3):
                    if self.sub_board_winners[br, bc] == 0:
                        for lr in range(3):
                            for lc in range(3):
                                if self.board[br, bc, lr, lc] == 0:
                                    moves.append((br, bc, lr, lc))

        return moves    
    

    def step(self, action):
        """
        Apply a move to the board.
        action: (br, bc, lr, lc) tuple

        Returns: (observation, reward, terminated, truncated, info)
        """
        assert not self.done, "Game is already over. Call reset()."
        assert action in self.get_legal_moves(), "Illegal move."

        br, bc, lr, lc = action

        # Place the current player's mark
        self.board[br, bc, lr, lc] = self.current_player

        # Check if this move won the local sub-board
        if self._check_local_winner(br, bc):
            self.sub_board_winners[br, bc] = self.current_player
        elif self._is_local_full(br, bc):
            self.sub_board_winners[br, bc] = 2  # draw

        # Set the next active board
        next_br, next_bc = lr, lc
        if self.sub_board_winners[next_br, next_bc] != 0:
            self.active_board = None  # sent to completed board, free choice
        else:
            self.active_board = (next_br, next_bc)

        # Check global win or draw
        if self._check_global_winner():
            self.winner = self.current_player
            self.done = True
        elif self._is_global_done():
            self.winner = 0  # draw
            self.done = True

        reward = self._get_reward()

        # Switch players
        self.current_player *= -1

        return self._get_observation(), reward, self.done, False, {}
    
    def _check_local_winner(self, br, bc):
        """Check if current_player has won the sub-board at (br, bc)."""
        b = self.board[br, bc]  # 3x3 slice
        p = self.current_player

        # Check rows and columns
        for i in range(3):
            if all(b[i, j] == p for j in range(3)):
                return True
            if all(b[j, i] == p for j in range(3)):
                return True

        # Check diagonals
        if all(b[i, i] == p for i in range(3)):
            return True
        if all(b[i, 2 - i] == p for i in range(3)):
            return True

        return False

    def _is_local_full(self, br, bc):
        """Check if sub-board at (br, bc) has no empty cells."""
        return not np.any(self.board[br, bc] == 0)

    def _check_global_winner(self):
        """Check if current_player has won the macro board."""
        w = self.sub_board_winners
        p = self.current_player

        # Check rows and columns
        for i in range(3):
            if all(w[i, j] == p for j in range(3)):
                return True
            if all(w[j, i] == p for j in range(3)):
                return True

        # Check diagonals
        if all(w[i, i] == p for i in range(3)):
            return True
        if all(w[i, 2 - i] == p for i in range(3)):
            return True

        return False

    def _is_global_done(self):
        """Check if no sub-boards remain playable."""
        return not np.any(self.sub_board_winners == 0)

    def _get_reward(self):
        """
        Returns reward from the perspective of current_player.
        """
        if self.done:
            if self.winner == self.current_player:
                return 1.0
            elif self.winner == 0:
                return 0.5  # draw
            else:
                return -1.0  # loss
        return 0.0  # game still in progress

    def _get_observation(self):
        """
        Returns a dictionary observation containing all state
        the agent needs.
        """
        return {
            "board": self.board.copy(),
            "sub_board_winners": self.sub_board_winners.copy(),
            "current_player": self.current_player,
            "active_board": self.active_board
        }
    
    def reset(self):
        """
        Reset the game to the initial state.
        Returns the initial observation.
        """
        self.board = np.zeros((3, 3, 3, 3), dtype=int)
        self.sub_board_winners = np.zeros((3, 3), dtype=int)
        self.current_player = 1
        self.active_board = None
        self.done = False
        self.winner = 0

        return self._get_observation()

    def render(self):
        """
        Print a text representation of the board to the console.
        """
        symbols = {0: ".", 1: "X", -1: "O"}
        sub_winner_symbols = {0: " ", 1: "X", -1: "O", 2: "-"}

        print("\n")
        for br in range(3):
            for lr in range(3):
                for bc in range(3):
                    print(" ", end="")
                    for lc in range(3):
                        print(symbols[self.board[br, bc, lr, lc]], end="")
                    if bc < 2:
                        print(" |", end="")
                for bc in range(3):
                    pass
                print()
            if br < 2:
                print("  ---+-----+---")

        print("\nSub-board winners:")
        for br in range(3):
            for bc in range(3):
                print(sub_winner_symbols[self.sub_board_winners[br, bc]], end=" ")
            print()

        print(f"\nCurrent player: {'X' if self.current_player == 1 else 'O'}")
        if self.active_board:
            print(f"Active board: {self.active_board}")
        else:
            print("Active board: Free choice")
        if self.done:
            if self.winner == 1:
                print("Game over: X wins!")
            elif self.winner == -1:
                print("Game over: O wins!")
            else:
                print("Game over: Draw!")
        print()