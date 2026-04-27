import sys, os
_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_DIR))  # project root -> finds uttt_game

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from uttt_game import UTTTGame


class CNNEnv(gym.Env):

    def __init__(self):
        super().__init__()
        self.game = UTTTGame()

        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(6, 9, 9),
            dtype=np.float32
        )

        self.action_space = spaces.Discrete(81)

    def action_masks(self):
        mask = np.zeros(81, dtype=bool)
        for br, bc, lr, lc in self.game.get_legal_moves():
            mask[br * 27 + bc * 9 + lr * 3 + lc] = True
        return mask

    def _get_obs(self):
        p = self.game.current_player
        grid = self.game.board.transpose(0, 2, 1, 3).reshape(9, 9)

        ch0 = (grid == p).astype(np.float32)         # my pieces
        ch1 = (grid == -p).astype(np.float32)        # opponent's pieces

        ch2 = np.zeros((9, 9), dtype=np.float32)     # legal moves (cell-level)
        for br, bc, lr, lc in self.game.get_legal_moves():
            ch2[br * 3 + lr, bc * 3 + lc] = 1.0

        sw = self.game.sub_board_winners
        ones3 = np.ones((3, 3), dtype=np.float32)
        ch3 = np.kron((sw == p).astype(np.float32),  ones3)   # my won sub-boards
        ch4 = np.kron((sw == -p).astype(np.float32), ones3)   # opponent's won sub-boards
        ch5 = np.kron((sw == 2).astype(np.float32),  ones3)   # drawn sub-boards

        return np.stack([ch0, ch1, ch2, ch3, ch4, ch5])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset()
        return self._get_obs(), {}

    def step(self, action):
        br = action // 27
        bc = (action % 27) // 9
        lr = (action % 9) // 3
        lc = action % 3

        _, reward, terminated, truncated, info = self.game.step((br, bc, lr, lc))

        if not terminated:
            opponent_moves = self.game.get_legal_moves()
            opponent_action = opponent_moves[np.random.randint(len(opponent_moves))]
            _, reward, terminated, truncated, info = self.game.step(opponent_action)

            if terminated:
                if self.game.winner == 0:
                    reward = 0.5
                else:
                    reward = -1.0

        return self._get_obs(), reward, terminated, truncated, info
