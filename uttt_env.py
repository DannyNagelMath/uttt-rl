import numpy as np
import gymnasium as gym
from gymnasium import spaces
from uttt_game import UTTTGame


class UTTTEnv(gym.Env):

    def __init__(self):
        super().__init__()
        self.game = UTTTGame()

        self.observation_space = spaces.Box(
            low=-1,
            high=1,
            shape=(109,),
            dtype=np.int8
        )

        self.action_space = spaces.Discrete(81)

    def _get_obs(self):
        board_flat = self.game.board.flatten().astype(np.int8)

        sub_board_winner = np.where(
            self.game.sub_board_winners == 2, 0, self.game.sub_board_winners
        ).flatten().astype(np.int8)

        sub_board_drawn = (self.game.sub_board_winners == 2).flatten().astype(np.int8)

        current_player = np.array([self.game.current_player], dtype=np.int8)

        mask = np.zeros((3, 3), dtype=np.int8)

        if self.game.active_board is not None:
            br, bc = self.game.active_board
            mask[br, bc] = 1
        else:
            mask[self.game.sub_board_winners == 0] = 1

        return np.concatenate([
            board_flat,
            sub_board_winner,
            sub_board_drawn,
            current_player,
            mask.flatten()
        ])
    
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

        # Used for random opponent training
        if not terminated:
            opponent_moves = self.game.get_legal_moves()
            opponent_action = opponent_moves[np.random.randint(len(opponent_moves))]
            _, reward, terminated, truncated, info = self.game.step(opponent_action)

            # Game ended on opponents turn
            if terminated:      
                if self.game.winner == 0:
                    reward = 0.5  # draw
                else:
                    reward = -1.0  # opponent won
        # End random opponent training

        return self._get_obs(), reward, terminated, truncated, info