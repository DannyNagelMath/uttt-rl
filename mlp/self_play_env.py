import sys, os
_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_DIR))  # project root -> finds uttt_game

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from uttt_game import UTTTGame
from uttt_env import UTTTEnv
from utils import flip_obs


class SelfPlayEnv(gym.Env):
    """
    Gymnasium wrapper for self-play training.

    Each episode, the current agent is randomly assigned X (1) or O (-1).
    The frozen opponent model plays the other side.

    When the current agent is O:
        - Observations are flipped before being returned (agent thinks it's X)
        - The frozen opponent receives normal (unflipped) observations as X

    When the current agent is X:
        - Observations are returned as-is
        - The frozen opponent receives flipped observations (it thinks it's X)

    Parameters
    ----------
    opponent : MaskablePPO
        Frozen opponent model. Sampled from the checkpoint pool by the
        training script and injected here.
    """

    def __init__(self, opponent):
        super().__init__()
        self.opponent = opponent
        self.game = UTTTGame()

        # Reuse UTTTEnv solely for _get_obs() and action_masks()
        self._env = UTTTEnv()
        self._env.game = self.game

        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(109,), dtype=np.int8
        )
        self.action_space = spaces.Discrete(81)

        # Set during reset()
        self.agent_side = 1   # 1 = X, -1 = O

    # ── Public Gymnasium interface ────────────────────────────────────────────

    def action_masks(self):
        return self._env.action_masks()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset()

        # Randomly assign sides for this episode
        self.agent_side = 1 if np.random.random() < 0.5 else -1

        # If agent is O, opponent (X) moves first
        if self.agent_side == -1:
            self._opponent_move()

        return self._agent_obs(), {}

    SUB_REWARD   = 0.3    # reward per sub-board won / lost
    STEP_PENALTY = -0.005 # per-step cost to encourage faster wins

    def step(self, action):
        sub_before = self.game.sub_board_winners.copy()

        # Apply the agent's chosen action
        br = action // 27
        bc = (action % 27) // 9
        lr = (action % 9)  // 3
        lc = action % 3
        _, reward, terminated, truncated, info = self.game.step((br, bc, lr, lc))

        # UTTTGame._get_reward() is called before current_player switches,
        # so reward always reflects the outcome for whoever just moved.
        # No sign flip needed regardless of which side the agent is playing.

        if not terminated:
            # Reward for sub-boards the agent just claimed
            agent_mark = self.agent_side
            newly_won = np.sum(
                (self.game.sub_board_winners == agent_mark) & (sub_before != agent_mark)
            )
            reward += self.SUB_REWARD * newly_won
            reward += self.STEP_PENALTY

            sub_before_opp = self.game.sub_board_winners.copy()
            self._opponent_move()

            if not self.game.done:
                # Penalise for sub-boards the opponent just claimed
                opp_mark = -self.agent_side
                newly_won_by_opp = np.sum(
                    (self.game.sub_board_winners == opp_mark) & (sub_before_opp != opp_mark)
                )
                reward -= self.SUB_REWARD * newly_won_by_opp

            # Check if opponent ended the game
            if self.game.done:
                terminated = True
                if self.game.winner == 0:
                    reward = 0.5
                elif self.game.winner == self.agent_side:
                    reward = 1.0
                else:
                    reward = -1.0

        return self._agent_obs(), reward, terminated, truncated, info

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _agent_obs(self):
        """Return observation from the agent's perspective."""
        obs = self._env._get_obs()
        if self.agent_side == -1:
            obs = flip_obs(obs)
        return obs

    def _opponent_obs(self):
        """Return observation from the opponent's perspective."""
        obs = self._env._get_obs()
        if self.agent_side == 1:
            # Opponent is O, so flip for it
            obs = flip_obs(obs)
        return obs

    def _opponent_masks(self):
        """Action masks are side-independent (same legal moves either way)."""
        return self._env.action_masks()

    def _opponent_move(self):
        """Have the frozen opponent pick and apply a move."""
        obs   = self._opponent_obs()
        masks = self._opponent_masks()
        action, _ = self.opponent.predict(
            obs, action_masks=masks, deterministic=False
        )
        br = action // 27
        bc = (action % 27) // 9
        lr = (action % 9)  // 3
        lc = action % 3
        self.game.step((br, bc, lr, lc))