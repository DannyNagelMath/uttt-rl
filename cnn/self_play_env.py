import sys, os
_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_DIR))
sys.path.insert(0, _DIR)

import numpy as np
import gymnasium as gym

from train_vs_random_mlp import FlatCNNEnv


class FlatCNNSelfPlayEnv(FlatCNNEnv):
    """
    Self-play env built on FlatCNNEnv.

    Key design decisions vs the old MLP self-play env:

    1. Opponent sampled from pool every episode (not once per iteration).
       The pool list is passed by reference — new checkpoints added between
       iterations are automatically available to the next episode.

    2. No flip function needed. FlatCNNEnv._get_obs() uses current_player
       to build ch0 (my pieces) and ch1 (opponent pieces), so the observation
       is already from the current player's perspective regardless of side.
       Both the agent and the frozen opponent receive correct observations
       from _get_obs() without any transformation.

    3. Terminal-only rewards — no sub-board shaping.
    """

    def __init__(self, pool, seed_sample_prob=0.25):
        super().__init__()
        self.pool             = pool        # shared list, updated externally
        self.seed_sample_prob = seed_sample_prob
        self.agent_side       = 1           # assigned each episode
        self.opponent         = None        # sampled each episode

    # ── Gymnasium interface ───────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        gym.Env.reset(self, seed=seed)      # skip FlatCNNEnv.reset()
        self.game.reset()

        self.opponent  = self._sample_opponent()
        self.agent_side = 1 if np.random.random() < 0.5 else -1

        # If agent is O, opponent (X) moves first
        if self.agent_side == -1:
            self._opponent_move()

        return self._get_obs(), {}

    def step(self, action):
        br = action // 27
        bc = (action % 27) // 9
        lr = (action % 9)  // 3
        lc = action % 3

        _, reward, terminated, truncated, info = self.game.step((br, bc, lr, lc))

        if not terminated:
            self._opponent_move()

            if self.game.done:
                terminated = True
                if self.game.winner == 0:
                    reward = 0.5
                elif self.game.winner == self.agent_side:
                    reward = 1.0
                else:
                    reward = -1.0

        return self._get_obs(), reward, terminated, truncated, info

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _sample_opponent(self):
        if len(self.pool) == 1 or np.random.random() < self.seed_sample_prob:
            return self.pool[0]
        rest = self.pool[1:]
        if np.random.random() < 0.5:
            return rest[-1]             # most recent
        return rest[np.random.randint(len(rest))]

    def _opponent_move(self):
        # _get_obs() is called when it is the opponent's turn, so current_player
        # equals the opponent's side. The perspective-aware encoding means ch0
        # shows the opponent's own pieces — exactly what the opponent model expects.
        obs   = self._get_obs()
        masks = self.action_masks()
        action, _ = self.opponent.predict(obs, action_masks=masks, deterministic=False)
        br = action // 27
        bc = (action % 27) // 9
        lr = (action % 9)  // 3
        lc = action % 3
        self.game.step((br, bc, lr, lc))
