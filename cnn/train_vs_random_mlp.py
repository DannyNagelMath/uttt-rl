import sys, os
_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_DIR))
sys.path.insert(0, _DIR)

import csv
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback

from cnn_env import CNNEnv

# ── Hyperparameters ───────────────────────────────────────────────────────────
TOTAL_TIMESTEPS     = 5_000_000
N_STEPS             = 2048
LEARNING_RATE       = 3e-4
EVAL_FREQ_ROLLOUTS  = 10
N_EVAL_GAMES        = 100

MODEL_DIR           = os.path.join(_DIR, "models_seed")
MODEL_PATH          = os.path.join(MODEL_DIR, "uttt_mlp_flatcnn_seed")
CSV_PATH            = os.path.join(_DIR, "diagnostics_mlp_flatcnn.csv")
# ─────────────────────────────────────────────────────────────────────────────


class FlatCNNEnv(CNNEnv):
    """CNNEnv with (6,9,9) observation flattened to (486,) for MlpPolicy."""

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(486,), dtype=np.float32
        )

    def _get_obs(self):
        return super()._get_obs().flatten()


class DiagnosticCallback(BaseCallback):

    def __init__(self, csv_path, eval_freq_rollouts, n_eval_games):
        super().__init__()
        self.csv_path           = csv_path
        self.eval_freq_rollouts = eval_freq_rollouts
        self.n_eval_games       = n_eval_games
        self.rollout_count      = 0
        self._csv_ready         = False

    def _init_csv(self):
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                csv.writer(f).writerow([
                    "timestep", "rollout",
                    "win_rate", "draw_rate", "loss_rate",
                    "avg_game_length", "avg_sub_board_diff",
                ])
        self._csv_ready = True

    def _on_rollout_end(self):
        if not self._csv_ready:
            self._init_csv()

        self.rollout_count += 1
        if self.rollout_count % self.eval_freq_rollouts != 0:
            return

        wins, draws, losses = 0, 0, 0
        game_lengths    = []
        sub_board_diffs = []

        eval_env = FlatCNNEnv()

        for _ in range(self.n_eval_games):
            obs, _ = eval_env.reset()
            done   = False
            steps  = 0

            while not done:
                masks = eval_env.action_masks()
                action, _ = self.model.predict(obs, action_masks=masks, deterministic=True)
                obs, reward, done, _, _ = eval_env.step(action)
                steps += 1

            game_lengths.append(steps)
            sw = eval_env.game.sub_board_winners
            sub_board_diffs.append(int(np.sum(sw == 1)) - int(np.sum(sw == -1)))

            if reward == 1.0:   wins   += 1
            elif reward == 0.5: draws  += 1
            else:               losses += 1

        n = self.n_eval_games
        row = [
            self.num_timesteps,
            self.rollout_count,
            round(wins   / n, 3),
            round(draws  / n, 3),
            round(losses / n, 3),
            round(float(np.mean(game_lengths)), 1),
            round(float(np.mean(sub_board_diffs)), 2),
        ]

        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

        print(
            f"  [eval] step={self.num_timesteps:>8}  "
            f"win={row[2]:.3f}  draw={row[3]:.3f}  loss={row[4]:.3f}  "
            f"len={row[5]:.1f}  sub_diff={row[6]:+.2f}"
        )

    def _on_step(self):
        return True


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("MlpPolicy on flattened CNNEnv (486-dim observation)")
    print("Hyperparameters:")
    print(f"  TOTAL_TIMESTEPS    = {TOTAL_TIMESTEPS:,}")
    print(f"  N_STEPS            = {N_STEPS}")
    print(f"  LEARNING_RATE      = {LEARNING_RATE}")
    print(f"  EVAL_FREQ_ROLLOUTS = {EVAL_FREQ_ROLLOUTS}")
    print(f"  N_EVAL_GAMES       = {N_EVAL_GAMES}")
    print()

    env = FlatCNNEnv()

    checkpoint = MODEL_PATH + ".zip"
    if os.path.exists(checkpoint):
        print(f"Resuming from checkpoint: {checkpoint}\n")
        model = MaskablePPO.load(checkpoint, env=env)
    else:
        model = MaskablePPO(
            "MlpPolicy",
            env,
            n_steps       = N_STEPS,
            learning_rate = LEARNING_RATE,
            verbose       = 1,
        )

    callback = DiagnosticCallback(
        csv_path           = CSV_PATH,
        eval_freq_rollouts = EVAL_FREQ_ROLLOUTS,
        n_eval_games       = N_EVAL_GAMES,
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)
    model.save(MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}.zip")


if __name__ == "__main__":
    main()
