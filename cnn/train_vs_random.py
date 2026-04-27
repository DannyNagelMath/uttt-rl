import sys, os
_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_DIR))
sys.path.insert(0, _DIR)

import csv
import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback

from cnn_env import CNNEnv
from feature_extractor import UTTTFeatureExtractor

# ── Hyperparameters ───────────────────────────────────────────────────────────
TOTAL_TIMESTEPS     = 200_000
N_STEPS             = 2048       # rollout length (timesteps per update)
LEARNING_RATE       = 3e-4
N_FILTERS           = 64
FEATURES_DIM        = 256
EVAL_FREQ_ROLLOUTS  = 10         # evaluate every N rollouts
N_EVAL_GAMES        = 100        # games per evaluation checkpoint

MODEL_DIR           = os.path.join(_DIR, "models_seed")
MODEL_PATH          = os.path.join(MODEL_DIR, "uttt_cnn_seed")
CSV_PATH            = os.path.join(_DIR, "diagnostics_seed.csv")
# ─────────────────────────────────────────────────────────────────────────────


class DiagnosticCallback(BaseCallback):

    def __init__(self, csv_path, eval_freq_rollouts, n_eval_games):
        super().__init__()
        self.csv_path           = csv_path
        self.eval_freq_rollouts = eval_freq_rollouts
        self.n_eval_games       = n_eval_games
        self.rollout_count      = 0
        self._csv_ready         = False

    def _init_csv(self):
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
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
        game_lengths   = []
        sub_board_diffs = []

        eval_env = CNNEnv()

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
            sub_board_diffs.append(
                int(np.sum(sw == 1)) - int(np.sum(sw == -1))
            )

            if reward == 1.0:
                wins += 1
            elif reward == 0.5:
                draws += 1
            else:
                losses += 1

        n = self.n_eval_games
        row = [
            self.num_timesteps,
            self.rollout_count,
            round(wins  / n, 3),
            round(draws / n, 3),
            round(losses / n, 3),
            round(float(np.mean(game_lengths)), 1),
            round(float(np.mean(sub_board_diffs)), 2),
        ]

        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

        print(
            f"  [eval] step={self.num_timesteps:>7}  "
            f"win={row[2]:.3f}  draw={row[3]:.3f}  loss={row[4]:.3f}  "
            f"len={row[5]:.1f}  sub_diff={row[6]:+.2f}"
        )

    def _on_step(self):
        return True


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Hyperparameters:")
    print(f"  TOTAL_TIMESTEPS    = {TOTAL_TIMESTEPS}")
    print(f"  N_STEPS            = {N_STEPS}")
    print(f"  LEARNING_RATE      = {LEARNING_RATE}")
    print(f"  N_FILTERS          = {N_FILTERS}")
    print(f"  FEATURES_DIM       = {FEATURES_DIM}")
    print(f"  EVAL_FREQ_ROLLOUTS = {EVAL_FREQ_ROLLOUTS}")
    print(f"  N_EVAL_GAMES       = {N_EVAL_GAMES}")
    print()

    env = CNNEnv()

    policy_kwargs = {
        "features_extractor_class":  UTTTFeatureExtractor,
        "features_extractor_kwargs": {
            "features_dim": FEATURES_DIM,
            "n_filters":    N_FILTERS,
        },
    }

    model = MaskablePPO(
        "CnnPolicy",
        env,
        policy_kwargs   = policy_kwargs,
        n_steps         = N_STEPS,
        learning_rate   = LEARNING_RATE,
        verbose         = 1,
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
