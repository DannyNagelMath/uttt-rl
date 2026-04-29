import sys, os
_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_DIR))
sys.path.insert(0, _DIR)

import csv
import time
import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback

from train_vs_random_mlp import FlatCNNEnv
from self_play_env import FlatCNNSelfPlayEnv

# ── Configuration ─────────────────────────────────────────────────────────────
SEED_MODEL_PATH    = os.path.join(_DIR, "models_seed", "uttt_mlp_flatcnn_seed.zip")
MODELS_DIR         = os.path.join(_DIR, "models_self_play")
STEPS_PER_ITER     = 500_000
NUM_ITERATIONS     = 10

SEED_SAMPLE_PROB   = 0.25
WIN_RATE_THRESHOLD = 0.55
GATE_GAMES         = 40        # played as each side → 80 total

METRICS_FILE       = os.path.join(_DIR, "self_play_metrics.csv")
EVAL_FREQ_ROLLOUTS = 10        # diagnostic eval every N rollouts
N_EVAL_GAMES       = 100
# ─────────────────────────────────────────────────────────────────────────────


# ── Diagnostic callback ───────────────────────────────────────────────────────
class DiagnosticCallback(BaseCallback):

    def __init__(self, iteration, metrics_file, eval_freq_rollouts, n_eval_games):
        super().__init__()
        self.iteration         = iteration
        self.metrics_file      = metrics_file
        self.eval_freq_rollouts = eval_freq_rollouts
        self.n_eval_games      = n_eval_games
        self.rollout_count     = 0

    def _on_rollout_end(self):
        self.rollout_count += 1
        if self.rollout_count % self.eval_freq_rollouts != 0:
            return

        wins, draws, losses = 0, 0, 0
        game_lengths = []
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
            if reward == 1.0:   wins   += 1
            elif reward == 0.5: draws  += 1
            else:               losses += 1

        n = self.n_eval_games
        win_rate = round(wins / n, 3)
        row = [
            self.iteration, self.num_timesteps, self.rollout_count,
            win_rate, round(draws / n, 3), round(losses / n, 3),
            round(float(np.mean(game_lengths)), 1),
        ]

        write_header = not os.path.exists(self.metrics_file)
        with open(self.metrics_file, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["iteration", "timestep", "rollout",
                            "win_vs_random", "draw_vs_random", "loss_vs_random",
                            "avg_game_length"])
            w.writerow(row)

        print(
            f"  [eval] iter={self.iteration}  step={self.num_timesteps:>8}  "
            f"win_vs_random={win_rate:.3f}  "
            f"draw={row[4]:.3f}  loss={row[5]:.3f}  len={row[6]:.1f}"
        )

    def _on_step(self):
        return True


# ── Pool evaluation (quality gate) ───────────────────────────────────────────
def eval_vs_pool(model, pool, n_games=GATE_GAMES):
    """
    Play n_games as each side vs random pool members.
    Returns score in [0,1]: win=1, draw=0.5, loss=0.
    """
    score = 0.0
    total = n_games * 2
    env   = FlatCNNSelfPlayEnv(pool, seed_sample_prob=0.0)  # uniform pool sampling

    for _ in range(total):
        obs, _ = env.reset()
        done   = False
        while not done:
            action, _ = model.predict(
                obs, action_masks=env.action_masks(), deterministic=False
            )
            obs, reward, done, _, _ = env.step(action)
        score += max(reward, 0)

    return score / total


# ── Main training loop ────────────────────────────────────────────────────────
def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    print(f"Loading seed model: {SEED_MODEL_PATH}")
    seed_model = MaskablePPO.load(SEED_MODEL_PATH)
    pool       = [seed_model]

    # Warm-start training model from seed weights
    env   = FlatCNNSelfPlayEnv(pool, seed_sample_prob=SEED_SAMPLE_PROB)
    model = MaskablePPO.load(SEED_MODEL_PATH, env=env)

    print(f"Seed loaded. Beginning self-play ({NUM_ITERATIONS} iterations × "
          f"{STEPS_PER_ITER:,} steps = {NUM_ITERATIONS * STEPS_PER_ITER:,} total)\n")

    for iteration in range(1, NUM_ITERATIONS + 1):
        print(f"-- Iteration {iteration}/{NUM_ITERATIONS}  "
              f"(pool size: {len(pool)}) --")

        callback = DiagnosticCallback(
            iteration          = iteration,
            metrics_file       = METRICS_FILE,
            eval_freq_rollouts = EVAL_FREQ_ROLLOUTS,
            n_eval_games       = N_EVAL_GAMES,
        )

        start = time.time()
        model.learn(
            total_timesteps   = STEPS_PER_ITER,
            callback          = callback,
            reset_num_timesteps = False,
        )
        print(f"   Completed in {time.time() - start:.1f}s")

        ckpt_path = os.path.join(MODELS_DIR, f"selfplay_{iteration * STEPS_PER_ITER}")
        model.save(ckpt_path)
        print(f"   Saved: {ckpt_path}.zip")

        score = eval_vs_pool(model, pool)
        print(f"   Score vs pool: {score:.1%}")

        if score >= WIN_RATE_THRESHOLD:
            frozen = MaskablePPO.load(ckpt_path + ".zip")
            pool.append(frozen)
            print(f"   Added to pool (pool size now: {len(pool)})\n")
        else:
            print(f"   Not added (below {WIN_RATE_THRESHOLD:.0%} threshold)\n")

    print("Self-play training complete.")


if __name__ == "__main__":
    main()
