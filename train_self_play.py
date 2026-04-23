import os
import csv
import time
import random as rnd
import numpy as np
from tqdm import tqdm

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback

from self_play_env import SelfPlayEnv
from uttt_game import UTTTGame
from uttt_env import UTTTEnv
from utils import flip_obs

# ── Configuration ─────────────────────────────────────────────────────────────
SEED_MODEL_PATH    = "uttt_maskable_ppo.zip"
MODELS_DIR         = "models"
STEPS_PER_ITER     = 1_500_000  # training steps between checkpoints
NUM_ITERATIONS     = 10         # total iterations  (15M steps)
PROGRESS_INTERVAL  = 1_000      # tqdm update frequency (steps)

SEED_SAMPLE_PROB   = 0.25       # guaranteed minimum probability of facing seed
WIN_RATE_THRESHOLD = 0.55       # min score vs pool to add checkpoint to pool
GATE_GAMES         = 40         # games per side (80 total) for gating eval

METRICS_FILE       = "training_metrics.csv"
LOG_INTERVAL       = 10_000     # steps between metric log rows
EVAL_INTERVAL      = 100_000    # steps between vs-random win-rate evals
EVAL_GAMES         = 20         # games for quick vs-random eval (10 as X, 10 as O)


# ── Progress bar callback ─────────────────────────────────────────────────────
class TqdmCallback(BaseCallback):
    """Updates a tqdm progress bar every PROGRESS_INTERVAL steps."""

    def __init__(self, total_steps, progress_interval=PROGRESS_INTERVAL):
        super().__init__()
        self.total_steps       = total_steps
        self.progress_interval = progress_interval
        self.pbar              = None
        self._last_update      = 0

    def _on_training_start(self):
        self.pbar = tqdm(
            total=self.total_steps,
            unit="step",
            dynamic_ncols=True,
        )
        self._last_update = 0

    def _on_step(self):
        delta = self.num_timesteps - self._last_update
        if delta >= self.progress_interval:
            self.pbar.update(delta)
            self._last_update = self.num_timesteps
        return True

    def _on_training_end(self):
        remaining = self.num_timesteps - self._last_update
        if remaining > 0:
            self.pbar.update(remaining)
        self.pbar.close()


# ── Metrics callback ──────────────────────────────────────────────────────────
class MetricsCallback(BaseCallback):
    """
    Logs training metrics to CSV every LOG_INTERVAL steps.
    Every EVAL_INTERVAL steps also runs a quick vs-random win-rate eval
    and records it in the same row.

    Columns: iteration, timestep, ep_rew_mean, value_loss,
             explained_variance, entropy_loss, vs_random_wr
    """

    def __init__(self, metrics_file, iteration,
                 log_interval=LOG_INTERVAL, eval_interval=EVAL_INTERVAL):
        super().__init__()
        self.metrics_file  = metrics_file
        self.iteration     = iteration
        self.log_interval  = log_interval
        self.eval_interval = eval_interval
        self._last_log     = 0
        self._last_eval    = 0

    def _on_training_start(self):
        # Anchor to current timestep so the first log fires after log_interval
        # steps into this iteration, not immediately.
        self._last_log  = self.num_timesteps
        self._last_eval = self.num_timesteps

        if not os.path.exists(self.metrics_file):
            with open(self.metrics_file, 'w', newline='') as f:
                csv.writer(f).writerow([
                    'iteration', 'timestep', 'ep_rew_mean',
                    'value_loss', 'explained_variance', 'entropy_loss',
                    'vs_random_wr',
                ])

    def _on_step(self):
        if self.num_timesteps - self._last_log < self.log_interval:
            return True

        vs_wr = ''
        if self.num_timesteps - self._last_eval >= self.eval_interval:
            vs_wr = f"{quick_eval_vs_random(self.model):.3f}"
            self._last_eval = self.num_timesteps

        lv = self.logger.name_to_value
        with open(self.metrics_file, 'a', newline='') as f:
            csv.writer(f).writerow([
                self.iteration,
                self.num_timesteps,
                lv.get('rollout/ep_rew_mean', ''),
                lv.get('train/value_loss', ''),
                lv.get('train/explained_variance', ''),
                lv.get('train/entropy_loss', ''),
                vs_wr,
            ])

        self._last_log = self.num_timesteps
        return True


# ── Opponent sampling ─────────────────────────────────────────────────────────
def sample_opponent(pool):
    """
    Recency-biased sampling with seed anchoring.

    SEED_SAMPLE_PROB: always face seed (pool[0]) at minimum rate so the
    model never forgets how to play general UTTT.
    Remaining probability: 50% most recent non-seed, 50% uniform non-seed.
    """
    if len(pool) == 1 or np.random.random() < SEED_SAMPLE_PROB:
        return pool[0]  # seed
    rest = pool[1:]
    if np.random.random() < 0.5:
        return rest[-1]  # most recent
    return rest[np.random.randint(len(rest))]


# ── Quick vs-random evaluation ────────────────────────────────────────────────
def quick_eval_vs_random(model, n_games=EVAL_GAMES):
    """
    Play n_games/2 as X and n_games/2 as O against a random opponent.
    Uses stochastic play (deterministic=False) for a realistic policy sample.
    Returns win rate in [0, 1].
    """
    env  = UTTTEnv()
    wins = 0
    half = n_games // 2
    for i in range(n_games):
        model_is_x = (i < half)
        game = UTTTGame()
        env.game = game
        while not game.done:
            obs   = env._get_obs()
            masks = env.action_masks()
            is_model_turn = (
                (model_is_x     and game.current_player ==  1) or
                (not model_is_x and game.current_player == -1)
            )
            if is_model_turn:
                agent_obs = obs if model_is_x else flip_obs(obs)
                action, _ = model.predict(
                    agent_obs, action_masks=masks, deterministic=False
                )
            else:
                legal  = np.where(masks)[0]
                action = int(rnd.choice(legal))
            br = action // 27
            bc = (action % 27) // 9
            lr = (action % 9)  // 3
            lc = action % 3
            game.step((br, bc, lr, lc))
        if (model_is_x and game.winner == 1) or (not model_is_x and game.winner == -1):
            wins += 1
    return wins / n_games


# ── Win-rate gating ───────────────────────────────────────────────────────────
def eval_vs_pool(model, pool, num_games=GATE_GAMES):
    """
    Stochastic evaluation: play num_games as each side vs random pool members.
    Returns score in [0, 1]: win=1, draw=0.5, loss=0.
    Uses deterministic=False so the score reflects the policy distribution,
    not a single deterministic path that could be an exploit.
    """
    score = 0.0
    total = num_games * 2
    for _ in range(total):
        opponent = pool[np.random.randint(len(pool))]
        env = SelfPlayEnv(opponent)
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(
                obs, action_masks=env.action_masks(), deterministic=False
            )
            obs, reward, done, _, _ = env.step(action)
        score += max(reward, 0)  # 1.0 win, 0.5 draw, 0.0 loss
    return score / total


# ── Main training loop ────────────────────────────────────────────────────────
def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Clear metrics file so each run starts fresh
    if os.path.exists(METRICS_FILE):
        os.remove(METRICS_FILE)

    print(f"Loading seed model from {SEED_MODEL_PATH} ...")
    seed_model = MaskablePPO.load(SEED_MODEL_PATH)
    pool = [seed_model]
    print("Seed model loaded. Starting self-play training.\n")

    # Warm-start from seed weights so self-play refines an already-competent
    # player rather than building UTTT knowledge from scratch.
    opponent = sample_opponent(pool)
    env      = SelfPlayEnv(opponent)
    model    = MaskablePPO.load(SEED_MODEL_PATH)
    model.set_env(env)

    total_steps_so_far = 0

    for iteration in range(1, NUM_ITERATIONS + 1):
        opponent = sample_opponent(pool)
        env      = SelfPlayEnv(opponent)
        model.set_env(env)

        steps_this_iter     = STEPS_PER_ITER
        total_steps_so_far += steps_this_iter

        print(f"-- Iteration {iteration}/{NUM_ITERATIONS} "
              f"(steps this iter: {steps_this_iter:,} | "
              f"total so far: {total_steps_so_far:,}) --")
        print(f"   Opponent: pool size {len(pool)}")

        start     = time.time()
        callbacks = [
            TqdmCallback(total_steps=steps_this_iter),
            MetricsCallback(METRICS_FILE, iteration=iteration),
        ]

        model.learn(
            total_timesteps=steps_this_iter,
            callback=callbacks,
            reset_num_timesteps=False,
        )

        elapsed = time.time() - start
        print(f"   Completed in {elapsed:.1f}s")

        # Always save the checkpoint for offline analysis.
        ckpt_path = os.path.join(
            MODELS_DIR, f"uttt_selfplay_{total_steps_so_far}"
        )
        model.save(ckpt_path)
        print(f"   Checkpoint saved: {ckpt_path}.zip")

        # Only add to the pool if the model demonstrates genuine improvement.
        # Stochastic evaluation prevents a narrow exploit from passing the gate.
        win_rate = eval_vs_pool(model, pool)
        print(f"   Win rate vs pool: {win_rate:.1%}")
        if win_rate >= WIN_RATE_THRESHOLD:
            frozen = MaskablePPO.load(ckpt_path + ".zip")
            pool.append(frozen)
            print(f"   Added to pool (pool size: {len(pool)})\n")
        else:
            print(f"   NOT added to pool (below {WIN_RATE_THRESHOLD:.0%} threshold)\n")

    print("\nSelf-play training complete.")
    print(f"Checkpoints saved in '{MODELS_DIR}/':")
    for fname in sorted(os.listdir(MODELS_DIR)):
        print(f"  {fname}")


if __name__ == "__main__":
    main()
