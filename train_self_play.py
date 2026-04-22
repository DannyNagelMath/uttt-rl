import os
import time
import numpy as np
from tqdm import tqdm

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback

from self_play_env import SelfPlayEnv

# ── Configuration ─────────────────────────────────────────────────────────────
SEED_MODEL_PATH    = "uttt_maskable_ppo.zip"
MODELS_DIR         = "models"
STEPS_PER_ITER     = 500_000   # training steps between checkpoints
NUM_ITERATIONS     = 10        # total iterations  (5M steps)
PROGRESS_INTERVAL  = 1_000     # tqdm update frequency (steps)

SEED_SAMPLE_PROB   = 0.25      # guaranteed minimum probability of facing seed
WIN_RATE_THRESHOLD = 0.55      # min score vs pool to add checkpoint to pool
GATE_GAMES         = 40        # games per side (80 total) for gating eval


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

        print(f"── Iteration {iteration}/{NUM_ITERATIONS} "
              f"(steps this iter: {steps_this_iter:,} | "
              f"total so far: {total_steps_so_far:,}) ──")
        print(f"   Opponent: pool size {len(pool)}")

        start    = time.time()
        callback = TqdmCallback(total_steps=steps_this_iter)

        model.learn(
            total_timesteps=steps_this_iter,
            callback=callback,
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
