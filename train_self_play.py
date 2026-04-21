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
        return True   # returning False would abort training

    def _on_training_end(self):
        # Flush any remaining steps
        remaining = self.num_timesteps - self._last_update
        if remaining > 0:
            self.pbar.update(remaining)
        self.pbar.close()


# ── Opponent sampling ─────────────────────────────────────────────────────────
def sample_opponent(pool):
    """
    Recency-biased sampling from the checkpoint pool.

    50% chance: most recent checkpoint
    50% chance: uniform draw from the full pool

    When pool has only one entry both branches return the same model.
    """
    if np.random.random() < 0.5:
        return pool[-1]                          # most recent
    else:
        idx = np.random.randint(len(pool))
        return pool[idx]                         # uniform


# ── Main training loop ────────────────────────────────────────────────────────
def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Load seed model into the pool
    print(f"Loading seed model from {SEED_MODEL_PATH} ...")
    seed_model = MaskablePPO.load(SEED_MODEL_PATH)
    pool = [seed_model]
    print("Seed model loaded. Starting self-play training.\n")

    # Build the current (trainable) model against a temporary env.
    # The env will be replaced each iteration with a freshly sampled opponent.
    opponent  = sample_opponent(pool)
    env       = SelfPlayEnv(opponent)
    model     = MaskablePPO("MlpPolicy", env, verbose=0)

    total_steps_so_far = 0

    for iteration in range(1, NUM_ITERATIONS + 1):
        opponent = sample_opponent(pool)
        env      = SelfPlayEnv(opponent)
        model.set_env(env)

        steps_this_iter = STEPS_PER_ITER
        total_steps_so_far += steps_this_iter

        print(f"── Iteration {iteration}/{NUM_ITERATIONS} "
              f"(steps this iter: {steps_this_iter:,} | "
              f"total so far: {total_steps_so_far:,}) ──")
        print(f"   Opponent: checkpoint {len(pool)} of {len(pool)} in pool")

        start = time.time()
        callback = TqdmCallback(total_steps=steps_this_iter)

        model.learn(
            total_timesteps=steps_this_iter,
            callback=callback,
            reset_num_timesteps=False,  # keeps global step count continuous
        )

        elapsed = time.time() - start
        print(f"   Completed in {elapsed:.1f}s\n")

        # Save checkpoint and add a fresh load to the pool
        ckpt_path = os.path.join(
            MODELS_DIR, f"uttt_selfplay_{total_steps_so_far}"
        )
        model.save(ckpt_path)
        print(f"   Checkpoint saved: {ckpt_path}.zip")

        frozen = MaskablePPO.load(ckpt_path + ".zip")
        pool.append(frozen)

    print("\nSelf-play training complete.")
    print(f"Checkpoints saved in '{MODELS_DIR}/':")
    for fname in sorted(os.listdir(MODELS_DIR)):
        print(f"  {fname}")


if __name__ == "__main__":
    main()