import sys, os
_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_DIR))
sys.path.insert(0, _DIR)

import numpy as np
from sb3_contrib import MaskablePPO

from train_vs_random_mlp import FlatCNNEnv

N_GAMES_PER_SIDE = 100   # × 2 sides = 200 games per pairing
DETERMINISTIC    = False

SEED_PATH    = os.path.join(_DIR, "models_seed",     "uttt_mlp_flatcnn_seed.zip")
SELFPLAY_DIR = os.path.join(_DIR, "models_self_play")


def play_match(model_x, model_o, n_games):
    """model_x = X (moves first), model_o = O. Returns (x_wins, draws, o_wins)."""
    env = FlatCNNEnv()
    x_wins = draws = o_wins = 0

    for _ in range(n_games):
        env.game.reset()
        done = False

        while not done:
            p     = env.game.current_player
            model = model_x if p == 1 else model_o
            obs   = env._get_obs()
            masks = env.action_masks()
            action, _ = model.predict(obs, action_masks=masks, deterministic=DETERMINISTIC)

            br = action // 27
            bc = (action % 27) // 9
            lr = (action % 9)  // 3
            lc = action % 3
            _, _, terminated, truncated, _ = env.game.step((br, bc, lr, lc))
            done = terminated or truncated

        w = env.game.winner
        if   w == 0:  draws  += 1
        elif w == 1:  x_wins += 1
        else:         o_wins += 1

    return x_wins, draws, o_wins


def main():
    # ── Load models ───────────────────────────────────────────────────────────
    models = [("seed", MaskablePPO.load(SEED_PATH))]
    print(f"Loaded: seed")

    for i in range(1, 11):
        path = os.path.join(SELFPLAY_DIR, f"selfplay_{i * 500_000}.zip")
        if os.path.exists(path):
            name = f"sp_{i * 500_000 // 1000}k"
            models.append((name, MaskablePPO.load(path)))
            print(f"Loaded: {name}")

    n = len(models)
    print(f"\n{n} models loaded. Running round-robin ({n*(n-1)//2} matchups × "
          f"{N_GAMES_PER_SIDE*2} games each)...\n")

    # ── Score matrix: wins[i][j] = wins for model i against model j ───────────
    wins   = np.zeros((n, n), dtype=int)
    draws  = np.zeros((n, n), dtype=int)

    for i in range(n):
        for j in range(i + 1, n):
            name_i, model_i = models[i]
            name_j, model_j = models[j]

            # i as X, j as O
            xw, d1, ow = play_match(model_i, model_j, N_GAMES_PER_SIDE)
            wins[i][j] += xw;  wins[j][i] += ow;  draws[i][j] += d1;  draws[j][i] += d1

            # j as X, i as O
            xw, d2, ow = play_match(model_j, model_i, N_GAMES_PER_SIDE)
            wins[j][i] += xw;  wins[i][j] += ow;  draws[j][i] += d2;  draws[i][j] += d2

            total_ij = N_GAMES_PER_SIDE * 2
            print(f"  {name_i:>10} vs {name_j:<10}  "
                  f"{name_i} {wins[i][j]}W/{draws[i][j]}D/{wins[j][i]}L")

    # ── Rankings by total score (win=1, draw=0.5) ─────────────────────────────
    total_games = (n - 1) * N_GAMES_PER_SIDE * 2
    scores = []
    for i, (name, _) in enumerate(models):
        pts = wins[i].sum() + 0.5 * draws[i].sum()
        scores.append((name, pts, wins[i].sum(), draws[i].sum(), wins[:, i].sum()))

    scores.sort(key=lambda x: x[1], reverse=True)

    print(f"\n── Final Rankings (out of {total_games} games each) ──────────────")
    print(f"  {'Rank':<5} {'Model':<12} {'Score':>7} {'Wins':>6} {'Draws':>6} {'Losses':>7}")
    print(f"  {'-'*50}")
    for rank, (name, pts, w, d, l) in enumerate(scores, 1):
        print(f"  {rank:<5} {name:<12} {pts:>7.1f} {w:>6} {d:>6} {l:>7}")


if __name__ == "__main__":
    main()
