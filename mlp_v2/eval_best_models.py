import sys, os
_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_DIR)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _DIR)
sys.path.insert(0, os.path.join(_ROOT, "mlp"))

import numpy as np
from sb3_contrib import MaskablePPO

from uttt_game import UTTTGame
from uttt_env import UTTTEnv
from train_vs_random_mlp import FlatCNNEnv

N_GAMES_PER_SIDE = 100
DETERMINISTIC    = False

BEST_MODELS_DIR  = os.path.join(_ROOT, "best_models")

# Map filename → env class.  Old runs used UTTTEnv; run5+ use FlatCNNEnv.
def env_for(filename):
    run = int(filename.split("run")[1].split("_")[0])
    return FlatCNNEnv if run >= 5 else UTTTEnv


def play_match(model_x, env_x, model_o, env_o, n_games):
    """model_x plays as X (player 1), model_o plays as O (player -1)."""
    game = UTTTGame()
    env_x.game = game
    env_o.game = game

    x_wins = draws = o_wins = 0

    for _ in range(n_games):
        game.reset()
        done = False

        while not done:
            p = game.current_player
            if p == 1:
                obs, masks = env_x._get_obs(), env_x.action_masks()
                action, _ = model_x.predict(obs, action_masks=masks, deterministic=DETERMINISTIC)
            else:
                obs, masks = env_o._get_obs(), env_o.action_masks()
                action, _ = model_o.predict(obs, action_masks=masks, deterministic=DETERMINISTIC)

            br = action // 27
            bc = (action % 27) // 9
            lr = (action % 9)  // 3
            lc = action % 3
            _, _, terminated, truncated, _ = game.step((br, bc, lr, lc))
            done = terminated or truncated

        w = game.winner
        if   w == 0:  draws  += 1
        elif w == 1:  x_wins += 1
        else:         o_wins += 1

    return x_wins, draws, o_wins


def main():
    # ── Load models ───────────────────────────────────────────────────────────
    entries = sorted(os.listdir(BEST_MODELS_DIR))
    models  = []
    for fname in entries:
        if not fname.endswith(".zip"):
            continue
        path  = os.path.join(BEST_MODELS_DIR, fname)
        name  = fname.replace(".zip", "")
        env   = env_for(fname)()
        model = MaskablePPO.load(path)
        models.append((name, model, env))
        print(f"Loaded: {name}  ({env.__class__.__name__})")

    n = len(models)
    print(f"\n{n} models loaded. Running round-robin ({n*(n-1)//2} matchups × "
          f"{N_GAMES_PER_SIDE * 2} games each)...\n")

    wins  = np.zeros((n, n), dtype=int)
    draws = np.zeros((n, n), dtype=int)

    for i in range(n):
        for j in range(i + 1, n):
            name_i, model_i, env_i = models[i]
            name_j, model_j, env_j = models[j]

            xw, d1, ow = play_match(model_i, env_i, model_j, env_j, N_GAMES_PER_SIDE)
            wins[i][j] += xw;  wins[j][i] += ow
            draws[i][j] += d1; draws[j][i] += d1

            xw, d2, ow = play_match(model_j, env_j, model_i, env_i, N_GAMES_PER_SIDE)
            wins[j][i] += xw;  wins[i][j] += ow
            draws[j][i] += d2; draws[i][j] += d2

            print(f"  {name_i[-20:]:>20} vs {name_j[-20:]:<20}  "
                  f"{name_i.split('_')[-1]} {wins[i][j]}W/{draws[i][j]}D/{wins[j][i]}L")

    # ── Rankings ──────────────────────────────────────────────────────────────
    total_games = (n - 1) * N_GAMES_PER_SIDE * 2
    scores = []
    for i, (name, _, __) in enumerate(models):
        pts = wins[i].sum() + 0.5 * draws[i].sum()
        scores.append((name, pts, int(wins[i].sum()), int(draws[i].sum()), int(wins[:, i].sum())))

    scores.sort(key=lambda x: x[1], reverse=True)

    print(f"\n-- Final Rankings (out of {total_games} games each) --------------")
    print(f"  {'Rank':<5} {'Model':<35} {'Score':>7} {'Wins':>6} {'Draws':>6} {'Losses':>7}")
    print(f"  {'-'*65}")
    for rank, (name, pts, w, d, l) in enumerate(scores, 1):
        print(f"  {rank:<5} {name:<35} {pts:>7.1f} {w:>6} {d:>6} {l:>7}")


if __name__ == "__main__":
    main()
