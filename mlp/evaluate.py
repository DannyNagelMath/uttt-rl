"""
evaluate.py — Evaluate UTTT model checkpoints via win rates and ELO ratings.

Metrics computed:
  - Win rate vs random opponent (as X and as O)
  - Win rate vs seed model (for self-play checkpoints)
  - ELO ratings from full round-robin between all models

Output:
  - Summary table printed to stdout
  - Full results saved to CSV (OUTPUT_FILE)
"""

import sys, os
_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_DIR))  # project root -> finds uttt_game

import re
import csv
import random
import numpy as np
from sb3_contrib import MaskablePPO

from uttt_game import UTTTGame
from uttt_env import UTTTEnv
from utils import flip_obs

# ── Constants ─────────────────────────────────────────────────────────────────

NUM_GAMES     = 100          # games per matchup (each side); increase for tighter estimates
INITIAL_ELO   = 1000         # starting ELO for all models
K_FACTOR      = 32           # ELO update rate
MODELS_DIR    = os.path.join(_DIR, "models")
SEED_MODEL    = os.path.join(_DIR, "uttt_maskable_ppo.zip")
OUTPUT_FILE   = os.path.join(_DIR, "evaluate_results.csv")

# ── Model loading ─────────────────────────────────────────────────────────────

def discover_models():
    """
    Returns an ordered list of (label, path) tuples.
    Seed model is first; self-play checkpoints follow, sorted by step count.
    """
    models = [(SEED_MODEL.replace(".zip", ""), SEED_MODEL)]

    pattern = re.compile(r"uttt_selfplay_(\d+)\.zip")
    checkpoints = []
    for fname in os.listdir(MODELS_DIR):
        m = pattern.match(fname)
        if m:
            steps = int(m.group(1))
            checkpoints.append((steps, fname))
    checkpoints.sort(key=lambda x: x[0])

    for steps, fname in checkpoints:
        label = f"selfplay_{steps}"
        path  = os.path.join(MODELS_DIR, fname)
        models.append((label, path))

    return models


def load_models(model_list):
    """
    Loads all models. Returns list of (label, model) tuples.
    """
    loaded = []
    for label, path in model_list:
        print(f"Loading {label} from {path} ...")
        model = MaskablePPO.load(path)
        loaded.append((label, model))
    print(f"\nLoaded {len(loaded)} models.\n")
    return loaded

# ── Action decoding ───────────────────────────────────────────────────────────

def decode_action(action):
    """
    Converts a flat action integer (0–80) to a (br, bc, lr, lc) tuple,
    matching the encoding in UTTTEnv: action = br*27 + bc*9 + lr*3 + lc
    """
    br = action // 27
    bc = (action % 27) // 9
    lr = (action % 9) // 3
    lc = action % 3
    return (br, bc, lr, lc)

# ── Result helper ─────────────────────────────────────────────────────────────

def _result_from_game(game, model_is_x):
    """
    Returns 1 (model won), 0 (draw), -1 (model lost).
    Reads game.winner directly — valid only after game.done is True.
    """
    if game.winner == 0:
        return 0  # draw
    if model_is_x:
        return 1 if game.winner == 1 else -1
    else:
        return 1 if game.winner == -1 else -1

# ── Random opponent ───────────────────────────────────────────────────────────

def play_vs_random(model, env, num_games, model_is_x):
    """
    Plays num_games against a random opponent.
    model_is_x: True if model plays as X, False if model plays as O.

    Returns (wins, draws, losses) from the model's perspective.
    """
    wins = draws = losses = 0

    for _ in range(num_games):
        game = UTTTGame()
        env.game = game
        terminated = False

        while not terminated:
            current_player = game.current_player
            obs = env._get_obs()
            masks = env.action_masks()

            model_turn = (model_is_x and current_player == 1) or \
                         (not model_is_x and current_player == -1)

            if model_turn:
                agent_obs = obs if model_is_x else flip_obs(obs)
                action, _ = model.predict(agent_obs, action_masks=masks, deterministic=True)
            else:
                # Random legal move
                legal = np.where(masks)[0]
                action = int(random.choice(legal))

            _, _, terminated, _, _ = game.step(decode_action(int(action)))

        result = _result_from_game(game, model_is_x)
        if result == 1:
            wins += 1
        elif result == 0:
            draws += 1
        else:
            losses += 1

    return wins, draws, losses

# ── Round-robin matchups ──────────────────────────────────────────────────────

def run_matchup(label_a, model_a, label_b, model_b, env, num_games):
    """
    Runs num_games with model_a as X and num_games with model_b as X.
    Returns dict with win/draw/loss counts for model_a across all 2*num_games games.
    """
    wins = draws = losses = 0

    # model_a as X
    for _ in range(num_games):
        game = UTTTGame()
        env.game = game
        terminated = False
        while not terminated:
            obs   = env._get_obs()
            masks = env.action_masks()
            if game.current_player == 1:
                action, _ = model_a.predict(obs, action_masks=masks, deterministic=True)
            else:
                action, _ = model_b.predict(flip_obs(obs), action_masks=masks, deterministic=True)
            _, _, terminated, _, _ = game.step(decode_action(int(action)))
        r = _result_from_game(game, model_is_x=True)
        if r == 1: wins += 1
        elif r == 0: draws += 1
        else: losses += 1

    # model_b as X (model_a as O)
    for _ in range(num_games):
        game = UTTTGame()
        env.game = game
        terminated = False
        while not terminated:
            obs   = env._get_obs()
            masks = env.action_masks()
            if game.current_player == 1:
                action, _ = model_b.predict(obs, action_masks=masks, deterministic=True)
            else:
                action, _ = model_a.predict(flip_obs(obs), action_masks=masks, deterministic=True)
            _, _, terminated, _, _ = game.step(decode_action(int(action)))
        r = _result_from_game(game, model_is_x=False)  # model_a is O
        if r == 1: wins += 1
        elif r == 0: draws += 1
        else: losses += 1

    return {"wins": wins, "draws": draws, "losses": losses,
            "total": 2 * num_games}

# ── ELO ───────────────────────────────────────────────────────────────────────

def expected_score(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def update_elo(ratings, label_a, label_b, wins_a, draws, losses_a, total):
    """
    Updates ELO ratings in-place based on matchup results.
    """
    ea = expected_score(ratings[label_a], ratings[label_b])
    eb = 1 - ea

    # Actual score for a: win=1, draw=0.5, loss=0
    score_a = (wins_a + 0.5 * draws) / total
    score_b = 1 - score_a

    ratings[label_a] += K_FACTOR * (score_a - ea)
    ratings[label_b] += K_FACTOR * (score_b - eb)

# ── Output ────────────────────────────────────────────────────────────────────

def print_table(rows, headers):
    col_widths = [max(len(str(r[i])) for r in [headers] + rows) for i in range(len(headers))]
    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    print(fmt.format(*headers))
    print("  ".join("-" * w for w in col_widths))
    for row in rows:
        print(fmt.format(*row))


def save_csv(filepath, rows, headers):
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    print(f"\nResults saved to {filepath}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Shared env — we'll swap env.game per game
    env = UTTTEnv()

    # Discover and load models
    model_list  = discover_models()
    loaded      = load_models(model_list)
    labels      = [label for label, _ in loaded]
    models      = {label: model for label, model in loaded}
    seed_label  = labels[0]

    # ── vs Random ─────────────────────────────────────────────────────────────
    print("=" * 60)
    print("Win rates vs Random Opponent")
    print("=" * 60)

    random_rows = []
    for label in labels:
        model = models[label]
        wx, dx, lx = play_vs_random(model, env, NUM_GAMES, model_is_x=True)
        wo, do, lo = play_vs_random(model, env, NUM_GAMES, model_is_x=False)

        total   = 2 * NUM_GAMES
        wins    = wx + wo
        draws   = dx + do
        losses  = lx + lo
        wr      = f"{wins / total:.1%}"
        random_rows.append([label, wx, dx, lx, wo, do, lo, wins, draws, losses, wr])

    random_headers = ["Model", "W(X)", "D(X)", "L(X)", "W(O)", "D(O)", "L(O)",
                      "W_tot", "D_tot", "L_tot", "WinRate"]
    print()
    print_table(random_rows, random_headers)

    # ── vs Seed ───────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"Win rates vs Seed Model ({seed_label})")
    print("=" * 60)

    seed_model  = models[seed_label]
    seed_rows   = []
    for label in labels[1:]:  # skip seed vs itself
        model   = models[label]
        result  = run_matchup(label, model, seed_label, seed_model, env, NUM_GAMES)
        total   = result["total"]
        wr      = f"{result['wins'] / total:.1%}"
        seed_rows.append([label, result["wins"], result["draws"], result["losses"], total, wr])

    seed_headers = ["Model", "Wins", "Draws", "Losses", "Total", "WinRate_vs_Seed"]
    print()
    print_table(seed_rows, seed_headers)

    # ── Round-robin ELO ───────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("Round-Robin ELO")
    print("=" * 60)

    ratings     = {label: float(INITIAL_ELO) for label in labels}
    matchup_log = []

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            la, lb  = labels[i], labels[j]
            result  = run_matchup(la, models[la], lb, models[lb], env, NUM_GAMES)
            update_elo(ratings, la, lb,
                       result["wins"], result["draws"], result["losses"], result["total"])
            matchup_log.append([la, lb, result["wins"], result["draws"],
                                 result["losses"], result["total"]])
            print(f"  {la} vs {lb}: {result['wins']}W {result['draws']}D {result['losses']}L")

    # Sort by ELO descending
    elo_rows = sorted([[label, f"{ratings[label]:.1f}"] for label in labels],
                      key=lambda x: -float(x[1]))
    print()
    print_table(elo_rows, ["Model", "ELO"])

    # ── Save CSV ──────────────────────────────────────────────────────────────
    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(["=== vs Random ==="])
        writer.writerow(random_headers)
        writer.writerows(random_rows)
        writer.writerow([])

        writer.writerow([f"=== vs Seed ({seed_label}) ==="])
        writer.writerow(seed_headers)
        writer.writerows(seed_rows)
        writer.writerow([])

        writer.writerow(["=== ELO Ratings ==="])
        writer.writerow(["Model", "ELO"])
        writer.writerows(elo_rows)
        writer.writerow([])

        writer.writerow(["=== Round-Robin Matchup Log ==="])
        writer.writerow(["Model_A", "Model_B", "A_Wins", "Draws", "A_Losses", "Total"])
        writer.writerows(matchup_log)

    print(f"\nResults saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()