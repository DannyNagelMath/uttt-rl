"""
plot_training.py -- Plot diagnostic curves from training_metrics.csv.

Generates training_metrics.png with four panels:
  1. Episode reward mean         -- is the policy improving?
  2. Explained variance          -- is the value function converging?
  3. Policy entropy              -- is the model collapsing to a fixed strategy?
  4. Win rate vs random          -- is general UTTT skill being retained?

Dashed vertical lines mark iteration boundaries.
Run after (or during) training:
    python plot_training.py
"""

import csv
import os
import sys

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
except ImportError:
    sys.exit("matplotlib is required: pip install matplotlib")

METRICS_FILE = "training_metrics.csv"
OUTPUT_FILE  = "training_metrics.png"


def load_csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def to_float(val):
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def main():
    if not os.path.exists(METRICS_FILE):
        sys.exit(f"{METRICS_FILE} not found -- run train_self_play.py first.")

    rows = load_csv(METRICS_FILE)
    if not rows:
        sys.exit("Metrics file is empty.")

    timesteps      = [int(r['timestep'])   for r in rows]
    iterations     = [int(r['iteration'])  for r in rows]
    ep_rew_mean    = [to_float(r['ep_rew_mean'])        for r in rows]
    explained_var  = [to_float(r['explained_variance']) for r in rows]
    entropy_loss   = [to_float(r['entropy_loss'])       for r in rows]

    # vs_random only logged at eval points -- scatter these separately
    vs_random_pts  = [
        (int(r['timestep']), to_float(r['vs_random_wr']))
        for r in rows if r.get('vs_random_wr')
    ]

    # Iteration boundary x-positions (where iteration number increments)
    boundaries = [
        timesteps[i]
        for i in range(1, len(iterations))
        if iterations[i] != iterations[i - 1]
    ]

    fig, axes = plt.subplots(4, 1, figsize=(13, 14), sharex=True)
    fig.suptitle('Self-Play Training Diagnostics', fontsize=13, y=0.995)

    def add_iter_lines(ax):
        for x in boundaries:
            ax.axvline(x, color='#888888', linestyle='--', linewidth=0.7, alpha=0.6)

    def plot_series(ax, ys, color, ylabel, title, hline=None):
        valid = [(t, y) for t, y in zip(timesteps, ys) if y is not None]
        if valid:
            ts, vals = zip(*valid)
            ax.plot(ts, vals, color=color, linewidth=1.0)
        if hline is not None:
            ax.axhline(hline, color=color, linestyle=':', linewidth=0.8, alpha=0.5)
        add_iter_lines(ax)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.25)

    # 1. Episode reward mean
    plot_series(axes[0], ep_rew_mean,
                color='steelblue',
                ylabel='reward',
                title='Episode Reward Mean  (rising = policy improving)')

    # 2. Explained variance
    plot_series(axes[1], explained_var,
                color='seagreen',
                ylabel='explained var',
                title='Explained Variance  (-> 1.0 = value function converged)',
                hline=1.0)
    axes[1].set_ylim(-1.1, 1.1)

    # 3. Policy entropy  (SB3 logs entropy_loss as negative; negate to show positive entropy)
    entropy_pos = [-v if v is not None else None for v in entropy_loss]
    plot_series(axes[2], entropy_pos,
                color='darkorange',
                ylabel='entropy',
                title='Policy Entropy  (collapsing early = overfitting / exploit)')

    # 4. Win rate vs random
    ax4 = axes[3]
    if vs_random_pts:
        ts_r, wr_r = zip(*vs_random_pts)
        ax4.plot(ts_r, wr_r, color='crimson', linewidth=1.0, alpha=0.7)
        ax4.scatter(ts_r, wr_r, color='crimson', s=18, zorder=5)
    ax4.axhline(0.55, color='gray', linestyle=':', linewidth=0.8,
                alpha=0.6, label='gate threshold (0.55)')
    add_iter_lines(ax4)
    ax4.set_ylabel('win rate', fontsize=9)
    ax4.set_title('Win Rate vs Random  (20-game stochastic eval every 100k steps)',
                  fontsize=10)
    ax4.set_ylim(0, 1.05)
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.25)

    axes[-1].set_xlabel('Timestep', fontsize=9)
    axes[-1].xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f'{x / 1e6:.1f}M')
    )

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches='tight')
    print(f"Saved to {OUTPUT_FILE}")
    plt.show()


if __name__ == "__main__":
    main()
