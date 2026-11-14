"""Turn the DQN training logs into reward, episode length and steps plots.

Author: Larissa Iriza

One log:          uv run plot_results.py models/test/test_run_log.csv plots/test
All experiments:  uv run plot_results.py models/Larissa_Iriza plots/Larissa_Iriza
"""

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import numpy as np

WINDOW = 100  


def read_log(log_path):
    """Load a log CSV so each column can be used by its name."""
    return np.genfromtxt(log_path, delimiter=",", names=True, ndmin=1)


def moving_average(values, window=WINDOW):
    """Average of the last `window` values at each episode."""
    averages = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        averages.append(values[start : i + 1].mean())
    return np.array(averages)


def save_plot(out_dir, file_name, title, x_label, y_label):
    """Add the title and labels, then save the figure as a PNG."""
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / file_name, dpi=150)
    plt.close()
    print(f"Saved {out_dir / file_name}")


def plot_reward(log, out_dir):
    """Reward per episode with the 100-episode average on top."""
    plt.figure(figsize=(10, 5))
    plt.plot(log["episode"], log["reward"], color="tab:blue", alpha=0.3, label="Reward per episode")
    plt.plot(log["episode"], moving_average(log["reward"]), color="tab:blue", label=f"{WINDOW}-episode average")
    plt.legend()
    save_plot(out_dir, "reward.png", "Reward per Episode", "Episode", "Reward")


def plot_episode_length(log, out_dir):
    """Steps per episode. Longer Pong games mean more ball returns."""
    plt.figure(figsize=(10, 5))
    plt.plot(log["episode"], log["episode_length"], color="tab:green", alpha=0.3, label="Episode length")
    plt.plot(log["episode"], moving_average(log["episode_length"]), color="tab:green", label=f"{WINDOW}-episode average")
    plt.legend()
    save_plot(out_dir, "episode_length.png", "Episode Length", "Episode", "Steps in episode")


def plot_steps(log, out_dir):
    """Total training steps reached at each episode."""
    plt.figure(figsize=(10, 5))
    plt.plot(log["episode"], log["total_steps"], color="tab:orange")
    save_plot(out_dir, "steps.png", "Total Training Steps", "Episode", "Total steps")


def plot_single_log(log_path, out_dir):
    """Make the three plots for one training run."""
    log = read_log(log_path)
    plot_reward(log, out_dir)
    plot_episode_length(log, out_dir)
    plot_steps(log, out_dir)


def plot_comparison(member_dir, out_dir):
    """Average reward of every experiment in a member's folder on one chart."""
    log_paths = sorted(member_dir.glob("*_log.csv"))  
    if not log_paths:
        print(f"No *_log.csv files found in {member_dir}")
        return

    plt.figure(figsize=(10, 6))
    for log_path in log_paths:
        log = read_log(log_path)
        label = log_path.stem.replace("_log", "")  
        plt.plot(log["total_steps"], moving_average(log["reward"]), label=label)

    plt.legend(fontsize=8)
    title = f"{member_dir.name}: {WINDOW}-episode Average Reward per Experiment"
    save_plot(out_dir, "experiments_comparison.png", title, "Total steps", "Average reward")


def main():
    parser = argparse.ArgumentParser(description="Plot DQN training logs.")
    parser.add_argument("path", help="A log CSV file, or a member folder for the comparison chart")
    parser.add_argument("out_dir", help="Folder where the PNG files are saved")
    args = parser.parse_args()

    path = Path(args.path)
    out_dir = Path(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    if path.is_dir():
        plot_comparison(path, out_dir)
    else:
        plot_single_log(path, out_dir)


if __name__ == "__main__":
    main()
