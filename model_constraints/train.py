"""
Training Script — Paper-Faithful R² + AlphaZero
=================================================
Implements the training procedure from Section 5 exactly:

  • 50 problems generated and solved per iteration
  • Reward buffer of size 250 → threshold r_α at percentile α
  • 300 MCTS simulations per move
  • 50 steps of gradient descent per iteration  (num_sgd_iter=50)
  • Mini-batches of size 32
  • Last 500 games used for training replay buffer
  • Adam optimiser

Single-model strategy
---------------------
One Agent is trained across ALL configurations:
  bin sizes  : 12×12, 13×13
  num_items  : 12, 13, 14, 15, 16

RLlib's multi-env rollout workers are assigned env_configs round-robin so
the model sees all problem variants during training.  The fixed-size
observation tensors (MAX_ITEMS=16 rows, MAX_ACTIONS slots) are padded with
−1 for smaller instances, meaning the same network input/output shapes hold.

Ranked Reward (R²) — Equation (2) from the paper
-------------------------------------------------
  Raw MDP reward:
    r = 1.0                      if all items placed (perfect)
    r = num_placed / num_items   if stuck (partial credit, always > 0)

  R² reshaping:
    z = +1  if r > r_α  or  r == 1
    z = −1  if r < r_α
    z ~ Bernoulli(0.5)  if r == r_α and r < 1

RLlib's built-in ranked_rewards config handles this; the parameters below
reproduce the paper values.

Usage
-----
  python train.py [--iterations N] [--workers W] [--checkpoint-dir DIR]
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")          # headless backend — safe on remote servers
import matplotlib.pyplot as plt
import numpy as np
import ray
from ray.rllib.algorithms import alpha_zero
from ray.tune.registry import register_env

from env import BPP
from model import Agent


def save_reward_plot(
    iterations:   list,
    mean_rewards: list,
    min_rewards:  list,
    max_rewards:  list,
    checkpoint_dir: str,
    filename: str = "reward_curve.png",
) -> None:
    """
    Save a reward-vs-iteration figure to *checkpoint_dir/filename*.

    Plots:
      • mean episode reward (solid line)
      • min/max band (shaded)
      • vertical dashed lines at every checkpoint (multiples of 50)
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    iters = np.array(iterations)
    means = np.array(mean_rewards)
    mins  = np.array(min_rewards)
    maxs  = np.array(max_rewards)

    # Shaded min-max band
    ax.fill_between(iters, mins, maxs, alpha=0.15, color="#4C8BF5", label="min/max")

    # Mean reward line
    ax.plot(iters, means, color="#4C8BF5", linewidth=1.8, label="mean reward")

    # Horizontal reference lines
    ax.axhline(1.0,  color="green",  linewidth=0.8, linestyle="--", alpha=0.6, label="1.0 (all items placed)")
    ax.axhline(0.5,  color="grey",   linewidth=0.5, linestyle=":",  alpha=0.4, label="0.5 (reference)")

    # Vertical lines at checkpoint iterations
    for ckpt_iter in iters[iters % 50 == 0]:
        ax.axvline(ckpt_iter, color="orange", linewidth=0.6, linestyle="--", alpha=0.5)
    # Add a single legend entry for checkpoint markers
    ax.axvline(np.nan, color="orange", linewidth=0.6, linestyle="--", alpha=0.5,
               label="checkpoint")

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Episode reward", fontsize=12)
    ax.set_title("BPP Agent — Training Reward Curve (R² + AlphaZero)", fontsize=13)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(iters[0], iters[-1])
    ax.set_ylim(-0.05, 1.1)
    ax.grid(True, linewidth=0.4, alpha=0.5)

    fig.tight_layout()
    out_path = os.path.join(checkpoint_dir, filename)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ reward plot saved  → {out_path}")


# ── env configs to train on simultaneously ──────────────────────────────────
ENV_CONFIGS = [
    {"bin_size": [12, 12], "num_items": n}
    for n in range(12, 17)
] + [
    {"bin_size": [13, 13], "num_items": n}
    for n in range(12, 17)
]
# 10 distinct problem types; workers cycle through them automatically via
# worker_index → ENV_CONFIGS[worker_index % len(ENV_CONFIGS)]


def make_env(env_config):
    """
    Factory called by RLlib for every rollout worker.
    worker_index is injected into env_config by RLlib automatically.
    """
    worker_index = env_config.get("worker_index", 0)
    # Pick one config deterministically based on worker identity so each
    # worker specialises in one variant but all variants are covered.
    cfg = ENV_CONFIGS[worker_index % len(ENV_CONFIGS)]
    return BPP(cfg)


register_env("Bpp-v2", make_env)


def build_config(num_workers: int) -> alpha_zero.AlphaZeroConfig:
    # ── MCTS ─────────────────────────────────────────────────────────────────
    mcts_config = {
        "puct_coefficient":  1.0,
        "num_simulations":   300,        # paper: 300 simulations per move
        "temperature":       1.5,
        "dirichlet_epsilon": 0.25,
        "dirichlet_noise":   0.03,
        "argmax_tree_policy": True,
        "add_dirichlet_noise": True,
    }

    # ── Ranked Rewards (R²) — paper Section 4.1 ─────────────────────────────
    # buffer_max_length = 250  (paper: "reward buffer of size 250")
    # percentile = 75          (r_α = r_{75} in the paper's main experiment)
    # num_init_rewards = 100   (fill buffer before ranking kicks in)
    ranked_rewards = {
        "enable":             True,
        "percentile":         75,
        "buffer_max_length":  250,
        "initialize_buffer":  True,
        "num_init_rewards":   100,
    }

    # ── Use a representative env_config for space inference ─────────────────
    # The actual per-worker config is set inside make_env(); this is only used
    # by RLlib to instantiate observation/action spaces on the driver.
    representative_env_config = {"bin_size": [13, 13], "num_items": 16,
                                 "worker_index": 0}

    cfg = (
        alpha_zero.AlphaZeroConfig()
        .environment(env="Bpp-v2", env_config=representative_env_config,
                     disable_env_checking=True)
        .training(
            model={"custom_model": Agent},
            mcts_config=mcts_config,
            ranked_rewards=ranked_rewards,
            # Paper: 50 gradient steps per iteration, batch size 32
            num_sgd_iter=50,
            train_batch_size=32,
            # Adam optimiser
            lr=1e-3,
            # Replay buffer: last 500 games (paper Section 5).
            # replay_buffer_capacity was removed in newer RLlib;
            # capacity is now passed inside replay_buffer_config.
            replay_buffer_config={
                "type": "ReplayBuffer",  # AlphaZero uses a simple uniform buffer
                "capacity": 500,         # last 500 games (paper Section 5)
            },
        )
        .rollouts(
            num_rollout_workers=num_workers,
            # Paper: 50 problems per iteration; each worker generates ≥1
            rollout_fragment_length=1,  # one full episode per call
            batch_mode="complete_episodes",
        )
        .resources(num_gpus=0)
    )

    return cfg


def _print_weight_stats(algo) -> None:
    """
    Print a compact summary of every named parameter in the policy network:
      mean, std, and max-abs value.

    This lets you verify the network is actually being updated — if all
    values stay identical across iterations the optimizer is not running.
    Printed every 10 training iterations.
    """
    import torch
    try:
        # Get the state dict from the local worker's policy
        policy     = algo.get_policy()
        state_dict = policy.model.state_dict()

        print("  ── weight stats ─────────────────────────────────────────────")
        for name, tensor in state_dict.items():
            t = tensor.float()
            print(
                f"  {name:<45s}  "
                f"mean={t.mean().item():+.4f}  "
                f"std={t.std().item():.4f}  "
                f"max|w|={t.abs().max().item():.4f}"
            )
        print("  ─────────────────────────────────────────────────────────────")
    except Exception as e:
        print(f"  [weight stats unavailable: {e}]")


def train(num_iterations: int = 500,
          num_workers: int = 10,
          checkpoint_dir: str = "./checkpoints"):
    """
    Main training loop.

    num_iterations : number of training iterations (paper ran ~days on V100;
                     500 is a reasonable starting point to observe convergence)
    num_workers    : should be >= len(ENV_CONFIGS) = 10 so every variant is
                     covered in parallel; more workers = faster data collection
    checkpoint_dir : directory to save model checkpoints
    """
    ray.init(ignore_reinit_error=True)

    os.makedirs(checkpoint_dir, exist_ok=True)

    cfg  = build_config(num_workers)
    algo = cfg.build()

    print(f"Training for {num_iterations} iterations across "
          f"{len(ENV_CONFIGS)} environment variants.")
    variants = [
        f"{c['bin_size'][0]}×{c['bin_size'][1]} / {c['num_items']} items"
        for c in ENV_CONFIGS
    ]
    print(f"Variants: {variants}")
    # ── reward history (for plotting) ───────────────────────────────────────
    hist_iters:   list = []
    hist_mean:    list = []
    hist_min:     list = []
    hist_max:     list = []

    for i in range(1, num_iterations + 1):
        results = algo.train()

        ep_reward_mean = results.get("episode_reward_mean", float("nan"))
        ep_reward_min  = results.get("episode_reward_min",  float("nan"))
        ep_reward_max  = results.get("episode_reward_max",  float("nan"))
        ep_len_mean    = results.get("episode_len_mean",    float("nan"))

        hist_iters.append(i)
        hist_mean.append(ep_reward_mean)
        hist_min.append(ep_reward_min)
        hist_max.append(ep_reward_max)

        print(
            f"[{i:4d}/{num_iterations}]  "
            f"reward_mean={ep_reward_mean:+.3f}  "
            f"reward_min={ep_reward_min:+.3f}  "
            f"reward_max={ep_reward_max:+.3f}  "
            f"ep_len_mean={ep_len_mean:.1f}"
        )

        # ── weight snapshot (every 10 iterations) ───────────────────────────
        if i % 10 == 0:
            _print_weight_stats(algo)

        # Every 50 iterations: save checkpoint + update reward plot
        if i % 50 == 0:
            path = algo.save(checkpoint_dir)
            print(f"  ✓ checkpoint saved  → {path}")
            save_reward_plot(
                hist_iters, hist_mean, hist_min, hist_max,
                checkpoint_dir,
                filename=f"reward_curve_iter{i:04d}.png",
            )

    # ── final checkpoint + final plot ────────────────────────────────────────
    final_path = algo.save(checkpoint_dir)
    print(f"\nTraining complete.  Final model saved to: {final_path}")

    save_reward_plot(
        hist_iters, hist_mean, hist_min, hist_max,
        checkpoint_dir,
        filename="reward_curve_final.png",
    )

    algo.stop()
    ray.shutdown()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BPP agent with R² + AlphaZero")
    parser.add_argument("--iterations",      type=int, default=500,
                        help="Number of training iterations (default: 500)")
    parser.add_argument("--workers",         type=int, default=10,
                        help="Number of rollout workers (default: 10, one per env variant)")
    parser.add_argument("--checkpoint-dir",  type=str, default="./checkpoints",
                        help="Directory for model checkpoints (default: ./checkpoints)")
    args = parser.parse_args()

    train(
        num_iterations=args.iterations,
        num_workers=args.workers,
        checkpoint_dir=args.checkpoint_dir,
    )