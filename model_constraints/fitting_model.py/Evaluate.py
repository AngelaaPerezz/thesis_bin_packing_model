"""
evaluate.py
-----------
Run the trained AlphaZero agent on a fixed stimulus (puzzle) n times
with a given MCTS depth limit and collect per-trial metrics for fitting
to human data.

Two-step workflow
-----------------
Step 1 — compute the global replan threshold ONCE from the optimal agent:

    threshold = compute_global_threshold(
        all_stimuli     = ["s1.json", "s2.json", ...],
        checkpoint_path = "path/to/checkpoint",
        n_runs          = 200,
    )
    # save it, never touch it again

Step 2 — run any model (varying depth_limit etc.) using that fixed threshold:

    results = get_model_predictions(
        stimulus        = "s1.json",
        depth_limit     = 4,
        checkpoint_path = "path/to/checkpoint",
        replan_threshold = threshold,   # <-- pass in the frozen threshold
        n_runs          = 50,
    )

Each result dict has:
    solved          : bool   — all items placed successfully
    coverage        : float  — fraction of items placed  [0, 1]
    replan_events   : int    — # steps with value drop below global threshold
    order_size_corr : float  — Spearman r(placement order, item area)
                               positive = big items first

Stimulus JSON format
--------------------
{
    "binXLen": 12,  "binYLen": 12,  "gameType": "bp",
    "items": [
        {"xLen": 4, "yLen": 5, "color": "hsl(...)"},
        ...
    ]
}
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from scipy import stats
import torch

import ray
from ray.rllib.algorithms import alpha_zero
from ray.tune.registry import register_env

from env import BPP
from model import Agent
from stimulus import stimulus_to_env, load_stimulus
from mcts_depth import build_depth_limited_mcts


# ---------------------------------------------------------------------------
# Register env & model (idempotent)
# ---------------------------------------------------------------------------
register_env("Bpp-v1", BPP)

# Optimal agent config: no depth cap (use a very large number)
_NO_DEPTH_LIMIT = 9999

MCTS_CONFIG = {
    "puct_coefficient":    1.0,
    "num_simulations":     300,
    "temperature":         1.5,
    "dirichlet_epsilon":   0.25,
    "dirichlet_noise":     0.03,
    "argmax_tree_policy":  True,
    "add_dirichlet_noise": True,
}

RANKED_REWARDS_CONFIG = {
    "enable":             True,
    "percentile":         75,
    "buffer_max_length":  1000,
    "initialize_buffer":  True,
    "num_init_rewards":   50,
}


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class EpisodeResult:
    solved:          bool
    coverage:        float
    replan_events:   int
    order_size_corr: float
    # Raw traces — useful for debugging / richer analysis
    v_values:        List[float] = field(default_factory=list)
    actions:         List[int]   = field(default_factory=list)
    rewards:         List[float] = field(default_factory=list)
    placement_order: List[int]   = field(default_factory=list)
    item_areas:      List[int]   = field(default_factory=list)


# ---------------------------------------------------------------------------
# Build algo (shared helper)
# ---------------------------------------------------------------------------

def _build_algo(checkpoint_path: str, env_template: BPP, stim: dict):
    """Construct an AlphaZero algo, restore checkpoint, return (algo, policy)."""
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    bx, by    = stim["binXLen"], stim["binYLen"]
    num_items = len(stim["items"])

    env_config = {
        "bin_size":     [bx, by],
        "max_bin_size": [bx, by],
        "num_items":    num_items,
    }

    config = (
        alpha_zero.AlphaZeroConfig()
        .environment(
            env="Bpp-v1",
            env_config=env_config,
            disable_env_checking=True,
            observation_space=env_template.observation_space,
            action_space=env_template.action_space,
        )
        .training(
            model={"custom_model": Agent},
            mcts_config=MCTS_CONFIG,
            ranked_rewards=RANKED_REWARDS_CONFIG,
            train_batch_size=128,
            lr=5e-5,
        )
        .rollouts(num_rollout_workers=0)
    )

    algo = config.build()
    algo.restore(checkpoint_path)
    return algo, algo.get_policy()


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _value_drops(v_values: List[float]) -> List[float]:
    """
    Returns per-step value deltas: v[t+1] - v[t].
    Negative = value dropped (worsening); positive = improved.
    Convention is consistent with compute_global_threshold().
    """
    return [
        float(v_values[t + 1]) - float(v_values[t])
        for t in range(len(v_values) - 1)
    ]


def compute_replan_events(v_values: List[float], threshold: float) -> int:
    """
    Count steps where the value drop is below *threshold* (i.e. worse than
    the global baseline).  threshold should come from compute_global_threshold().

    Parameters
    ----------
    v_values  : per-step value estimates for one episode
    threshold : global 10th-percentile drop (a negative number)
    """
    if len(v_values) < 2:
        return 0
    drops = _value_drops(v_values)
    return int(sum(d < threshold for d in drops))


# ---------------------------------------------------------------------------
# Single-episode runner
# ---------------------------------------------------------------------------

def run_model(
    env_template: BPP,
    policy,
    depth_limit: int,
    replan_threshold: float,
) -> EpisodeResult:
    """
    Run one episode on a fresh copy of env_template.

    Parameters
    ----------
    env_template     : BPP env pre-loaded via stimulus_to_env()
    policy           : restored RLlib policy
    depth_limit      : MCTS ply cap from root
    replan_threshold : global threshold from compute_global_threshold()
    """
    env = copy.deepcopy(env_template)
    build_depth_limited_mcts(policy, depth_limit)
    obs = env._get_obs()

    done            = False
    v_values        : List[float] = []
    actions_taken   : List[int]   = []
    rewards         : List[float] = []
    placement_order : List[int]   = []
    prev_placed_ids : set         = set()

    while not done:
        # --- Value estimate (pre-action) ---
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            policy.model({"obs_flat": obs_tensor}, [], None)
            v_val = float(policy.model.value_function().squeeze().item())
        v_values.append(v_val)

        # --- MCTS-backed action ---
        action, _, _ = policy.compute_single_action(obs, explore=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        actions_taken.append(int(action))
        rewards.append(float(reward))

        # --- Track placement order ---
        current_placed_ids = {it[0] for it in env.items if it[3] != -1}
        newly_placed = current_placed_ids - prev_placed_ids
        placement_order.extend(sorted(newly_placed))
        prev_placed_ids = current_placed_ids

    # --- Metrics ---
    replan_events = compute_replan_events(v_values, replan_threshold)

    area_by_id    = {it[0]: it[1] * it[2] for it in env.items}
    ordered_areas = [area_by_id[iid] for iid in placement_order if iid in area_by_id]
    if len(ordered_areas) >= 2:
        r, _ = stats.spearmanr(range(len(ordered_areas)), ordered_areas)
        order_size_corr = float(r) if not np.isnan(r) else 0.0
    else:
        order_size_corr = 0.0

    return EpisodeResult(
        solved          = bool(env.num_placed == env.num_items),
        coverage        = env.num_placed / env.num_items,
        replan_events   = replan_events,
        order_size_corr = order_size_corr,
        v_values        = v_values,
        actions         = actions_taken,
        rewards         = rewards,
        placement_order = placement_order,
        item_areas      = [area_by_id[iid] for iid in placement_order if iid in area_by_id],
    )


# ---------------------------------------------------------------------------
# Step 1: compute global threshold from optimal agent
# ---------------------------------------------------------------------------

def compute_global_threshold(
    all_stimuli:     List,
    checkpoint_path: str,
    n_runs:          int   = 200,
    percentile:      float = 10.0,
    verbose:         bool  = True,
) -> float:
    """
    Run the *optimal* agent (no depth limit) across all stimuli and collect
    every per-step value drop into one pool.  Return the `percentile`-th
    percentile of that pool as the global replan threshold.

    This should be called ONCE before any model fitting and the result
    saved / frozen.  The same threshold is then passed to get_model_predictions()
    for every condition.

    Parameters
    ----------
    all_stimuli     : list of JSON paths or dicts
    checkpoint_path : path to the trained checkpoint
    n_runs          : episodes per stimulus (200 gives a stable estimate)
    percentile      : lower tail — default 10 (most negative 10% of drops)
    verbose         : print progress

    Returns
    -------
    threshold : float  (a negative number; drops below this = replan event)
    """
    all_drops: List[float] = []

    for stim_idx, stimulus in enumerate(all_stimuli):
        stim         = load_stimulus(stimulus)
        env_template = stimulus_to_env(stim)
        algo, policy = _build_algo(checkpoint_path, env_template, stim)

        bx, by    = stim["binXLen"], stim["binYLen"]
        num_items = len(stim["items"])
        if verbose:
            print(f"  Stimulus {stim_idx+1}/{len(all_stimuli)}: "
                  f"{bx}×{by}, {num_items} items — running {n_runs} episodes")

        for run_i in range(n_runs):
            env = copy.deepcopy(env_template)
            # Optimal agent: no depth cap
            build_depth_limited_mcts(policy, _NO_DEPTH_LIMIT)
            obs  = env._get_obs()
            done = False
            episode_v: List[float] = []

            while not done:
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    policy.model({"obs_flat": obs_tensor}, [], None)
                    v_val = float(policy.model.value_function().squeeze().item())
                episode_v.append(v_val)

                action, _, _ = policy.compute_single_action(obs, explore=True)
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

            all_drops.extend(_value_drops(episode_v))

            if verbose and (run_i + 1) % 50 == 0:
                print(f"    {run_i+1}/{n_runs} episodes done")

        algo.stop()

    threshold = float(np.percentile(all_drops, percentile))

    if verbose:
        print(f"\nGlobal threshold (p{percentile:.0f} of {len(all_drops)} drops): "
              f"{threshold:.4f}")
        print("Freeze this value and pass it to get_model_predictions().\n")

    return threshold


# ---------------------------------------------------------------------------
# Step 2: run model predictions with frozen threshold
# ---------------------------------------------------------------------------

def get_model_predictions(
    stimulus:         object,
    depth_limit:      int,
    checkpoint_path:  str,
    replan_threshold: float,
    n_runs:           int  = 50,
    verbose:          bool = True,
) -> List[dict]:
    """
    Run the agent on one stimulus with a fixed MCTS depth limit.

    Parameters
    ----------
    stimulus         : JSON path or dict
    depth_limit      : MCTS ply cap from root
    checkpoint_path  : path to the trained checkpoint
    replan_threshold : frozen global threshold from compute_global_threshold()
    n_runs           : simulated trials (50 matches typical human n)
    verbose          : print progress

    Returns
    -------
    List of n_runs dicts, each with:
        solved, coverage, replan_events, order_size_corr
    """
    stim         = load_stimulus(stimulus)
    env_template = stimulus_to_env(stim)
    algo, policy = _build_algo(checkpoint_path, env_template, stim)

    bx, by    = stim["binXLen"], stim["binYLen"]
    num_items = len(stim["items"])

    if verbose:
        print(f"Stimulus: {bx}×{by}, {num_items} items | "
              f"depth_limit={depth_limit} | threshold={replan_threshold:.4f}")
        print(f"Running {n_runs} trials...")

    results = []
    for run_i in range(n_runs):
        episode = run_model(env_template, policy, depth_limit, replan_threshold)
        results.append({
            "solved":          episode.solved,
            "coverage":        episode.coverage,
            "replan_events":   episode.replan_events,
            "order_size_corr": episode.order_size_corr,
        })

        if verbose:
            print(
                f"  run {run_i+1:>3}/{n_runs} | "
                f"solved={episode.solved} "
                f"coverage={episode.coverage:.2f} "
                f"replans={episode.replan_events} "
                f"corr={episode.order_size_corr:+.2f}"
            )

    algo.stop()
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    CHECKPOINT = r"C:\Users\sira1\ray_results\AlphaZero_Bpp-v1_2026-06-03_13-40-10wnmi65_m\checkpoint_000050"

    # --- Step 1: compute threshold once across all your stimuli ---
    all_stimuli = ["stimulus.json"]   # replace with your full list
    threshold   = compute_global_threshold(
        all_stimuli     = all_stimuli,
        checkpoint_path = CHECKPOINT,
        n_runs          = 200,
    )
    # In practice: save this to disk and reload instead of recomputing
    # e.g.  np.save("replan_threshold.npy", threshold)

    # --- Step 2: run a model condition ---
    stimulus_path = sys.argv[1] if len(sys.argv) > 1 else "stimulus.json"
    depth         = int(sys.argv[2]) if len(sys.argv) > 2 else 4

    results = get_model_predictions(
        stimulus         = stimulus_path,
        depth_limit      = depth,
        checkpoint_path  = CHECKPOINT,
        replan_threshold = threshold,
        n_runs           = 50,
    )

    print("\n--- Aggregate stats ---")
    print(f"  P(solved)       = {np.mean([r['solved'] for r in results]):.2f}")
    print(f"  Mean coverage   = {np.mean([r['coverage'] for r in results]):.3f}")
    print(f"  Mean replans    = {np.mean([r['replan_events'] for r in results]):.2f}")
    print(f"  Mean order corr = {np.mean([r['order_size_corr'] for r in results]):.3f}")