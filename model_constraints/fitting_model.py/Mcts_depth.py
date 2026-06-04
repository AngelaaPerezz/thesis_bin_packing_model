"""
mcts_depth.py
-------------
Patches RLlib's AlphaZero MCTS to enforce a maximum tree depth
(plies from root).  When a node is at depth >= depth_limit, its
children are not expanded and the node is treated as a terminal leaf
for the purposes of back-propagation (value = 0, or the current
network value estimate — your choice via LEAF_VALUE_AT_LIMIT).

Usage
-----
    from mcts_depth import build_depth_limited_mcts
    mcts = build_depth_limited_mcts(policy, depth_limit=4)
    action, v_values = mcts.compute_action(obs)
"""

from __future__ import annotations
import types
from typing import List

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_node_depth(node) -> int:
    """Walk parent pointers to compute depth from root."""
    depth = 0
    current = node
    while current.parent is not None:
        depth += 1
        current = current.parent
    return depth


# ---------------------------------------------------------------------------
# Core patch
# ---------------------------------------------------------------------------

def build_depth_limited_mcts(policy, depth_limit: int):
    """
    Return a patched MCTS object from *policy* that respects *depth_limit*.

    The MCTS object is retrieved from the policy and its `_simulate` method
    (or equivalent expansion step) is wrapped to refuse expansion beyond
    the depth cap.

    Parameters
    ----------
    policy      : RLlib TorchPolicy for an AlphaZero agent
    depth_limit : int, maximum number of plies from root to expand

    Returns
    -------
    mcts : the (patched) MCTS object
    """
    # RLlib AlphaZero stores the MCTS object on the policy
    mcts = policy.mcts

    original_simulate = mcts._simulate  # original recursive sim method

    def _simulate_limited(node, env, depth_limit_cap=depth_limit):
        """
        Wraps _simulate: at depth >= depth_limit_cap, return immediately
        with the node's value estimate (network prediction) rather than
        expanding children.
        """
        node_depth = _get_node_depth(node)
        if node_depth >= depth_limit_cap:
            # Leaf — return value without expanding
            if not node.is_terminal:
                # Use the stored prior value if available, else 0
                return getattr(node, 'value', 0.0)
            else:
                return node.reward
        return original_simulate(node, env)

    # Bind the patched method
    mcts._simulate = types.MethodType(
        lambda self, node, env: _simulate_limited(node, env),
        mcts
    )

    return mcts


# ---------------------------------------------------------------------------
# High-level: run one episode with depth limit, collect v-values
# ---------------------------------------------------------------------------

def run_episode_with_depth_limit(env, policy, depth_limit: int):
    """
    Run a single episode of *env* using *policy*'s MCTS with *depth_limit*.

    Returns
    -------
    dict with keys:
        obs_sequence    : list of flat obs arrays
        actions         : list of int actions taken
        v_values        : list of float value estimates (one per step)
        rewards         : list of float step rewards
        solved          : bool — all items placed
        coverage        : float — fraction of items placed
        num_steps       : int
    """
    mcts = build_depth_limited_mcts(policy, depth_limit)

    obs, _ = env.reset()  # NOTE: caller should pre-set env state if using stimulus
    # If env was pre-loaded via stimulus_to_env, reset() would randomise again.
    # Use env.set_state() / direct patching instead — see evaluate.py.

    done = False
    obs_seq  : List[np.ndarray] = []
    actions  : List[int]        = []
    v_values : List[float]      = []
    rewards  : List[float]      = []

    while not done:
        obs_seq.append(obs.copy())

        # Get action + value from MCTS
        action, _, extra = policy.compute_single_action(obs, explore=False)
        v_val = float(policy.model.value_function().detach().cpu().numpy().flatten()[0])
        v_values.append(v_val)

        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        actions.append(int(action))
        rewards.append(float(reward))

    return {
        "obs_sequence": obs_seq,
        "actions":      actions,
        "v_values":     v_values,
        "rewards":      rewards,
        "solved":       bool(env.num_placed == env.num_items),
        "coverage":     env.num_placed / env.num_items,
        "num_steps":    len(actions),
    }