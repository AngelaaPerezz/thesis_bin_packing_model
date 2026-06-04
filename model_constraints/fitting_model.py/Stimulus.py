"""
stimulus.py
-----------
Convert a JSON puzzle (from the human experiment) into a BPP environment
with a fixed, deterministic item list — no random splitting.

JSON format expected:
{
    "binXLen": 12,
    "binYLen": 12,
    "gameType": "bp",
    "items": [
        {"xLen": 4, "yLen": 5, "color": "hsl(...)"},
        ...
    ]
}
"""

import json
import numpy as np
from env import BPP


def load_stimulus(path_or_dict) -> dict:
    """Load stimulus from a file path or an already-parsed dict."""
    if isinstance(path_or_dict, (str, bytes)):
        with open(path_or_dict) as f:
            return json.load(f)
    return path_or_dict


def stimulus_to_env(stimulus, rng_seed: int = 0) -> BPP:
    """
    Build a BPP environment whose item list is fixed to the stimulus items.

    The env's reset() normally does random splitting; here we bypass that
    by overriding the item list after reset and resetting internal state,
    so the agent always sees exactly the puzzle items.

    Returns a ready-to-go BPP env (already reset, _get_obs called).
    """
    bx = stimulus["binXLen"]
    by = stimulus["binYLen"]
    raw_items = stimulus["items"]
    num_items = len(raw_items)

    assert num_items <= BPP.MAX_ITEMS, (
        f"Stimulus has {num_items} items but MAX_ITEMS={BPP.MAX_ITEMS}"
    )
    assert bx <= BPP.MAX_BIN_DIM and by <= BPP.MAX_BIN_DIM, (
        f"Bin size [{bx},{by}] exceeds MAX_BIN_DIM={BPP.MAX_BIN_DIM}"
    )

    env_config = {
        "bin_size":     [bx, by],
        "max_bin_size": [bx, by],
        "num_items":    num_items,
    }

    env = BPP(env_config)

    # Seed numpy so colours are stable across runs (cosmetic only)
    np.random.seed(rng_seed)
    env.colors = np.random.rand(env.MAX_ITEMS, 3)

    # Build the fixed item list:
    # env.items format: [item_id, width, height, placed_x, placed_y]
    #   placed_x / placed_y = -1  means not yet placed
    fixed_items = []
    for i, it in enumerate(raw_items):
        fixed_items.append([i + 1, it["xLen"], it["yLen"], -1, -1])

    # Patch the env internals directly (bypass random reset)
    env.bin_size     = [bx, by]
    env.max_bin_size = [bx, by]
    env.num_items    = num_items
    env.items        = fixed_items
    env.bin          = np.zeros([bx, by], dtype=np.float32)
    env.num_placed   = 0
    env.running_reward = 0

    # Build initial_setting for render() compatibility:
    # format: [x0, y0, x1, y1]  (top-left / bottom-right corners)
    env.initial_setting = [
        [0, 0, it["xLen"], it["yLen"]] for it in raw_items
    ]

    # Compute first obs so env.actions / env.action_mask are populated
    env._get_obs()

    return env


def env_from_json(json_path_or_dict, rng_seed: int = 0) -> BPP:
    """Convenience wrapper: path/dict → ready BPP env."""
    stimulus = load_stimulus(json_path_or_dict)
    return stimulus_to_env(stimulus, rng_seed=rng_seed)