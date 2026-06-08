"""
Standalone environment checker — run this before train.py to diagnose issues.

Usage:
    python check_env.py

Runs three checks in order:
  1. Manual step-through: reset → 10 random valid actions → done
  2. ray.rllib.utils.check_env — the same checker RLlib runs on workers
  3. Observation / action space consistency checks

Any error here is the real root cause of the RolloutWorker crash.
"""

import traceback
import numpy as np
import gymnasium

# ── 1. Manual smoke-test ─────────────────────────────────────────────────────
print("=" * 60)
print("CHECK 1: manual reset + step loop")
print("=" * 60)

try:
    from env import BPP, MAX_ACTIONS

    for cfg in [
        {"bin_size": [12, 12], "num_items": 12},
        {"bin_size": [13, 13], "num_items": 16},
    ]:
        env = BPP(cfg)
        obs, _ = env.reset()

        print(f"\nConfig: {cfg}")
        print(f"  states  shape : {obs['obs']['states'].shape}")
        print(f"  actions shape : {obs['obs']['actions'].shape}")
        print(f"  mask    shape : {obs['action_mask'].shape}")
        print(f"  valid actions : {int(obs['action_mask'].sum())}")

        done = False
        steps = 0
        while not done:
            valid = np.where(obs["action_mask"] > 0)[0]
            if len(valid) == 0:
                print("  → no valid actions, episode should be terminal")
                break
            action = int(np.random.choice(valid))
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1

        print(f"  steps={steps}  reward={reward:.4f}  done={done}")

    print("\nCHECK 1 PASSED ✓")

except Exception:
    print("\nCHECK 1 FAILED ✗")
    traceback.print_exc()


# ── 2. RLlib check_env ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("CHECK 2: ray.rllib.utils.check_env")
print("=" * 60)

try:
    from ray.rllib.utils import check_env
    from env import BPP

    env = BPP({"bin_size": [13, 13], "num_items": 16})
    check_env(env)
    print("CHECK 2 PASSED ✓")

except Exception:
    print("\nCHECK 2 FAILED ✗  ← this is the real RolloutWorker error")
    traceback.print_exc()


# ── 3. Space dtype / bounds checks ──────────────────────────────────────────
print("\n" + "=" * 60)
print("CHECK 3: obs / action dtype and bounds")
print("=" * 60)

try:
    from env import BPP, MAX_ACTIONS, MAX_ITEMS

    env = BPP({"bin_size": [13, 13], "num_items": 16})
    obs, _ = env.reset()

    states  = obs["obs"]["states"]
    actions = obs["obs"]["actions"]
    mask    = obs["action_mask"]

    # dtypes
    assert states.dtype  == np.float32, f"states dtype {states.dtype} != float32"
    assert actions.dtype == np.float32, f"actions dtype {actions.dtype} != float32"
    assert mask.dtype    == np.int8,    f"mask dtype {mask.dtype} != int8"

    # shapes
    assert states.shape  == (MAX_ITEMS,   3),           f"states shape {states.shape}"
    assert actions.shape == (MAX_ACTIONS, 3),           f"actions shape {actions.shape}"
    assert mask.shape    == (MAX_ACTIONS,),             f"mask shape {mask.shape}"

    # observation_space containment
    assert env.observation_space.contains(obs), \
        "obs NOT contained in observation_space — check Box bounds"

    # action_space
    assert env.action_space.n == MAX_ACTIONS, \
        f"action_space.n={env.action_space.n} != MAX_ACTIONS={MAX_ACTIONS}"

    print("CHECK 3 PASSED ✓")

except Exception:
    print("\nCHECK 3 FAILED ✗")
    traceback.print_exc()

print("\n" + "=" * 60)
print("Done.  Fix any FAILED checks above before running train.py.")
print("=" * 60)