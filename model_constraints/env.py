"""
Bin Packing Problem (BPP) Environment — Free Placement, Gymnasium API
======================================================================
Key properties:
  - Uses `gymnasium` (Ray 2.3+ requires gymnasium, not gym).
  - Actions are (item_id, x, y) triples; items can be placed at ANY valid
    position — no gravity.
  - Supports variable bin sizes (12×12, 13×13) and variable item counts
    (12–16); fixed-size padded tensors let one model handle all variants.
  - Observation is the SET OF FEASIBLE ACTIONS, matching the paper arch.
  - reset() returns (obs, info); step() returns (obs, reward, terminated,
    truncated, info) — gymnasium-style.

Reward — paper Equation (1), 2-D case:
  rt = C*/C  if all items placed,  0 otherwise (stuck or non-terminal).
  C  = W + H            (bounding-box cost of minimal enclosing rectangle)
  C* = 2 * sqrt(A)      (cost of ideal square with area A = Σ wᵢhᵢ)

Observation space
-----------------
  obs / states  : (MAX_ITEMS,   3)  — [item_id, width, height]  (−1 if placed)
  obs / actions : (MAX_ACTIONS, 3)  — [item_id, x, y]           (−1 if invalid)
  action_mask   : (MAX_ACTIONS,)    — 1 = valid, 0 = invalid/padding
"""

import copy
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


# ── fixed tensor dimensions (upper bounds across all supported configs) ──────
MAX_BIN     = 13          # largest bin dimension (width or height)
MAX_ITEMS   = 16          # largest item count
MAX_ACTIONS = MAX_ITEMS * MAX_BIN * MAX_BIN   # worst-case action budget

# Observation value ranges
# item_id  : 1 … MAX_ITEMS  (or −1 for padding)
# width/height/x/y : 1 … MAX_BIN (or −1 for padding)
OBS_LOW  = -1.0
OBS_HIGH = float(MAX_ITEMS)   # item_id is the largest value that appears


class BPP(gym.Env):
    """2-D Bin Packing with free placement (no gravity). Gymnasium API."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, env_config: dict):
        super().__init__()

        # ── configuration ────────────────────────────────────────────────────
        self.bin_w: int     = env_config["bin_size"][0]
        self.bin_h: int     = env_config["bin_size"][1]
        self.num_items: int = env_config["num_items"]

        assert self.bin_w     <= MAX_BIN,   f"bin_w {self.bin_w} > MAX_BIN {MAX_BIN}"
        assert self.bin_h     <= MAX_BIN,   f"bin_h {self.bin_h} > MAX_BIN {MAX_BIN}"
        assert self.num_items <= MAX_ITEMS, f"num_items {self.num_items} > MAX_ITEMS {MAX_ITEMS}"

        # ── spaces ───────────────────────────────────────────────────────────
        self.observation_space = spaces.Dict({
            "obs": spaces.Dict({
                "states":  spaces.Box(OBS_LOW, OBS_HIGH,
                                      shape=(MAX_ITEMS, 3),   dtype=np.float32),
                "actions": spaces.Box(OBS_LOW, OBS_HIGH,
                                      shape=(MAX_ACTIONS, 3), dtype=np.float32),
            }),
            "action_mask": spaces.Box(0, 1,
                                      shape=(MAX_ACTIONS,),   dtype=np.int8),
        })

        self.action_space = spaces.Discrete(MAX_ACTIONS)

        # ── runtime state ────────────────────────────────────────────────────
        self.items:           list       = []
        self.bin:             np.ndarray = np.zeros((MAX_BIN, MAX_BIN), dtype=np.int8)
        self.num_placed:      int        = 0
        self.running_reward:  float      = 0.0
        self._actions_cache:  np.ndarray = np.full((MAX_ACTIONS, 3), -1, dtype=np.float32)
        self._mask_cache:     np.ndarray = np.zeros(MAX_ACTIONS, dtype=np.int8)
        self.initial_setting: list       = []
        self.colors = np.random.rand(MAX_ITEMS, 3)

        # Minimal spec for RLlib horizon handling
        class _Spec:
            def __init__(self, n):
                self.id = 0
                self.max_episode_steps = n
        self.spec = _Spec(self.num_items)

    # ── helpers ──────────────────────────────────────────────────────────────

    def _can_place(self, x: int, y: int, w: int, h: int) -> bool:
        if x + w > self.bin_w or y + h > self.bin_h:
            return False
        return not np.any(self.bin[x:x + w, y:y + h])

    def _compute_reward(self, all_placed: bool) -> float:
        """
        Reward function adapted for single-bin guillotine puzzles.

        The paper's C*/C collapses to a constant 1.0 for square bins with
        guillotine-generated items (items always tile the bin perfectly, so
        the bounding box is always the full bin). Instead we use:

            r = 1.0                        if all items placed (perfect)
            r = num_placed / num_items     if stuck (partial credit)

        This gives R² a meaningful distribution in (0, 1] to compute
        percentiles over, and provides gradient signal for partial progress.
        """
        if all_placed:
            return 1.0
        return self.num_placed / self.num_items

    def _build_obs(self) -> dict:
        # states
        states = np.full((MAX_ITEMS, 3), -1, dtype=np.float32)
        for i, item in enumerate(self.items):
            if item[3] == -1:                             # unplaced
                states[i] = [item[0], item[1], item[2]]  # id, w, h

        # actions
        actions = np.full((MAX_ACTIONS, 3), -1, dtype=np.float32)
        mask    = np.zeros(MAX_ACTIONS, dtype=np.int8)

        idx = 0
        for item in self.items:
            if item[3] != -1:                      # already placed
                idx += self.bin_w * self.bin_h     # skip its slot budget
                continue
            iid, w, h = item[0], item[1], item[2]
            for x in range(self.bin_w):
                for y in range(self.bin_h):
                    if idx >= MAX_ACTIONS:
                        break
                    if self._can_place(x, y, w, h):
                        actions[idx] = [iid, x, y]
                        mask[idx]    = 1
                    idx += 1
                if idx >= MAX_ACTIONS:
                    break

        self._actions_cache = actions
        self._mask_cache    = mask

        return {"obs": {"states": states, "actions": actions}, "action_mask": mask}

    # ── gymnasium API ────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        """Generate a new puzzle. Returns (obs, info) — gymnasium style."""
        super().reset(seed=seed)

        self.bin[:] = 0
        self.num_placed    = 0
        self.running_reward = 0.0
        self.items = []

        # Item generation: repeated random axis-aligned splits of the bin
        proto = [(0, 0, self.bin_w, self.bin_h)]

        def area(p):
            return (p[2] - p[0]) * (p[3] - p[1])

        while len(proto) < self.num_items:
            weights = np.array([area(p) for p in proto], dtype=float)
            weights /= weights.sum()
            idx    = np.random.choice(len(proto), p=weights)
            chosen = proto.pop(idx)
            x0, y0, x1, y1 = chosen

            axes       = list(np.random.permutation([0, 1]))
            split_done = False
            for axis in axes:
                if axis == 0 and x1 - x0 >= 2:
                    cut = np.random.randint(x0 + 1, x1)
                    proto.append((x0, y0, cut, y1))
                    proto.append((cut, y0, x1, y1))
                    split_done = True
                    break
                elif axis == 1 and y1 - y0 >= 2:
                    cut = np.random.randint(y0 + 1, y1)
                    proto.append((x0, y0, x1, cut))
                    proto.append((x0, cut, x1, y1))
                    split_done = True
                    break
            if not split_done:
                proto.append(chosen)

        self.initial_setting = list(proto)
        for i, (x0, y0, x1, y1) in enumerate(proto):
            self.items.append([i + 1, x1 - x0, y1 - y0, -1, -1])

        return self._build_obs(), {}

    def step(self, action: int):
        """
        Returns (obs, reward, terminated, truncated, info) — gymnasium style.
        truncated is always False (we never cut episodes short by time limit).
        """
        # Safety: if the chosen action is invalid, fall back to first valid one
        masked = (np.arange(MAX_ACTIONS) == action).astype(float) * self._mask_cache
        if masked.sum() == 0:
            valid = np.where(self._mask_cache > 0)[0]
            if len(valid) == 0:
                return self._build_obs(), 0.0, True, False, {}
            chosen_idx = valid[0]
        else:
            chosen_idx = int(masked.argmax())

        chosen = self._actions_cache[chosen_idx]
        iid, x, y = int(chosen[0]), int(chosen[1]), int(chosen[2])

        item   = self.items[iid - 1]
        w, h   = item[1], item[2]
        self.bin[x:x + w, y:y + h] = 1
        item[3] = x
        item[4] = y
        self.num_placed += 1

        # Terminal: all items placed → reward = 1.0
        if self.num_placed == self.num_items:
            obs    = self._build_obs()
            reward = self._compute_reward(all_placed=True)
            self.running_reward += reward
            return obs, reward, True, False, {}

        obs = self._build_obs()

        # Terminal: stuck before all items placed → partial credit
        if self._mask_cache.sum() == 0:
            reward = self._compute_reward(all_placed=False)
            self.running_reward += reward
            return obs, reward, True, False, {}

        return obs, 0.0, False, False, {}

    def render(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        for ax, title in zip([ax1, ax2], ["Original layout", "Agent placement"]):
            ax.set_xlim(0, self.bin_w);  ax.set_ylim(0, self.bin_h)
            ax.set_title(title);         ax.set_aspect("equal", "box")
            ax.set_xticks(range(self.bin_w + 1))
            ax.set_yticks(range(self.bin_h + 1))
            ax.grid(True, linewidth=0.4)

        for i, (x0, y0, x1, y1) in enumerate(self.initial_setting):
            ax1.add_patch(Rectangle((x0, y0), x1-x0, y1-y0,
                                    color=self.colors[i], alpha=0.8, ec="black"))
            ax1.text(x0+(x1-x0)/2, y0+(y1-y0)/2, str(i+1),
                     ha="center", va="center", fontsize=7)

        for i, item in enumerate(self.items):
            if item[3] != -1:
                ax2.add_patch(Rectangle((item[3], item[4]), item[1], item[2],
                                        color=self.colors[i], alpha=0.8, ec="black"))
                ax2.text(item[3]+item[1]/2, item[4]+item[2]/2, str(item[0]),
                         ha="center", va="center", fontsize=7)
        plt.tight_layout()
        plt.show()

    # ── state save/restore for MCTS ──────────────────────────────────────────

    def get_state(self):
        return (
            copy.deepcopy([self.items, self.bin.copy(),
                           self.num_placed, self.initial_setting]),
            self.running_reward,
        )

    def set_state(self, state):
        saved, reward        = state
        self.items           = copy.deepcopy(saved[0])
        self.bin[:]          = saved[1]
        self.num_placed      = saved[2]
        self.initial_setting = copy.deepcopy(saved[3])
        self.running_reward  = reward
        return self._build_obs()