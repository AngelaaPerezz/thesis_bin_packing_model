import copy
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import random


class BPP(gym.Env):

    def __init__(self, env_config):

        super().__init__()

        # -----------------------
        # CONFIG
        # -----------------------
        self.bin_size = env_config.get("bin_size", None)
        self.max_bin_size = tuple(env_config.get("max_bin_size", [13, 13]))
        self.num_items = env_config.get("num_items", 10)
        self.max_items = env_config.get("max_items", 20)

        self.dim = 2

        # -----------------------
        # STATE
        # -----------------------
        self.items = []
        self.bin = np.zeros(self.max_bin_size, dtype=np.int32)

        self.num_placed = 0
        self.running_reward = 0

        # -----------------------
        # ACTION SPACE (NO GRAVITY)
        # action = (item_id, x, y)
        # flattened for RLlib
        # -----------------------
        self.max_actions = self.max_items * self.max_bin_size[0] * self.max_bin_size[1]

        self.action_space = spaces.Discrete(self.max_actions)

        # -----------------------
        # OBSERVATION SPACE
        # -----------------------
        self.observation_space = spaces.Dict({
            "obs": spaces.Dict({
                "states": spaces.Box(
                    low=-1,
                    high=max(self.max_bin_size),
                    shape=(self.max_items, 5),
                    dtype=np.float32
                ),
                "actions": spaces.Box(
                    low=-1,
                    high=max(self.max_bin_size),
                    shape=(self.max_actions, 3),
                    dtype=np.int32
                ),
            }),
            "action_mask": spaces.Box(
                low=0,
                high=1,
                shape=(self.max_actions,),
                dtype=np.int8
            )
        })

        # -----------------------
        # RENDER COLORS
        # -----------------------
        self.colors = np.random.rand(self.max_items, 3)

        # -----------------------
        # RLLIB COMPAT (max episode)
        # -----------------------
        class Spec:
            def __init__(self, max_episode_steps):
                self.id = 0
                self.max_episode_steps = max_episode_steps

        self.spec = Spec(max_episode_steps=self.max_items)

    # =========================================================
    # OBSERVATION
    # =========================================================
    def _get_obs(self):

        # -----------------------
        # PAD STATES
        # -----------------------
        states = np.full((self.max_items, 5), -1, dtype=np.float32)

        for i, item in enumerate(self.items):
            states[i] = item

        # -----------------------
        # ACTION LIST (ALL POSSIBLE)
        # -----------------------
        actions = np.full((self.max_actions, 3), -1, dtype=np.int32)
        mask = np.zeros(self.max_actions, dtype=np.int8)

        idx = 0

        for item_id in range(len(self.items)):

            # skip placed items
            if self.items[item_id][3] != -1:
                continue

            w, h = self.items[item_id][1:3]

            for x in range(self.max_bin_size[0]):
                for y in range(self.max_bin_size[1]):

                    if idx >= self.max_actions:
                        break

                    if x + w > self.max_bin_size[0] or y + h > self.max_bin_size[1]:
                        idx += 1
                        continue

                    block = self.bin[x:x+w, y:y+h]

                    if np.any(block):
                        idx += 1
                        continue

                    actions[idx] = [item_id + 1, x, y]
                    mask[idx] = 1
                    idx += 1

        return {
            "obs": {
                "states": states,
                "actions": actions
            },
            "action_mask": mask
        }

    # =========================================================
    # RESET
    # =========================================================
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.items = []
        self.bin = np.zeros(self.max_bin_size, dtype=np.int32)
        self.num_placed = 0
        self.running_reward = 0

        self.bin_size = random.choices(
            [(12,12),(13,13)],
            weights=[0.7, 0.3]
        )[0]

        base = [0, 0, self.bin_size[0], self.bin_size[1]]
        items = [base]

        while len(items) < self.num_items:
            parent = items[np.random.randint(len(items))]
            axis = np.random.choice([0, 1])

            if parent[axis + 2] - parent[axis] <= 1:
                continue

            cut = np.random.randint(parent[axis] + 1, parent[axis + 2])

            left = parent.copy()
            right = parent.copy()

            left[axis + 2] = cut
            right[axis] = cut

            items.remove(parent)
            items.extend([left, right])

        self.items = []
        for i, it in enumerate(items):
            w = it[2] - it[0]
            h = it[3] - it[1]
            self.items.append([i + 1, w, h, -1, -1])

        return self._get_obs(), {}
    # =========================================================
    # STEP (NO GRAVITY)
    # =========================================================
    def step(self, action):
        action = int(action)

        decoded = self._get_obs()["obs"]["actions"][action]

        if decoded[0] == -1:
            return self._get_obs(), -1, True, False, {}

        item_id, x, y = decoded
        item_id -= 1

        w, h = self.items[item_id][1:3]

        self.bin[x:x+w, y:y+h] = 1
        self.items[item_id][3:5] = [x, y]
        self.num_placed += 1

        terminated = False
        truncated = False

        if self.num_placed == self.num_items:
            terminated = True
            reward = 1
        elif self._get_obs()["action_mask"].sum() == 0:
            terminated = True
            reward = -1
        else:
            reward = 0

        return self._get_obs(), reward, terminated, truncated, {}

    # =========================================================
    # RENDER
    # =========================================================
    def render(self, mode="human"):

        fig, ax = plt.subplots()

        ax.set_xlim(0, self.max_bin_size[0])
        ax.set_ylim(0, self.max_bin_size[1])
        ax.set_aspect("equal")

        for item in self.items:
            if item[3] != -1:
                ax.add_patch(Rectangle(
                    (item[3], item[4]),
                    item[1],
                    item[2],
                    color=self.colors[item[0] % self.max_items]
                ))

        plt.show()

    # =========================================================
    # STATE SAVE/LOAD (UNCHANGED COMPAT)
    # =========================================================
    def set_state(self, state):
        self.items = copy.deepcopy(state[0][0])
        self.bin = copy.deepcopy(state[0][1])
        self.num_placed = state[0][2]
        self.running_reward = state[1]
        return self._get_obs()

    def get_state(self):
        return copy.deepcopy([self.items, self.bin, self.num_placed]), self.running_reward