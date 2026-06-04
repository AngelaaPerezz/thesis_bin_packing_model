import copy

import gym
from gym import spaces
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


class MaskedBox(spaces.Box):
    """
    spaces.Box that also supports dict indexing for ranked_rewards.py.
    Flat vector layout: [states(80) | actions(3*2704) | action_mask(2704)]
    Actions are (item_id, x, y) triples — free placement, no gravity.
    """
    MAX_ITEMS   = 16
    MAX_BIN_DIM = 13
    DIM         = 2
    N_ACTIONS   = MAX_ITEMS * MAX_BIN_DIM * MAX_BIN_DIM  # 2704
    S = MAX_ITEMS * (2 * DIM + 1)   # 80  — states
    A = N_ACTIONS * 3               # 8112 — actions (item, x, y)
    M = N_ACTIONS                   # 2704 — action mask

    def __init__(self):
        total = self.S + self.A + self.M   # 10896
        super().__init__(low=-1, high=self.MAX_ITEMS, shape=(total,), dtype=np.float32)
        self.original_space = spaces.Box(
            low=-1, high=self.MAX_ITEMS, shape=(total,), dtype=np.float32
        )


class ObsArray(np.ndarray):
    """np.ndarray that supports string indexing for ranked_rewards.py."""
    MAX_ITEMS   = 16
    MAX_BIN_DIM = 13
    DIM         = 2
    N_ACTIONS   = MAX_ITEMS * MAX_BIN_DIM * MAX_BIN_DIM  # 2704
    _S = MAX_ITEMS * (2 * DIM + 1)  # 80
    _A = N_ACTIONS * 3              # 8112
    _M = N_ACTIONS                  # 2704

    def __getitem__(self, key):
        if key == 'action_mask':
            start = self._S + self._A
            return np.asarray(self)[start : start + self._M]
        if key == 'states':
            return np.asarray(self)[:self._S].reshape(self.MAX_ITEMS, 2 * self.DIM + 1)
        if key == 'actions':
            return np.asarray(self)[self._S : self._S + self._A].reshape(self.N_ACTIONS, 3)
        return super().__getitem__(key)


class BPP(gym.Env):
    MAX_ITEMS   = 16
    MAX_BIN_DIM = 13
    N_ACTIONS   = MAX_ITEMS * MAX_BIN_DIM * MAX_BIN_DIM  # 2704

    def __init__(self, env_config):
        super().__init__()
        if 'configs' in env_config:
            self.configs = env_config['configs']
        else:
            self.configs = [{
                'bin_size': env_config['bin_size'],
                'max_bin_size': env_config['max_bin_size'],
                'num_items': env_config['num_items'],
            }]

        cfg = self.configs[0]
        self.bin_size     = cfg['bin_size']
        self.max_bin_size = cfg['max_bin_size']
        self.dim          = len(self.bin_size)
        self.num_items    = cfg['num_items']

        self.items = []
        self.bin   = np.zeros(self.max_bin_size)

        self.observation_space = MaskedBox()
        self.action_space      = spaces.Discrete(self.N_ACTIONS)
        self.running_reward    = 0

        class Spec:
            def __init__(self, max_episode_steps):
                self.id = getattr(env_config, 'worker_index', 0)
                self.max_episode_steps = max_episode_steps

        self.spec   = Spec(max_episode_steps=self.MAX_ITEMS)
        self.colors = np.random.rand(self.MAX_ITEMS, 3)

    # ------------------------------------------------------------------
    def _get_obs(self):
        DIM = self.dim

        # States
        states = np.full((self.MAX_ITEMS, 2 * DIM + 1), -1, dtype=np.float32)
        states[:self.num_items] = np.array(self.items)

        # Build all (item, x, y) triples vectorized
        items = np.arange(1, self.num_items + 1)
        xs    = np.arange(0, self.max_bin_size[0])
        ys    = np.arange(0, self.max_bin_size[1])
        ii, xx, yy = np.meshgrid(items, xs, ys, indexing='ij')
        actions_real = np.stack([ii.ravel(), xx.ravel(), yy.ravel()], axis=1).astype(np.float32)

        n = len(actions_real)
        valid = np.ones(n, dtype=bool)

        # 1. Mask already-placed items (vectorized)
        placed = np.array([self.items[i][-1] != -1 for i in range(self.num_items)])
        # item ids are 1-indexed; actions_real[:,0] gives item id
        item_ids = actions_real[:, 0].astype(int) - 1
        valid[placed[item_ids]] = False

        # 2. For remaining actions, check bounds and overlap vectorized
        still_valid = np.where(valid)[0]
        if len(still_valid) > 0:
            ids = actions_real[still_valid, 0].astype(int) - 1
            xs_ = actions_real[still_valid, 1].astype(int)
            ys_ = actions_real[still_valid, 2].astype(int)
            ws  = np.array([self.items[i][1] for i in ids])
            hs  = np.array([self.items[i][2] for i in ids])

            # Out of bounds check
            oob = (xs_ + ws > self.max_bin_size[0]) | (ys_ + hs > self.max_bin_size[1])
            valid[still_valid[oob]] = False

            # Overlap check using cumsum for fast rectangular queries
            in_bounds = still_valid[~oob]
            if len(in_bounds) > 0:
                ids2 = actions_real[in_bounds, 0].astype(int) - 1
                xs2  = actions_real[in_bounds, 1].astype(int)
                ys2  = actions_real[in_bounds, 2].astype(int)
                ws2  = np.array([self.items[i][1] for i in ids2])
                hs2  = np.array([self.items[i][2] for i in ids2])

                # Use 2D prefix sum for O(1) rectangle sum queries
                prefix = np.pad(self.bin.cumsum(0).cumsum(1), ((1,0),(1,0)))
                x1, y1 = xs2, ys2
                x2, y2 = xs2 + ws2, ys2 + hs2
                rect_sums = (prefix[x2, y2] - prefix[x1, y2]
                             - prefix[x2, y1] + prefix[x1, y1])
                valid[in_bounds[rect_sums > 0]] = False

        # Mark invalid actions as -1
        actions_real[~valid] = -1

        # Pad to N_ACTIONS
        actions = np.full((self.N_ACTIONS, 3), -1, dtype=np.float32)
        actions[:n] = actions_real
        self.actions = actions

        # Action mask
        action_mask = np.zeros(self.N_ACTIONS, dtype=np.float32)
        action_mask[:n] = valid.astype(np.float32)
        self.action_mask = action_mask

        flat = np.concatenate([states.flatten(), actions.flatten(), action_mask])
        return flat.view(ObsArray)

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        cfg = self.configs[np.random.randint(len(self.configs))]
        self.bin_size     = cfg['bin_size']
        self.max_bin_size = cfg['max_bin_size']
        self.num_items    = cfg['num_items']

        self.items = []
        items = [[0] * self.dim + self.bin_size + [(self.bin_size[0] * self.bin_size[1])]]

        while len(items) < self.num_items:
            weights     = [item[4] * 100 for item in items]
            weight_sum  = sum(weights)
            weights     = [w / weight_sum for w in weights]
            random_item = items.pop(np.random.choice(len(items), p=weights))
            random_axis = np.random.choice(self.dim)

            while random_item[random_axis] + 1 == random_item[random_axis + self.dim]:
                items.append(random_item)
                random_item = items.pop(np.random.choice(len(items)))
                random_axis = np.random.choice(self.dim)

            random_pos = np.random.choice(
                np.arange(random_item[random_axis] + 1, random_item[random_axis + self.dim])
            )
            item1 = copy.deepcopy(random_item)
            item1[random_axis + self.dim] = random_pos
            item1[4] = (item1[2] - item1[0]) * (item1[3] - item1[1])

            item2 = copy.deepcopy(random_item)
            item2[random_axis] = random_pos
            item2[4] = (item2[2] - item2[0]) * (item2[3] - item2[1])

            items.extend([item1, item2])

        self.initial_setting = items

        for i, item in enumerate(items):
            self.items.append([i + 1, item[2] - item[0], item[3] - item[1], -1, -1])

        self.bin        = np.zeros(self.max_bin_size)
        self.num_placed = 0

        return self._get_obs(), {}

    # ------------------------------------------------------------------
    def step(self, action):
        # If the chosen action is somehow invalid, fall back to first valid one
        if self.action_mask[action] == 0:
            action = int(self.action_mask.argmax())

        chosen   = self.actions[action]
        item_idx = int(chosen[0]) - 1
        x, y     = int(chosen[1]), int(chosen[2])
        size     = self.items[item_idx][1:3]

        # Place item at exact (x, y) — free placement, no gravity
        self.bin[x : x + size[0], y : y + size[1]] = 1
        self.items[item_idx][3:5] = [x, y]
        self.num_placed += 1

        if self.num_placed == self.num_items:
            return self._get_obs(), 1.0, True, False, {}

        obs = self._get_obs()

        if self.action_mask.sum() == 0:
            reward = self.num_placed / self.num_items
            return obs, reward, True, False, {}

        return obs, 0, False, False, {}
    # ------------------------------------------------------------------
    def render(self, mode='human'):
        if self.dim != 2:
            raise NotImplementedError('Rendering only supported for 2D bins')

        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.set_xlim(0, self.max_bin_size[0]); ax1.set_ylim(0, self.max_bin_size[1])
        ax1.set_title('Original placement');   ax1.set_aspect('equal', 'box')
        ax2.set_xlim(0, self.max_bin_size[0]); ax2.set_ylim(0, self.max_bin_size[1])
        ax2.set_title('Achieved placement');   ax2.set_aspect('equal', 'box')

        for i, item in enumerate(self.initial_setting):
            ax1.add_patch(Rectangle((item[0], item[1]), item[2]-item[0], item[3]-item[1],
                          color=self.colors[i]))
        for i, item in enumerate(self.items):
            if item[3] != -1:
                ax2.add_patch(Rectangle((item[3], item[4]), item[1], item[2],
                              color=self.colors[i]))
        plt.savefig('render.png')
        plt.close()

    def set_state(self, state):
        self.items        = copy.deepcopy(state[0][0])
        self.bin          = copy.deepcopy(state[0][1])
        self.num_placed   = state[0][2]
        self.num_items    = state[0][3]
        self.bin_size     = state[0][4]
        self.max_bin_size = state[0][5]
        self.running_reward = state[1]
        return self._get_obs()

    def get_state(self):
        return copy.deepcopy([
            self.items, self.bin, self.num_placed,
            self.num_items, self.bin_size, self.max_bin_size,
        ]), self.running_reward