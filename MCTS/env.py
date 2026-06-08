import copy

import gym
from gym import spaces
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from typing import List, Tuple, Union


class BPP(gym.Env):
    def __init__(self, env_config):
        super().__init__()
        self.bin_size = env_config['bin_size']
        self.max_bin_size = env_config['max_bin_size']
        self.dim = len(self.bin_size)
        self.num_items = env_config['num_items']

        self.items = dict()
        self.bin = np.zeros(self.max_bin_size)

        # Action space: (item, x, y) — full 2D placement, no gravity snap
        self.observation_space = spaces.Dict(
            {
                'obs': spaces.Dict(
                        {
                            'states': spaces.Box(-1, self.max_bin_size[0],
                                                 shape=(self.num_items, 2 * self.dim + 1)),
                            'actions': spaces.Box(-1, self.max_bin_size[0],
                                                  shape=(self.num_items * self.max_bin_size[0] * self.max_bin_size[1], 3)),
                        }
                    ),

                'action_mask': spaces.Box(low=0, high=1, shape=(self.num_items * self.max_bin_size[0] * self.max_bin_size[1], )),
            }
        )

        self.action_space = spaces.Discrete(self.num_items * self.max_bin_size[0] * self.max_bin_size[1])
        self.running_reward = 0

        # Workaround for max_episode_steps
        class Spec:
            def __init__(self, max_episode_steps):
                self.id = getattr(env_config, 'worker_index', 0)
                self.max_episode_steps = max_episode_steps

        self.spec = Spec(max_episode_steps=self.num_items)


    def _is_contact_valid(self, x, y, size):
        """
        Check whether placing an item of given size at (x, y) is contact-valid:
        the item must touch at least one bin wall or an already-placed item on
        at least one of its four sides.
        """
        w, h = size  # width (x-axis), height (y-axis)
        bw, bh = self.bin_size

        # Touch left wall or item to the left
        if x == 0 or np.any(self.bin[x - 1, y:y + h]):
            return True
        # Touch right wall or item to the right
        if x + w == bw or np.any(self.bin[x + w, y:y + h]):
            return True
        # Touch bottom wall or item below
        if y == 0 or np.any(self.bin[x:x + w, y - 1]):
            return True
        # Touch top wall or item above
        if y + h == bh or np.any(self.bin[x:x + w, y + h]):
            return True

        return False

    def _get_obs(self):
        obs = dict()
        obs['states'] = np.array(self.items)

        bw, bh = self.bin_size
        max_bw, max_bh = self.max_bin_size

        # Enumerate all (item, x, y) combinations up to max_bin_size
        items_range    = np.arange(1, self.num_items + 1)
        x_range        = np.arange(0, max_bw)
        y_range        = np.arange(0, max_bh)

        ii, xx, yy = np.meshgrid(items_range, x_range, y_range, indexing='ij')
        actions = np.stack([ii.ravel(), xx.ravel(), yy.ravel()], axis=1)  # (N, 3)
        actions = actions.astype(int)

        action_mask = np.zeros(len(actions), dtype=np.float32)

        for idx, action in enumerate(actions):
            item_id, x, y = action
            item = self.items[item_id - 1]

            # Already placed
            if item[3] != -1:
                continue

            w, h = item[1], item[2]

            # Out of bounds for this puzzle's actual bin size
            if x + w > bw or y + h > bh:
                continue

            # Overlapping
            if np.any(self.bin[x:x + w, y:y + h]):
                continue

            # Must be contact-valid
            if not self._is_contact_valid(x, y, (w, h)):
                continue

            action_mask[idx] = 1

        # Mark invalid actions as [-1, -1, -1]
        actions[action_mask == 0] = [-1, -1, -1]

        obs['actions'] = actions
        self.actions = actions
        self.action_mask = action_mask

        observation = dict(obs=obs, action_mask=action_mask)
        return observation

    def reset(self):
        self.items = []
        items = [[0] * self.dim + self.bin_size + [(self.bin_size[0] * self.bin_size[1])]]

        while len(items) < self.num_items:
            weights = [item[4] * 100 for item in items]
            weight_sum = sum(weights)
            weights = [item / weight_sum for item in weights]

            random_item = items.pop(np.random.choice(len(items), p=weights))
            random_axis = np.random.choice(self.dim)

            while random_item[random_axis] + 1 == random_item[random_axis + self.dim]:
                items.append(random_item)
                random_item = items.pop(np.random.choice(len(items)))
                random_axis = np.random.choice(self.dim)

            random_pos = np.random.choice(np.arange(random_item[random_axis] + 1,
                                                    random_item[random_axis + self.dim]))

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

        self.bin = np.zeros(self.max_bin_size)
        self.num_placed = 0

        return self._get_obs()

    def step(self, action_idx):
        """Step using a direct index into self.actions (no gravity snap)."""
        action = self.actions[action_idx]
        item_id, x, y = int(action[0]), int(action[1]), int(action[2])

        size = self.items[item_id - 1][1:3]
        w, h = size

        self.bin[x:x + w, y:y + h] = 1
        self.items[item_id - 1][3:5] = [x, y]
        self.num_placed += 1

        if self.num_placed == self.num_items:
            self.running_reward += 1
            return self._get_obs(), 1, True, {}

        obs = self._get_obs()

        if self.action_mask.sum() == 0:
            self.running_reward += -1
            return obs, -1, True, {}

        return obs, 0, False, {}

    # def render(self, mode='human'):
    #     if self.dim != 2:
    #         raise NotImplementedError('Rendering only supported for 2D bins')

    #     fig, (ax1, ax2) = plt.subplots(1, 2)

    #     ax1.set_xlim(0, self.max_bin_size[0])
    #     ax1.set_ylim(0, self.max_bin_size[1])
    #     ax1.set_title('Original placement')
    #     ax1.set_aspect('equal', 'box')

    #     ax2.set_xlim(0, self.max_bin_size[0])
    #     ax2.set_ylim(0, self.max_bin_size[1])
    #     ax2.set_title('Achieved placement')
    #     ax2.set_aspect('equal', 'box')

    #     for i, item in enumerate(self.initial_setting):
    #         ax1.add_patch(Rectangle((item[0], item[1]), item[2] - item[0], item[3] - item[1],
    #                       color=self.item_colors[i]))

    #     for i, item in enumerate(self.items):
    #         if item[3] != -1:
    #             ax2.add_patch(Rectangle((item[3], item[4]), item[1], item[2], color=self.item_colors[i]))

    #     plt.show()

    def render(self, mode='human'):
        if self.dim != 2:
            raise NotImplementedError('Rendering only supported for 2D bins')

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # ── Left panel: items laid out shelf-style ──
        shelf_x, shelf_y, row_height = 0, 0, 0
        display_positions = []

        for item in self.items:
            w, h = item[1], item[2]
            if shelf_x + w > 8:
                shelf_x = 0
                shelf_y += row_height
                row_height = 0
            display_positions.append((shelf_x, shelf_y, w, h))
            shelf_x += w
            row_height = max(row_height, h)

        total_height = shelf_y + row_height

        ax1.set_xlim(0, 8)
        ax1.set_ylim(0, max(total_height, self.bin_size[1]))
        ax1.set_title('Items')
        ax1.set_aspect('equal', 'box')

        for i, (x, y, w, h) in enumerate(display_positions):
            ax1.add_patch(Rectangle((x, y), w, h,
                        color=self.item_colors[i], edgecolor='white', linewidth=0.5))

        # ── Right panel: achieved placement ──
        ax2.set_xlim(0, self.bin_size[0])
        ax2.set_ylim(0, self.bin_size[1])
        ax2.set_title('Achieved placement')
        ax2.set_aspect('equal', 'box')

        for i, item in enumerate(self.items):
            if item[3] != -1:
                ax2.add_patch(Rectangle((item[3], item[4]), item[1], item[2],
                            color=self.item_colors[i], edgecolor='white', linewidth=0.5))

        plt.tight_layout()
        plt.show()

    def set_state(self, state):
        self.items      = copy.deepcopy(state[0][0])
        self.bin        = copy.deepcopy(state[0][1])
        self.num_placed = state[0][2]
        self.bin_size   = state[0][3]          # per-puzzle bin size
        self.running_reward = state[1]
        return self._get_obs()

    def get_state(self):
        return copy.deepcopy([self.items, self.bin, self.num_placed, self.bin_size]), self.running_reward