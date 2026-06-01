import copy

import gym
from gym import spaces
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from typing import List, Tuple, Union


class MaskedBox(spaces.Box):
    """
    spaces.Box que además actúa como dict para que ranked_rewards.py
    pueda hacer obs["action_mask"] sobre la observación cruda.
    El vector aplanado tiene layout: [states(80) | actions(416) | action_mask(208)]
    """
    MAX_ITEMS   = 16
    MAX_BIN_DIM = 13
    DIM         = 2
    S = MAX_ITEMS * (2 * DIM + 1)           # 80
    A = MAX_ITEMS * MAX_BIN_DIM * 2         # 416
    M = MAX_ITEMS * MAX_BIN_DIM             # 208

    def __init__(self):
        total = self.S + self.A + self.M    # 704
        # low=-1 cubre padding; high=MAX_ITEMS cubre índices de item (1..16)
        # que son mayores que MAX_BIN_DIM
        super().__init__(low=-1, high=self.MAX_ITEMS, shape=(total,), dtype=np.float32)
        self.original_space = self


class ObsArray(np.ndarray):
    """
    np.ndarray que además soporta indexación por string,
    para que ranked_rewards.py pueda hacer obs["action_mask"].
    """
    MAX_ITEMS   = 16
    MAX_BIN_DIM = 13
    DIM         = 2
    _S = MAX_ITEMS * (2 * DIM + 1)      # 80
    _A = MAX_ITEMS * MAX_BIN_DIM * 2    # 416
    _M = MAX_ITEMS * MAX_BIN_DIM        # 208

    def __getitem__(self, key):
        if key == 'action_mask':
            start = self._S + self._A
            return np.asarray(self)[start : start + self._M]
        if key == 'states':
            return np.asarray(self)[:self._S].reshape(self.MAX_ITEMS, 2 * self.DIM + 1)
        if key == 'actions':
            return np.asarray(self)[self._S : self._S + self._A].reshape(self.MAX_ITEMS * self.MAX_BIN_DIM, 2)
        return super().__getitem__(key)


class BPP(gym.Env):
    MAX_ITEMS   = 16
    MAX_BIN_DIM = 13

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
        self.action_space      = spaces.Discrete(self.MAX_ITEMS * self.MAX_BIN_DIM)
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

        states = np.full((self.MAX_ITEMS, 2 * DIM + 1), -1, dtype=np.float32)
        states[:self.num_items] = np.array(self.items)

        items      = np.arange(1, self.num_items + 1)
        placements = np.arange(0, self.max_bin_size[0])

        actions_real = np.vstack((
            np.repeat(items, self.max_bin_size[0]),
            np.tile(placements, self.num_items)
        )).T.astype(np.float32)

        for action in actions_real:
            already_placed = self.items[int(action[0]) - 1][-1] != -1
            if already_placed:
                action[:] = [-1, -1]
            else:
                size = self.items[int(action[0]) - 1][1:3]
                a1   = int(action[1])
                block = self.bin[a1 : a1 + size[0], :]
                y     = np.where(block.sum(0) == 0)[0]
                if len(y) == 0:
                    action[:] = [-1, -1]
                else:
                    anchor = np.array([a1, int(y[0])])
                    block  = self.bin[anchor[0]:anchor[0]+size[0], anchor[1]:anchor[1]+size[1]]
                    if anchor[0]+size[0] > self.max_bin_size[0] or anchor[1]+size[1] > self.max_bin_size[1]:
                        action[:] = [-1, -1]
                    elif np.any(block):
                        action[:] = [-1, -1]

        actions = np.full((self.MAX_ITEMS * self.MAX_BIN_DIM, 2), -1, dtype=np.float32)
        actions[:len(actions_real)] = actions_real
        self.actions = actions

        action_mask = np.zeros(self.MAX_ITEMS * self.MAX_BIN_DIM, dtype=np.float32)
        real_valid  = np.ones(len(actions_real), dtype=np.float32)
        real_valid[np.where(actions_real[:, 0] == -1)[0]] = 0
        action_mask[:len(actions_real)] = real_valid
        self.action_mask = action_mask

        flat = np.concatenate([
            states.flatten(),
            actions.flatten(),
            action_mask,
        ])
        # Devolver ObsArray para que obs["action_mask"] funcione en ranked_rewards
        obs = flat.view(ObsArray)
        print(f"valid actions: {action_mask.sum():.0f}/{len(action_mask)}")

        return obs

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        cfg = self.configs[np.random.randint(len(self.configs))]
        self.bin_size     = cfg['bin_size']
        self.max_bin_size = cfg['max_bin_size']
        self.num_items    = cfg['num_items']

        self.items = []
        items = [[0] * self.dim + self.bin_size + [(self.bin_size[0] * self.bin_size[1])]]

        while len(items) < self.num_items:
            weights      = [item[4] * 100 for item in items]
            weight_sum   = sum(weights)
            weights      = [w / weight_sum for w in weights]
            random_item  = items.pop(np.random.choice(len(items), p=weights))
            random_axis  = np.random.choice(self.dim)

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
        action += 1
        action = self.actions[(action * self.action_mask).argmax()]

        size   = self.items[int(action[0]) - 1][1:3]
        block  = self.bin[int(action[1]) : int(action[1]) + size[0], :]
        anchor = np.array([int(action[1]), np.where(block.sum(0) == 0)[0][0]])

        block  = self.bin[anchor[0]:anchor[0]+size[0], anchor[1]:anchor[1]+size[1]]
        block[:, :] = 1

        self.items[int(action[0]) - 1][3:5] = anchor
        self.num_placed += 1

        if self.num_placed == self.num_items:
            self.running_reward += 1
            return self._get_obs(), 1, True, False, {}

        obs = self._get_obs()

        if self.action_mask.sum() == 0:
            self.running_reward += -1
            return obs, -1, True, False, {}

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
        plt.show()

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