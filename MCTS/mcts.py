import copy
import json
import math
import numpy as np
from scipy import stats
from itertools import product
from collections import defaultdict
from env import BPP
import colorsys
import re
import time

# ─────────────────────────────────────────────
#  ENV HELPERS
# ─────────────────────────────────────────────

def make_env(max_bin_size, max_num_items):
    """
    Create env sized for the worst-case puzzle.
    bin_size is a placeholder here; load_stimulus sets it per puzzle.
    """
    config = {
        'bin_size':     max_bin_size,
        'max_bin_size': max_bin_size,
        'num_items':    max_num_items,
    }
    return BPP(config)

def hsl_to_rgb(hsl_str):
    # Parse "hsl(294,90%,60%)"
    h, s, l = map(float, re.findall(r'[\d.]+', hsl_str))
    h /= 360
    s /= 100
    l /= 100
    r, g, b = colorsys.hls_to_rgb(h, l, s)  # note: hls not hsl
    return (r, g, b)

def load_stimulus(env, stimulus):
    """
    Load a specific puzzle into the env.

    stimulus format (from JSON):
    {
        "binXLen": 12,
        "binYLen": 12,
        "items": [
            {"xLen": 3, "yLen": 8, "color": "..."},
            ...
        ]
    }
    """
    bw = stimulus['binXLen']
    bh = stimulus['binYLen']
    items = stimulus['items']

    env.bin_size    = [bw, bh]
    env.bin         = np.zeros(env.max_bin_size)
    env.num_items   = len(items)
    env.num_placed  = 0
    env.running_reward = 0

    env.items = [[i + 1, item['xLen'], item['yLen'], -1, -1]
                 for i, item in enumerate(items)]

    # For rendering — original items laid out sequentially
    env.initial_setting = [
        [0, 0, item['xLen'], item['yLen'], item['xLen'] * item['yLen']]
        for item in items
    ]

    # Pad to max_num_items so observation shapes stay consistent
    max_n = env.observation_space['obs']['states'].shape[0]
    while len(env.items) < max_n:
        env.items.append([0, 0, 0, 0, 0])
    env.item_colors = [hsl_to_rgb(item['color']) for item in items]
    while len(env.item_colors) < max_n:
        env.item_colors.append((0.8, 0.8, 0.8))  

    return env._get_obs()


# ─────────────────────────────────────────────
#  MCTS NODE
# ─────────────────────────────────────────────

class MCTSNode:
    def __init__(self, state, reward, done, parent=None, action=None):
        self.state       = state
        self.reward      = reward
        self.done        = done
        self.parent      = parent
        self.action      = action       # action_idx that led here
        self.children    = {}           # action_idx -> MCTSNode
        self.visit_count = 0
        self.value_sum   = 0.0
        self.valid_actions = None

    @property
    def value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def is_expanded(self):
        return self.valid_actions is not None

    def ucb_score(self, parent_visits, c=1.41):
        if self.visit_count == 0:
            return float('inf')
        return self.value + c * math.sqrt(math.log(parent_visits) / self.visit_count)


# ─────────────────────────────────────────────
#  MCTS SEARCH HELPERS
# ─────────────────────────────────────────────

def get_valid_action_indices(env, max_items=None):
    """
    Returns indices into env.actions that are currently valid.
    If max_items is set, restrict to the max_items largest unplaced items by area.
    """
    mask          = env.action_mask
    valid_indices = np.where(mask == 1)[0]

    if max_items is None or len(valid_indices) == 0:
        return valid_indices

    valid_actions = env.actions[valid_indices]   # (n, 3): item_id, x, y
    item_ids      = valid_actions[:, 0]

    unique_items = np.unique(item_ids)
    item_areas   = {iid: env.items[iid - 1][1] * env.items[iid - 1][2]
                    for iid in unique_items}

    sorted_items  = sorted(item_areas, key=lambda x: item_areas[x], reverse=True)
    allowed_items = set(sorted_items[:max_items])

    return valid_indices[np.isin(item_ids, list(allowed_items))]


def direct_step(env, action_idx):
    """
    Step the env using a direct index into env.actions.
    No gravity snap — anchor (x, y) is taken directly from the action.
    """
    action  = env.actions[action_idx]
    item_id = int(action[0])
    x       = int(action[1])
    y       = int(action[2])

    w, h = env.items[item_id - 1][1], env.items[item_id - 1][2]

    env.bin[x:x + w, y:y + h] = 1
    env.items[item_id - 1][3:5] = [x, y]
    env.num_placed += 1

    if env.num_placed == env.num_items:
        env.running_reward += 1
        return env._get_obs(), 1, True, {}

    obs = env._get_obs()
    if env.action_mask.sum() == 0:
        env.running_reward += -1
        return obs, -1, True, {}

    return obs, 0, False, {}


def random_rollout(env, state, max_items=None):
    """Play randomly from a given state. Returns final bin coverage."""
    env.set_state(state)
    done = False
    while not done:
        valid = get_valid_action_indices(env, max_items)
        if len(valid) == 0:
            break
        action_idx = np.random.choice(valid)
        _, _, done, _ = direct_step(env, action_idx)

    bw, bh = env.bin_size
    coverage = env.bin[:bw, :bh].sum() / (bw * bh)
    return coverage


# ─────────────────────────────────────────────
#  MCTS SEARCH
# ─────────────────────────────────────────────


def run_mcts_simulations(env, root, current_state, num_simulations, max_depth, max_items):
    """
    Run MCTS simulations from a given root node and state.
    Builds the local search tree via selection, expansion, rollout, backprop.
    """
    for _ in range(num_simulations):
        node = root
        env.set_state(current_state)
        depth = 0
 
        # ── Selection: traverse existing children via UCB ──
        while not node.done:
            if max_depth is not None and depth >= max_depth:
                break
            valid = get_valid_action_indices(env, max_items)
            # print(f"Valid actions at depth {depth}: {valid}")
            if len(valid) == 0:
                break
 
            visited   = [a for a in valid if a in node.children]
            unvisited = [a for a in valid if a not in node.children]
 
            if unvisited:
                # ── Expansion: pick a random unvisited action ──
                action_idx         = np.random.choice(unvisited)
                node.valid_actions = valid
                _, reward, done, _ = direct_step(env, action_idx)
                child_state        = env.get_state()
                child              = MCTSNode(state=child_state, reward=reward,
                                             done=done, parent=node, action=action_idx)
                node.children[action_idx] = child
                node  = child
                depth += 1
                break  # go to rollout
 
            else:
                # All actions visited — select best by UCB and go deeper
                action_idx = max(visited,
                                 key=lambda a: node.children[a].ucb_score(node.visit_count))
                node = node.children[action_idx]
                _, _, done, _ = direct_step(env, action_idx)
                depth += 1
                if done:
                    break
 
        # ── Rollout: play randomly to end from current node ──
        rollout_state = env.get_state()
        if node.done:
            bw, bh = env.bin_size
            value  = env.bin[:bw, :bh].sum() / (bw * bh)
        else:
            value = random_rollout(env, rollout_state, max_items)
 
        # ── Backprop: update all nodes on the path to root ──
        while node is not None:
            node.visit_count += 1
            node.value_sum   += value
            node = node.parent
 
 
def mcts_search(env, stimulus, num_simulations=200, max_depth=None, max_items=None):
    """
    Run MCTS on one puzzle stimulus.
    At each decision point, runs a fresh local search tree, picks the best
    action, places the item, then repeats until the game is done.
    This means max_depth controls lookahead per decision, not total placements.
    Returns placement_order, bin_coverage, success.
    """
    load_stimulus(env, stimulus)
    placement_order = []
    done            = False
 
    while not done:
        # Save current state as root for this decision
        current_state = env.get_state()
        valid         = get_valid_action_indices(env, max_items)
        # print(f"Valid actions: {valid}")
        if len(valid) == 0:
            break
 
        # Build a fresh local tree for this decision point
        root = MCTSNode(state=current_state, reward=0, done=False)
        run_mcts_simulations(env, root, current_state,
                             num_simulations, max_depth, max_items)
 
        # Restore state (simulations modified env)
        env.set_state(current_state)
 
        # Pick the most visited child as the best action
        if not root.children:
            # No children built (e.g. num_simulations too low) — fall back to random
            action_idx = np.random.choice(valid)
        else:
            action_idx = max(root.children,
                             key=lambda a: root.children[a].visit_count)
 
        # Execute the chosen action
        item_id = int(env.actions[action_idx][0])
        area    = env.items[item_id - 1][1] * env.items[item_id - 1][2]
        placement_order.append((item_id, area))
        _, _, done, _ = direct_step(env, action_idx)
 
    bw, bh       = env.bin_size
    bin_coverage = env.bin[:bw, :bh].sum() / (bw * bh)
    success      = env.num_placed == env.num_items
    env.render()
 
    return {
        'placement_order': placement_order,
        'bin_coverage':    bin_coverage,
        'success':         success,
    }

def size_order_correlation(placement_order):
    """
    Spearman correlation between placement order and item area.
    Positive = larger items placed first.
    """
    if len(placement_order) < 3:
        return np.nan
    areas = [p[1] for p in placement_order]
    order = list(range(len(areas)))
    corr, _ = stats.spearmanr(order, areas)
    return -corr   # negate: positive = big-first


def compute_trial_stats(result):
    return {
        'bin_coverage':    result['bin_coverage'],
        'size_order_corr': size_order_correlation(result['placement_order']),
        'success':         float(result['success']),
    }


def greedy_bfd(env, stimulus):
    load_stimulus(env, stimulus)
    placement_order = []
    done = False
    
    while not done:
        valid = get_valid_action_indices(env, max_items=None)
        if len(valid) == 0:
            break
        
        # Among valid actions, pick the one that:
        # 1. belongs to the largest unplaced item
        # 2. among those, maximizes coverage after placement
        valid_actions = env.actions[valid]
        
        # Get areas for all valid actions
        areas = np.array([env.items[int(a[0]) - 1][1] * env.items[int(a[0]) - 1][2]
                          for a in valid_actions])
        
        # Filter to largest item only
        max_area = areas.max()
        largest_item_indices = valid[areas == max_area]
        
        # Among those, pick position that maximizes coverage
        best_idx = None
        best_coverage = -1
        
        for idx in largest_item_indices:
            action = env.actions[idx]
            x, y = int(action[1]), int(action[2])
            w, h = env.items[int(action[0]) - 1][1], env.items[int(action[0]) - 1][2]
            
            # Simulate placement
            env.bin[x:x + w, y:y + h] = 1
            bw, bh = env.bin_size
            coverage = env.bin[:bw, :bh].sum()
            env.bin[x:x + w, y:y + h] = 0  # undo
            
            if coverage > best_coverage:
                best_coverage = coverage
                best_idx = idx
        
        item_id = int(env.actions[best_idx][0])
        area = env.items[item_id - 1][1] * env.items[item_id - 1][2]
        placement_order.append((item_id, area))
        _, _, done, _ = direct_step(env, best_idx)
    
    bw, bh = env.bin_size
    bin_coverage = env.bin[:bw, :bh].sum() / (bw * bh)
    success = env.num_placed == env.num_items
    
    return {
        'placement_order': placement_order,
        'bin_coverage': bin_coverage,
        'success': success,
    }

# ─────────────────────────────────────────────
#  SIMULATION DISTRIBUTION
# ─────────────────────────────────────────────

def simulate_stimulus(env, stimulus, num_simulations=200, max_depth=None,
                      max_items=None, num_runs=50):
    """
    Run MCTS num_runs times on a stimulus.
    Returns list of stat dicts.
    """
    results = []
    for _ in range(num_runs):
        result     = mcts_search(env, stimulus, num_simulations, max_depth, max_items)
        stats_dict = compute_trial_stats(result)
        results.append(stats_dict)
    return results


if __name__ == '__main__':
    with open('puzzles/001.json', 'r') as f:
        stimulus = json.load(f)

    env = make_env(max_bin_size=[13, 13], max_num_items=16)
    start_time = time.time()
    result = simulate_stimulus(env, stimulus, num_simulations=100,  num_runs=1, max_depth=4, max_items=13)
    end_time = time.time()
    print(f"Simulated 1 run in {end_time - start_time:.2f} seconds")
    print("Sample results: ", result)