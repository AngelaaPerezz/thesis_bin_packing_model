from env import BPP
import numpy as np

env = BPP({'configs': [{'bin_size': [12,12], 'max_bin_size': [12,12], 'num_items': 13}]})
obs, _ = env.reset()

# Check corner points
cxs, cys = env._get_corner_points()
print(f'Corner xs at start: {cxs}')   # should be [0]
print(f'Corner ys at start: {cys}')   # should be [0]
print(f'Valid actions at start: {env.action_mask.sum():.0f}')  # should be 13 (one per item at (0,0))

# Random solve rate
solved = 0
for trial in range(500):
    obs, _ = env.reset()
    for step in range(25):
        valid = np.where(env.action_mask > 0)[0]
        if len(valid) == 0: break
        action = np.random.choice(valid) - 1
        obs, reward, done, _, _ = env.step(action)
        if done:
            if reward == 1.0: solved += 1
            break
print(f'Random corner solve rate: {solved}/500')