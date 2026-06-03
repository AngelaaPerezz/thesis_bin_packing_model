from env import BPP
import numpy as np

env = BPP({'configs': [{'bin_size': [12,12], 'max_bin_size': [12,12], 'num_items': 13}]})
obs, _ = env.reset()

# Try to solve greedily: always pick first valid action
solved = 0
for trial in range(100):
    obs, _ = env.reset()
    for step in range(20):
        valid = np.where(env.action_mask > 0)[0]
        if len(valid) == 0:
            break
        action = valid[0] - 1  # first valid action
        obs, reward, done, _, _ = env.step(action)
        if done:
            if reward == 1.0:
                solved += 1
            break

print(f"Greedy solve rate: {solved}/100")