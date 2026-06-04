from Evaluate import get_model_predictions, compute_global_threshold

import numpy as np



results = get_model_predictions(
    stimulus        = "my_puzzle.json",   # or a parsed dict
    depth_limit     = 4,
    checkpoint_path = r"C:\Users\...\checkpoint_000050",
    n_runs          = 50,
)


checkpoint = r"C:\Users\sira1\ray_results\AlphaZero_Bpp-v1_2026-06-03_13-40-10wnmi65_m\checkpoint_000050"
# Every model condition:
threshold = float(np.load("replan_threshold.npy"))
depth_values = [2,3,4,5,8,12]

stimulus_paths = [f"puzzles/{d:03d}.json" for d in range(1, 21)]

threshold = compute_global_threshold(stimulus_paths, checkpoint, n_runs=200)
np.save("replan_threshold.npy", threshold)

for stimulus_path in stimulus_paths:
    print(f"Stimulus: {stimulus_path}")
    for depth in depth_values:
        print(f"  Depth limit: {depth}")
        results = get_model_predictions(
            stimulus        = stimulus_path,
            depth_limit     = depth,
            checkpoint_path = r"C:\Users\...\checkpoint_000050",
            n_runs          = 50,
            replan_threshold       = threshold,
        )
        print(results)