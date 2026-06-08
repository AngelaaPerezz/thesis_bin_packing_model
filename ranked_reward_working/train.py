import ray
from ray.rllib.algorithms import alpha_zero
from ray.tune.registry import register_env
import matplotlib.pyplot as plt
import os
from env import BPP
from model import Agent
import time 
# Check GPU visibility
import torch
import gc


register_env('Bpp-v1', BPP)


def train():
    ray.init(
        num_cpus=6,
        num_gpus=0,
        object_store_memory=512 * 1024 ** 2,  # cap object store at 2GB (default grabs too much)
        include_dashboard=False,             # dashboard sometimes causes startup issues on Windows
    )
    print("started")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
    print(f"Ray resources: {ray.available_resources()}")    

    mcts_config = {
        "puct_coefficient": 1.0,
        "num_simulations": 50,
        "temperature": 1.5,
        "dirichlet_epsilon": 0.25,
        "dirichlet_noise": 0.03,
        "argmax_tree_policy": True,
        "add_dirichlet_noise": True
    }

    env_config = {'bin_size': [10, 10], 'max_bin_size': [10, 10], 'num_items': 10}

    ranked_rewards = {
        "enable": True,
        "percentile": 75,
        "buffer_max_length": 1000,
        "initialize_buffer": True,
        "num_init_rewards": 10,
    }

    config = (
        alpha_zero.AlphaZeroConfig()
        .environment(env='Bpp-v1', env_config=env_config)
        .training(
            model={"custom_model": Agent},
            mcts_config=mcts_config,
            num_sgd_iter=1,
            ranked_rewards=ranked_rewards,
        )
        .rollouts(num_rollout_workers=1)
        .resources(num_gpus=0,
                   num_gpus_per_worker=0,)  
    )

    num_iterations = 10
    checkpoint_every = 20

    algo = config.build()
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    mean_rewards = []

    for i in range(num_iterations):
        start_time = time.time()
        results = algo.train()
        end_time = time.time()    
        gc.collect()                  # prevent memory accumulation

        iteration_time = end_time - start_time
        mean = results.get("episode_reward_mean", float('nan'))
        min_ = results.get("episode_reward_min", float('nan'))
        max_ = results.get("episode_reward_max", float('nan'))
        mean_rewards.append(mean)

        print(f"Iteration {i + 1:>4} | mean: {mean:>8.3f} | min: {min_:>8.3f} | max: {max_:>8.3f} | time: {iteration_time:.2f}s")

        if (i + 1) % checkpoint_every == 0:
            path = algo.save('checkpoints')
            print(f"  [checkpoint saved to {path}]")

        if (i + 1) % 50 == 0:
            plot_rewards(mean_rewards, i + 1)

    # Final plot if total iterations not a multiple of 50
    if num_iterations % 50 != 0:
        plot_rewards(mean_rewards, num_iterations)

    path = algo.save('checkpoints')
    print(f'Model saved to {path}')
    algo.stop()
    ray.shutdown()


def plot_rewards(mean_rewards, up_to_iter):
    plt.figure(figsize=(10, 4))
    plt.plot(range(1, len(mean_rewards) + 1), mean_rewards, marker='o', linewidth=1.5)
    plt.title(f'Mean Reward (iterations 1–{up_to_iter})')
    plt.xlabel('Iteration')
    plt.ylabel('Mean Reward')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'plots/reward_plot_iter_{up_to_iter}.png')
    print(f'  [plots/plot saved to reward_plot_iter_{up_to_iter}.png]')


if __name__ == '__main__':
    train()