import os
import ray
from ray.rllib.algorithms import alpha_zero
from ray.tune.registry import register_env
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # non-interactive backend, no tkinter needed

import numpy as np

from env import BPP
from model import Agent

register_env('Bpp-v1', BPP)

PLOT_DIR = 'plots'
os.makedirs(PLOT_DIR, exist_ok=True)


def save_plot(reward_means, reward_mins, reward_maxs, ep_lens, iteration):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    iters = np.arange(1, len(reward_means) + 1)

    ax1.plot(iters, reward_means, label='reward mean', color='blue')
    if any(r != 'N/A' for r in reward_mins):
        mins = [r for r in reward_mins if r != 'N/A']
        maxs = [r for r in reward_maxs if r != 'N/A']
        valid_iters = [i for i, r in zip(iters, reward_mins) if r != 'N/A']
        ax1.fill_between(valid_iters, mins, maxs, alpha=0.2, color='blue', label='reward min/max')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Reward')
    ax1.set_title(f'Reward (iteration {iteration})')
    ax1.legend()
    ax1.grid(True)
    ax1.set_ylim(-1.1, 1.1)

    ax2.plot(iters, ep_lens, label='ep_len mean', color='green')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Episode length')
    ax2.set_title('Episode length mean')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f'training_iter_{iteration:04d}.png')
    plt.savefig(path)
    plt.close()
    print(f'  Plot saved: {path}')


def train():
    ray.init(num_gpus=1)

    mcts_config = {
        "puct_coefficient": 1.0,
        "num_simulations": 100,
        "temperature": 1.5,
        "dirichlet_epsilon": 0.25,
        "dirichlet_noise": 0.03,
        "argmax_tree_policy": True,
        "add_dirichlet_noise": True,
    }

    # ranked_rewards = {
    #     "enable": True,
    #     "percentile": 75,
    #     "buffer_max_length": 250,
    #     "initialize_buffer": True,
    #     "num_init_rewards": 50,
    # }
    ranked_rewards = {
        "enable": False
    }


    env_config = {
        'configs': [
            {'bin_size': [12, 12], 'max_bin_size': [13, 13], 'num_items': 12},
        ]
    }

    _tmp_env = BPP(env_config)
    obs_space    = _tmp_env.observation_space
    action_space = _tmp_env.action_space
    _tmp_env.close()

    config = (
        alpha_zero.AlphaZeroConfig()
        .environment(
            env='Bpp-v1',
            env_config=env_config,
            disable_env_checking=True,
            observation_space=obs_space,
            action_space=action_space,
        )
        .training(
            model={"custom_model": Agent},
            mcts_config=mcts_config,
            num_sgd_iter=25,
            ranked_rewards=ranked_rewards,
            train_batch_size=64,
            lr=1e-4,
            grad_clip=0.5,
        )
        .rollouts(
            num_rollout_workers=4,
        )
    )

    num_iterations = 200

    algo = config.build()
    reward_means = []
    reward_mins  = []
    reward_maxs  = []
    ep_lens      = []

    for i in range(num_iterations):
        print(f'Iteración {i + 1}/{num_iterations}')
        results = algo.train()

        reward_mean = results.get('episode_reward_mean', 'N/A')
        reward_min  = results.get('episode_reward_min',  'N/A')
        reward_max  = results.get('episode_reward_max',  'N/A')
        ep_len_mean = results.get('episode_len_mean',    'N/A')

        reward_means.append(reward_mean)
        reward_mins.append(reward_min)
        reward_maxs.append(reward_max)
        ep_lens.append(ep_len_mean)

        policy = algo.get_policy()
        params = list(policy.model.parameters())
        weight_norm = sum(p.norm().item() for p in params)

        print(f'  reward={reward_mean:.3f} (min={reward_min:.3f}, max={reward_max:.3f})')
        print(f'  ep_len={ep_len_mean:.3f}')
        print(f'  weight_norm={weight_norm:.3f}')

        # checkpoint + plot every 50 iterations
        if (i + 1) % 50 == 0:
            path = algo.save()
            print(f'  Checkpoint: {path}')
            save_plot(reward_means, reward_mins, reward_maxs, ep_lens, i + 1)

    # final checkpoint + plot
    path = algo.save()
    print(f'Modelo guardado en {path}')
    save_plot(reward_means, reward_mins, reward_maxs, ep_lens, num_iterations)

    algo.stop()
    ray.shutdown()


if __name__ == '__main__':
    train()