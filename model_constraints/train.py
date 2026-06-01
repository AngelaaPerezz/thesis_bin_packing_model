import ray
from ray.rllib.algorithms import alpha_zero
from ray.tune.registry import register_env

from env import BPP
from model import Agent

register_env('Bpp-v1', BPP)


def train():
    ray.init(num_gpus=1)

    mcts_config = {
        "puct_coefficient": 1.0,
        "num_simulations": 50,
        "temperature": 1.5,
        "dirichlet_epsilon": 0.25,
        "dirichlet_noise": 0.03,
        "argmax_tree_policy": True,
        "add_dirichlet_noise": True
    }

    ranked_rewards = {
        "enable": True,
        "percentile": 75,
        "buffer_max_length": 1000,
        "initialize_buffer": True,
        "num_init_rewards": 100,
    }

    env_config = {
        'configs': [
            {'bin_size': [12, 12], 'max_bin_size': [12, 12], 'num_items': 13},
            {'bin_size': [12, 12], 'max_bin_size': [12, 12], 'num_items': 14},
            {'bin_size': [12, 12], 'max_bin_size': [12, 12], 'num_items': 15},
            {'bin_size': [12, 12], 'max_bin_size': [12, 12], 'num_items': 16},
            {'bin_size': [13, 13], 'max_bin_size': [13, 13], 'num_items': 13},
            {'bin_size': [13, 13], 'max_bin_size': [13, 13], 'num_items': 14},
            {'bin_size': [13, 13], 'max_bin_size': [13, 13], 'num_items': 15},
            {'bin_size': [13, 13], 'max_bin_size': [13, 13], 'num_items': 16},
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
            num_sgd_iter=10,
            ranked_rewards=ranked_rewards,
            train_batch_size=512,
            lr=1e-3
        )
        .rollouts(
            num_rollout_workers=4,
            rollout_fragment_length=32,
        )
    )

    num_iterations = 500

    algo = config.build()

    for i in range(num_iterations):
        print(f'Iteración {i + 1}/{num_iterations}')
        results = algo.train()
        reward_mean = results.get('episode_reward_mean', 'N/A')
        ep_len_mean = results.get('episode_len_mean', 'N/A')  # add this
        print(f'Iteración {i+1}: reward={reward_mean:.3f}, ep_len={ep_len_mean}')

    path = algo.save()
    print(f'Modelo guardado en {path}')
    algo.stop()
    ray.shutdown()


if __name__ == '__main__':
    train()