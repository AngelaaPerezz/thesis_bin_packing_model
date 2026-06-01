import ray
from ray.rllib.algorithms import alpha_zero
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

from env import BPP
from model import Agent

register_env("Bpp-v1", BPP)
ModelCatalog.register_custom_model("bpp_model", Agent)


def train():

    ray.init(ignore_reinit_error=True)

    mcts_config = {
        "puct_coefficient": 1.0,
        "num_simulations": 100,
        "temperature": 1.5,
        "dirichlet_epsilon": 0.25,
        "dirichlet_noise": 0.03,
        "argmax_tree_policy": True,
        "add_dirichlet_noise": True,
    }

    env_config = {
        "bin_size": [13, 13],
        "max_bin_size": [13, 13],
        "num_items": 12,
        "max_items": 20,
    }

    ranked_rewards = {
        "enable": True,
        "percentile": 75,
        "buffer_max_length": 1000,
        "initialize_buffer": True,
        "num_init_rewards": 100,
    }

    config = (
        alpha_zero.AlphaZeroConfig()
        .environment(env="Bpp-v1", env_config=env_config)
        .framework("torch")
        .training(
            model={
                "custom_model": "bpp_model",
                "custom_model_config": {
                    "state_dim": 100,
                    "action_dim": 3,
                },
            },
            mcts_config=mcts_config,
            num_sgd_iter=10,
            ranked_rewards=ranked_rewards,
        )
        .rollouts(num_rollout_workers=2)
    )

    algo = config.build()

    for i in range(50):
        print(f"Iteration {i+1}")
        result = algo.train()
        print(result["episode_reward_mean"])

    path = algo.save()
    print("Saved to:", path)

    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    train()