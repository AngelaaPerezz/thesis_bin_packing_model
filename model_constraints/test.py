import argparse

import ray
from ray.rllib.algorithms import alpha_zero
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.policy.policy_map import PolicyMap
from ray.rllib.evaluation.episode import Episode
from ray.tune.registry import register_env

from env import BPP
from model import Agent

register_env('Bpp-v1', BPP)


def evaluate():
    parser = argparse.ArgumentParser(description='Testing script for ranked reward on bin packing problem')
    parser.add_argument('--checkpoint-path', type=str, help='Checkpoint path for trained policy', required=True)
    args = parser.parse_args()

    ray.init(num_gpus=1)

    mcts_config = {
                "puct_coefficient": 1.0,
                "num_simulations": 300,
                "temperature": 1.5,
                "dirichlet_epsilon": 0.25,
                "dirichlet_noise": 0.03,
                "argmax_tree_policy": False,
                "add_dirichlet_noise": True,
                "max_depth": 5,  # 

            }

    env_config = {'bin_size': [10, 10], 'max_bin_size': [10, 10], 'num_items': 10}

    ranked_rewards = {
        "enable": True,
        "percentile": 75,
        "buffer_max_length": 1000,
        "initialize_buffer": True,
        "num_init_rewards": 100,
    }

    config = (
                alpha_zero.AlphaZeroConfig()
                .environment(env='Bpp-v1', env_config=env_config)
                .training(model={"custom_model": Agent},
                          mcts_config=mcts_config,
                          ranked_rewards=ranked_rewards)
                .rollouts(num_rollout_workers=1)
            )
    agent = config.build()

    agent.restore(args.checkpoint_path)
    policy = agent.get_policy(DEFAULT_POLICY_ID)

    env = BPP(env_config=env_config)
    obs = env.reset()

    episode = Episode(
        PolicyMap(0, 0),
        lambda _, __: DEFAULT_POLICY_ID,
        lambda: None,
        lambda _: None,
        0,
    )

    episode.user_data['initial_state'] = env.get_state()

    done = False

    while not done:
        action, _, _ = policy.compute_single_action(obs, episode=episode)
        v_current = policy.model.value_function().item()
        obs, reward, done, _ = env.step(action)
        episode.v_values.append(v_current)
        episode.length += 1

    # Compute value drops
    value_drops = []

    for t in range(len(episode.v_values) - 1):

        v_t = float(episode.v_values[t])
        v_t1 = float(episode.v_values[t + 1])

        # Positive delta = worsening
        delta = v_t - v_t1

        value_drops.append(delta)

    # Percentile threshold (top 10% largest drops)
    threshold = np.percentile(value_drops, 90)

    # Count backtracking events
    replan_events = sum(
        delta > threshold
        for delta in value_drops
    )

    episode.replan_events = replan_events

    print(f"Backtracking threshold: {threshold:.4f}")
    print(f"Backtracking events: {replan_events}")

    env.render()
    ray.shutdown()


if __name__ == '__main__':
    evaluate()
