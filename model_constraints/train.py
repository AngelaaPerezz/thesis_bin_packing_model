import matplotlib
matplotlib.use('Agg')  # non-interactive backend, must be before pyplot import
import matplotlib.pyplot as plt
import pandas as pd
import ray
from ray.rllib.algorithms import alpha_zero
from ray.tune.registry import register_env
import time
from env import BPP
from model import Agent

register_env('Bpp-v1', BPP)

# Curriculum: ordered by num_items (main difficulty driver)
# Both bin sizes trained together at each stage
CURRICULUM = [
    # Stage 1: 13 items (easiest)
    [{'bin_size': [6,6], 'max_bin_size': [13,13], 'num_items': 4}],
    [{'bin_size': [8,8], 'max_bin_size': [13,13], 'num_items': 6}],
    [
     {'bin_size': [13, 13], 'max_bin_size': [13, 13], 'num_items': 13}],
    # Stage 2: add 14 items
    [{'bin_size': [12, 12], 'max_bin_size': [12, 12], 'num_items': 13},
     {'bin_size': [13, 13], 'max_bin_size': [13, 13], 'num_items': 13},
     {'bin_size': [12, 12], 'max_bin_size': [12, 12], 'num_items': 14},
     {'bin_size': [13, 13], 'max_bin_size': [13, 13], 'num_items': 14}],
    # Stage 3: add 15 items
    [{'bin_size': [12, 12], 'max_bin_size': [12, 12], 'num_items': 13},
     {'bin_size': [13, 13], 'max_bin_size': [13, 13], 'num_items': 13},
     {'bin_size': [12, 12], 'max_bin_size': [12, 12], 'num_items': 14},
     {'bin_size': [13, 13], 'max_bin_size': [13, 13], 'num_items': 14},
     {'bin_size': [12, 12], 'max_bin_size': [12, 12], 'num_items': 15},
     {'bin_size': [13, 13], 'max_bin_size': [13, 13], 'num_items': 15}],
    # Stage 4: full curriculum with 16 items
    [{'bin_size': [12, 12], 'max_bin_size': [12, 12], 'num_items': 13},
     {'bin_size': [13, 13], 'max_bin_size': [13, 13], 'num_items': 13},
     {'bin_size': [12, 12], 'max_bin_size': [12, 12], 'num_items': 14},
     {'bin_size': [13, 13], 'max_bin_size': [13, 13], 'num_items': 14},
     {'bin_size': [12, 12], 'max_bin_size': [12, 12], 'num_items': 15},
     {'bin_size': [13, 13], 'max_bin_size': [13, 13], 'num_items': 15},
     {'bin_size': [12, 12], 'max_bin_size': [12, 12], 'num_items': 16},
     {'bin_size': [13, 13], 'max_bin_size': [13, 13], 'num_items': 16}],
]

# Iterations per stage — more on early stages to build a solid foundation
ITERATIONS_PER_STAGE = [2, 10, 200, 200, 200, 400]


def make_config(env_config):
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
        "buffer_max_length": 500,
        "initialize_buffer": True,
        "num_init_rewards": 50,
    }
    _tmp_env = BPP(env_config)
    obs_space    = _tmp_env.observation_space
    action_space = _tmp_env.action_space
    _tmp_env.close()

    return (
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
            train_batch_size=256,
            lr=5e-5,
            grad_clip=0.5,
        )
        .rollouts(
            num_rollout_workers=4,
            rollout_fragment_length=50,
        )
    )


def train():
    ray.init(num_gpus=1)

    all_rewards   = []
    all_ep_lens   = []
    all_norms     = []
    # checkpoint_path = r"C:\Users\sira1\ray_results\AlphaZero_Bpp-v1_2026-06-03_13-40-10wnmi65_m\checkpoint_000050"
    checkpoint_path = None  # Set to None to start from scratch, or provide path to resume
    for stage_idx, configs in enumerate(CURRICULUM):
        stage_num    = stage_idx + 1
        num_iters    = ITERATIONS_PER_STAGE[stage_idx]
        env_config   = {'configs': configs}
        config       = make_config(env_config)
        algo         = config.build()

        # Restore weights from previous stage
        if checkpoint_path is not None:
            algo.restore(checkpoint_path)
            print(f"\n=== Stage {stage_num}: restored from {checkpoint_path} ===")
        else:
            print(f"\n=== Stage {stage_num}: starting from scratch ===")

        config_strings = ", ".join(
            f"{c['bin_size'][0]}x{c['bin_size'][1]} n={c['num_items']}"
            for c in configs
        )

        print(f"Configs: [{config_strings}]")
        stage_rewards = []
        stage_ep_lens = []

        for i in range(num_iters):
            t0 = time.time()
            results    = algo.train()
            t1 = time.time()
            print(f"Iteration {i+1} took {t1-t0:.1f}s")
            print(f"  episodes this iter: {results.get('episodes_this_iter', '?')}")
            print(f"  timesteps total: {results.get('timesteps_total', '?')}")
            print(f"  episodes total: {results.get('episodes_total', '?')}")
            reward     = results.get('episode_reward_mean', float('nan'))
            ep_len     = results.get('episode_len_mean', float('nan'))
            policy     = algo.get_policy()
            norm       = sum(p.norm().item() for p in policy.model.parameters())

            stage_rewards.append(reward)
            stage_ep_lens.append(ep_len)
            all_rewards.append(reward)
            all_ep_lens.append(ep_len)
            all_norms.append(norm)

            print(f"  S{stage_num} iter {i+1}/{num_iters} | reward={reward:.3f} ep_len={ep_len:.2f} norm={norm:.3f}")

            # Checkpoint every 50 iterations
            if (i + 1) % 50 == 0:
                checkpoint_path = algo.save()
                print(f"  Checkpoint: {checkpoint_path}")

                # Save plot
                fig, axes = plt.subplots(3, 1, figsize=(12, 9))
                smoothed_r = pd.Series(all_rewards).rolling(10, min_periods=1).mean()
                smoothed_e = pd.Series(all_ep_lens).rolling(10, min_periods=1).mean()
                axes[0].plot(all_rewards, alpha=0.3, color='blue')
                axes[0].plot(smoothed_r, color='blue', linewidth=2)
                axes[0].set_title('Reward mean (raw + smoothed)')
                axes[0].set_xlabel('Iteration (total)')
                axes[1].plot(all_ep_lens, alpha=0.3, color='green')
                axes[1].plot(smoothed_e, color='green', linewidth=2)
                axes[1].set_title('Episode length mean (raw + smoothed)')
                axes[1].set_xlabel('Iteration (total)')
                axes[2].plot(all_norms, color='orange')
                axes[2].set_title('Weight norm')
                axes[2].set_xlabel('Iteration (total)')
                # Mark stage boundaries
                boundary = 0
                for s, n in enumerate(ITERATIONS_PER_STAGE[:stage_idx+1]):
                    boundary += n
                    for ax in axes:
                        ax.axvline(x=boundary, color='red', linestyle='--', alpha=0.5,
                                   label=f'Stage {s+2}' if boundary < len(all_rewards) else '')
                plt.tight_layout()
                plt.savefig(f'training_stage{stage_num}_iter{i+1}.png')
                plt.close()
                print(f"  Plot saved: training_stage{stage_num}_iter{i+1}.png")

        checkpoint_path = algo.save()
        print(f"\nStage {stage_num} complete. Checkpoint: {checkpoint_path}")
        algo.stop()

    ray.shutdown()
    print("\nTraining complete.")


if __name__ == '__main__':
    train()