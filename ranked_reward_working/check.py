from ray.rllib.utils.pre_checks.env import check_env
from env import BPP 

env = BPP(env_config={
    'bin_size': [10, 10],
    'max_bin_size': [10, 10],
    'num_items': 10,
})

check_env(env) 