from ray.tune.registry import register_env


from env import BPP
from model import Agent

register_env('Bpp-v1', BPP)


def evaluate():
    

    env_config = {'bin_size': [10, 10], 'max_bin_size': [10, 10], 'num_items': 10}

    
    env = BPP(env_config=env_config)
    obs, items = env.reset()
    agent = Agent()

   
    done = False

    while not done:
        
            agent.add_items_to_memory(items)
            retrieved_items = agent.retrieve_items()

            for item in retrieved_items:
                action = agent.get_position_of_item(obs, item)
                obs, _, done, _ = env.step(action)
                
    final_reward = agent.cognitive_cost + env.running_reward

    print(f"The resulting reward of this trial is {final_reward}")
    env.render()


if __name__ == '__main__':
    evaluate()
