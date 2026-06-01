from ray.tune.registry import register_env


from env import BPP
from model import Agent
from demo_video import get_demo


register_env('Bpp-v1', BPP)


def evaluate():
    

    env_config = {'bin_size': [10, 10], 'max_bin_size': [10, 10], 'num_items': 10}
    frame_id = 0
    
    env = BPP(env_config=env_config)
    obs, items = env.reset()

    env.render(frame_id)
    agent = Agent()
   
    done = False


    while not done:


        agent.add_items_to_memory(items)
        retrieved_items = agent.retrieve_items(env.num_remaining_items)

        for item in retrieved_items:
            
            action = agent.get_position_of_item(obs, item)
            if len(action) == 0:
                done = True
                break
            obs, items, _, done, _ = env.step(action)
            env.render(frame_id=frame_id)
            frame_id += 1

    
            
    final_reward = agent.cognitive_cost + env.running_reward

    print(f"The resulting reward of this trial is {final_reward}")
    env.render()


if __name__ == '__main__':
    evaluate()
    get_demo()