from short_term_memory import ShortTermMemory
from cognitive_constraints import get_candidate_actions


class Agent():
    def __init__(self):
        self.stm = ShortTermMemory()
        self.cognitive_cost = 0

    def add_item(self, item_id):
        self.stm.add_plan(item_id)
        self.decay_activations()
    
    def add_items_to_memory(self, items):
        for item in items:
            self.add_item(item)

    def retrieve_items(self):
        retrieved_items, cognitive_cost = self.stm.retrieve_from_memory()
        self.cognitive_cost += cognitive_cost
        return retrieved_items

    
    def get_position_of_item(self, obs, item_id):
        candidate_actions = get_candidate_actions(obs, item_id)
        return candidate_actions[0]



