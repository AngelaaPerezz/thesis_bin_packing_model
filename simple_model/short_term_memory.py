import math
import random
import numpy as np
from scipy.stats import gamma

class ShortTermMemory:
    def __init__(self, capacity=4, decay=0.001, threshold=0.5, noise_sd=0.01):
        """
        Short-term memory for storing plans with forgetting as retrieval failure.
        """
        self.capacity = capacity       # max number of plans STM can hold
        self.decay = decay             # decay rate per timestep
        self.threshold = threshold     # activation threshold for retrieval
        self.noise_sd = noise_sd       # standard deviation of noise
        self.memory = []               # list of plans


    def softmax(self, x, tau=0.1):
        """Compute softmax probabilities."""
        e_x = [math.exp(a / tau) for a in x]
        s = sum(e_x)
        return [v / s for v in e_x]

    def add_plan(self, plan):
        """Add a new plan to STM"""
        if len(self.memory) <= self.capacity:
            self.memory.append({'plan': plan, 'activation': 1.0})

    def remove_plan(self, plan):
        self.memory = [entry for entry in self.memory if entry['plan'] != plan]




    def decay_activations(self):
        """Decay all plan activations to simulate forgetting over time."""
        for p in self.memory:
            p['activation'] -= self.decay


    def retrieve_from_memory(self, number_remaining_items):
        """
        Retrieve up to memory_limit items from memory probabilistically.
        Retrieval can fail for some items.
        Gamma - (12, 0.273) (simulate peak of retrieving blocks probability at
        point (3, 0.45) following the graph in the SCH paper)
        """
        cognitive_cost = 0
        temp_memory = self.memory.copy()  
        retrieved = []

        activations = [p['activation'] for p in temp_memory]

        # gamma parameters
        encoding_memory = [1, 2, 3, 4, 5]
        parameters = [[7.3225, 0.1582], ]

        # parameters for 4 objects planned ahead
        k = 12.0        # shape
        theta = 0.273   # scale 

        boxes = np.arange(0,number_remaining_items) 
        print("remaining items", number_remaining_items)
        weights = gamma.pdf(boxes, a=k, scale=theta)
        probabilities = weights / weights.sum()
        number_of_boxes_retrieved = np.random.choice(boxes, size=1, p=probabilities)

        probs = self.softmax(activations, self.noise_sd)
        retrieved_indexes = np.random.choice(
            len(probs),
            size=number_of_boxes_retrieved,
            replace=False,
            p=probs
        )

        for i in retrieved_indexes:
            retrieved.append(self.memory[i]['plan'])        
        
        for entry in retrieved:
            self.remove_plan(entry)

        print("Rertrieved indexes", retrieved_indexes)
    
        return retrieved, cognitive_cost
        
    # def retrieve_from_memory(self):
    #     """
    #     Retrieve up to memory_limit items from memory probabilistically.
    #     Retrieval can fail for some items.
    #     """
    #     cognitive_cost = 0
    #     temp_memory = self.memory.copy()  
    #     retrieved = []
    #     activations = [p['activation'] for p in temp_memory]
    #     probs = self.softmax(activations, self.noise_sd)
    #     print("probs", probs)
    #     for i, item in enumerate(temp_memory):
    #         if len(retrieved) >= self.capacity:
    #             break

    #         if random.random() < probs[i]:
    #             self.memory = [
    #                 m for m in self.memory
    #                 if m['plan'] != item['plan']
    #             ]
    #         else:
    #             cognitive_cost += 1
    
    #     return retrieved, cognitive_cost
    

    