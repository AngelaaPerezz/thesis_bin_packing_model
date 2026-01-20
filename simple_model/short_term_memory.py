import math
import random


class ShortTermMemory:
    def __init__(self, capacity=4, decay=0.1, threshold=0.5, noise_sd=0.1):
        """
        Short-term memory for storing plans with forgetting as retrieval failure.
        """
        self.capacity = capacity       # max number of plans STM can hold
        self.decay = decay             # decay rate per timestep
        self.threshold = threshold     # activation threshold for retrieval
        self.noise_sd = noise_sd       # standard deviation of noise
        self.memory = []               # list of plans


    def softmax(x, tau=0.1):
        """Compute softmax probabilities."""
        e_x = [math.exp(a / tau) for a in x]
        s = sum(e_x)
        return [v / s for v in e_x]

    def add_plan(self, plan):
        """Add a new plan to STM"""
        self.memory.append({'plan': plan, 'activation': 1.0})

    def decay_activations(self):
        """Decay all plan activations to simulate forgetting over time."""
        for p in self.memory:
            p['activation'] -= self.decay

    
    def retrieve_from_memory(self):
        """
        Retrieve up to memory_limit items from memory probabilistically.
        Retrieval can fail for some items.
        """
        cognitive_cost = 0
        temp_memory = self.memory.copy()  
        retrieved = []
        activations = [p['activation'] for p in temp_memory]
        probs = self.softmax(activations, tau=self.noise_sd)
        
        for i, item in enumerate(temp_memory):
            if len(retrieved) >= self.capacity:
                break

            if random.random() < probs[i]:
                retrieved.append(item)
                self.memory[item].remove()
            else:
                cognitive_cost += 1
    
        return retrieved, cognitive_cost
    

    