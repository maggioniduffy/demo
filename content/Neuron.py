import numpy as np
from typing import Tuple

class Neuron:
    def __init__(self, weights: list, count: int, position: Tuple[int,int]):
        self.weights = weights
        self.count = count
        self.elements = np.array([])
        self.position = position
                 
    def add_element(self, elem):
        self.elements = np.append(self.elements,elem)
        
    def get_weights(self):
        return self.weights