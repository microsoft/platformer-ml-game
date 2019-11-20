# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from malmopy.agents import QLearnerAgent
from malmopy import AgentMode
import numpy as np
from numpy.random import choice

def softmax(x):
    # from https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)
    
def sharpen_probability_distribution(x, sharpening_rate):
    x_sharpened_unnormalized = np.power(x, sharpening_rate)
    return x_sharpened_unnormalized / np.sum(x_sharpened_unnormalized)

class StochasticEvaluationQLearnerAgent (QLearnerAgent):
    def __init__(self, *args):
        super(StochasticEvaluationQLearnerAgent, self).__init__(*args)
        
    def act(self, observation):
        qvalues = self.qfunction.compute(observation, is_training=False).reshape([-1])
        qvalues_choice_distribution = sharpen_probability_distribution(softmax(qvalues), 4)
        action = np.random.choice(range(len(qvalues_choice_distribution)), 1, p=qvalues_choice_distribution)[0]
        return action
