import torch

class CostFunction:
    def __init__(self, costmap, map_params):
        self.costmap = costmap
        self.map_params = map_params
    def stagecost(self, state, action):
        pass
    def termcost(self, state):
        pass
    def batch_stagecost(self, states, actions):
        pass
    def batch_termcost(self, states):
