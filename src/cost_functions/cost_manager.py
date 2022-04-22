import torch
import numpy as np

from cost_functions.base import CostFunction
from cost_functions.costmap import Costmap
from cost_functions.goal_cost import GoalCost

class CostManager(CostFunction):
    '''CostManager class to be used with MPPI algorithm. 

    Combines multiple cost functions with different weights for stagecost and termcost

    Args:
        cost_functions:
            List of k CostFunction objects to be combined.
        stagecost_weights:
            Tensor(k,) of linear combination weights for cost_functions.
        termcost_weights:
            Tensor(k,) of linear combination weights for cost_functions.
    '''
    def __init__(self, cost_functions, stagecost_weights, termcost_weights, device="cpu"):
        self.cost_functions = cost_functions
        self.stagecost_weights = stagecost_weights
        self.termcost_weights = termcost_weights
        self.device = device


    def stagecost(self, state, action=None):
        '''Stage cost: Linear combination of stage costs.
        
        Args:
            state:
                Tensor(4,) or Tensor(K, 4) containing state of the vehicle [x, y, theta, delta]
            action:
                Tensor(2,) or Tensor(K, 2) containing the input control to the vehicle [v, phi]
        Returns:
            cost:
                Float or Tensor(K, 1) representing the stage cost (non-terminal cost) of the current state and action pair
        ''' 
        cost = 0

        for i, cost_fn in enumerate(self.cost_functions):
            cost += self.stagecost_weights[i] * cost_fn.stagecost(state, action)

        return cost
        
    def termcost(self, state):
        '''Termination cost: Linear combination of termination costs.

        Args:
            state:
                Tensor(4,) or Tensor(K, 4) containing state of the vehicle [x, y, theta, delta]

        Returns:
            cost:
                Float or Tensor(K, 1) representing the terminal cost of the state input
        '''
        cost = 0

        for i, cost_fn in enumerate(self.cost_functions):
            cost += self.termcost_weights[i] * cost_fn.termcost(state)

        return cost

def main():
    ## 1. Instantiate costmap function
    height = 10
    width  = 10
    resolution  = 0.05
    origin = [0.0, 0.0]
    # origin = [0.0, 0.0]

    map_params = {
        'height': height,
        'width': width,
        'resolution': resolution,
        'origin': origin
        }

    map_size = (int(height/resolution), int(width/resolution))

    costmap = np.zeros((map_size))

    ## 2. Add high cost in the middle of cost function to do "barrel test"
    map_height_third = int(height/(3*resolution))
    map_width_third  = int(width/(3*resolution))
    costmap[map_height_third:2*map_height_third, map_width_third:2*map_width_third] = 3 + np.random.randn(map_height_third, map_width_third)

    costmap = torch.from_numpy(costmap)

    ## Initialize Costmap object
    device = "cuda" if torch.cuda.is_available() else "cpu"
    costmap_fn = Costmap(costmap, map_params, device)


    ## 2. Initialize CostFunction object
    goal   = torch.Tensor([5.0, 5.0])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    goalcost_fn = GoalCost(goal, threshold=0.5, fail_cost=10, device=device)


    ## 3. Initialize cost manager
    cost_functions = [costmap_fn, goalcost_fn]
    stagecost_weights = torch.Tensor([1, 0.1])
    termcost_weights = torch.Tensor([1, 5])
    cost_fn = CostManager(cost_functions, stagecost_weights, termcost_weights, device="cpu")


    ## 4. Verify that stagecost works
    state  = torch.Tensor([[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0], [5.0, 1.0, 0.0, 0.0], [5.0, 5.0, 0.0, 0.0]])
    # state = torch.Tensor([1.0, 0.0, 0.0, 0.0])
    action = torch.Tensor([0.0, 0.0])
    
    for i in range(10):
        print(f"Iteration: {i}")
        cost = cost_fn.stagecost(state, action)
        print(cost)
    cost = cost_fn.stagecost(state, action)
    
    print("---")
    print(f"Input states are: ")
    print(state)
    print(f"Stage costs for states are: ")
    print(cost)

    cost = cost_fn.termcost(state)
    print("---")
    print(f"Term costs for states are: ")
    print(cost)


if __name__=="__main__":
    main()