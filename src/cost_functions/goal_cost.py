import torch
import numpy as np
import matplotlib.pyplot as plt

from cost_functions.base import CostFunction

class GoalCost(CostFunction):
    '''Distance-to-goal cost class to be used with MPPI algorithm. 

    Handles cost querying for single state or state-action pairs, as well as batch querying.

    Args:
        goal:
            Tensor(2,) containing the goal position [x_g, y_g] in world coordinates
        threshold:
            Float. Tolerance [m] for termination cost 
        fail_cost:
            Flaot. Cost of not reaching goal at terminal state
    '''
    def __init__(self, goal, threshold=0.5, fail_cost=10, device="cpu"):
        self.goal = goal
        self.threshold = threshold
        self.fail_cost = fail_cost
        self.device = device

        self.goal = self.goal.to(self.device)

    def update_goal(self, goal):
        '''Updates the goal to be used for distance-to-goal cost

        Args:
            goal:
                Tensor(2,) containing the goal position [x_g, y_g] in world coordinates
        '''
        self.goal = goal.to(self.device)

    def stagecost(self, state, action=None):
        '''Stage cost: Cost based on distance to goal
        
        Args:
            state:
                Tensor(4,) or Tensor(K, 4) containing state of the vehicle [x, y, theta, delta]
            action:
                Tensor(2,) or Tensor(K, 2) containing the input control to the vehicle [v, phi]
        Returns:
            cost:
                Float or Tensor(K, 1) representing the stage cost (non-terminal cost) of the current state and action pair
        ''' 
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            action = action.unsqueeze(0) if action is not None else None
            cost = self.stagecost(state, action)
            return cost.item()

        state = state.to(self.device)
        action = action.to(self.device) if action is not None else None

        cost = 0.0

        # Extract world positions
        world_pos = state[:,:2]
        # TODO: Check if necessary on real robot
        world_pos = torch.index_select(world_pos, 1, torch.LongTensor([1, 0]).to(self.device))
        cost = torch.linalg.vector_norm(world_pos - self.goal, dim=1) #.cuda()

        return cost
        
    def termcost(self, state):
        '''Termination cost: Cost based on distance to goal.

        Args:
            state:
                Tensor(4,) or Tensor(K, 4) containing state of the vehicle [x, y, theta, delta]
            goal:
                Tensor(2,) containing the goal position [x_g, y_g] in world coordinates
        Returns:
            cost:
                Float or Tensor(K, 1) representing the terminal cost of the state input
        '''
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            cost = self.termcost(state)
            return cost.item()
    
        state = state.to(self.device)

        world_pos = state[:,:2]
        goal_mask = torch.linalg.vector_norm(world_pos - self.goal, dim=1) < self.threshold
        cost = torch.zeros(world_pos.shape[0]).to(self.device)
        cost[~goal_mask] = self.fail_cost

        return cost


def main():
    pass


if __name__=="__main__":
    ## 1. Initialize CostFunction object
    goal   = torch.Tensor([5.0, 5.0])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cost_fn = GoalCost(goal, threshold=0.5, fail_cost=10, device=device)

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