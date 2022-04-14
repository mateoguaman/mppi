import torch
import numpy as np
import matplotlib.pyplot as plt

class CostFunction:
    '''CostFunction class to be used with MPPI algorithm. 

    Handles cost querying for single state or state-action pairs, as well as batch querying.

    Args:
        costmap:
            Input 2D tensor containing the initial costmap.
        map_params:
            Dictionary containing metadata for the costmap in metric space. Contains the following keys: 
            {
                'height': Float [m],
                'width': Float [m],
                'resolution': Float [m],
                'origin': [Float, Float] [m]
            }
        goal:
            Tensor(2,) containing the goal position [x_g, y_g] in world coordinates
    '''
    def __init__(self, costmap, map_params, goal=None):
        self.costmap = costmap.cuda()
        self.map_params = map_params
        self.goal = goal.cuda()

        self.invalid_cost = 1000000
    def update_costmap(self, costmap):
        '''Updates the costmap to be queried.

        Args:
            costmap:
                Input 2D tensor containing the initial costmap.
        '''
        self.costmap = costmap.cuda()

    def update_goal(self, goal):
        '''Updates the goal to be used for distance-to-goal cost

        Args:
            goal:
                Tensor(2,) containing the goal position [x_g, y_g] in world coordinates
        '''
        self.goal = goal.cuda()

    def stagecost(self, state, action):
        '''Stage cost: Queries the costmap at the input state and adds cost based on distance to goal
        
        Args:
            state:
                Tensor(4,) or Tensor(K, 4) containing state of the vehicle [x, y, theta, delta]
            action:
                Tensor(2,) or Tensor(K, 2) containing the input control to the vehicle [v, phi]
        Returns:
            cost:
                Float or Tensor(K, 1) representing the stage cost (non-terminal cost) of the current state and action pair
        ''' 
        cost = 0.0
        # Get (x,y) positions from state
        if len(state.shape) == 1:
            world_pos = state[:2]
            if self.goal is not None:
                goal_cost = torch.linalg.vector_norm(world_pos - self.goal).cuda()
        else:
            # import pdb;pdb.set_trace()
            world_pos = state[:,:2]
            if self.goal is not None:
                goal_cost = torch.linalg.vector_norm(world_pos - self.goal, dim=1).cuda()
        
        # Get grid indices in costmap from world positions
        grid_pos, invalid_mask = self.world_to_grid(world_pos)

        # Switch grid axes to align with robot centric axes: +x forward, +y left
        if len(state.shape) == 1:
            grid_pos = torch.Tensor([grid_pos[1], grid_pos[0]]).cuda()
        else:
            grid_pos = torch.index_select(grid_pos, 1, torch.LongTensor([1, 0]).cuda()).cuda()

        # Assign invalid costmap indices to a temp value and then set them to invalid cost
        grid_pos[invalid_mask] = 0.0
        grid_pos = grid_pos.long()

        if len(state.shape) == 1:
            cost = torch.clone(self.costmap[grid_pos[0], grid_pos[1]]).cuda()
        else:    
            cost = torch.clone(self.costmap[grid_pos[:,0], grid_pos[:,1]]).cuda()

        if self.goal is not None:
            cost += 0.1*goal_cost

        cost[invalid_mask] = self.invalid_cost

        if len(state.shape) == 1:
            cost = cost.item()

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
            world_pos = state[:2]
            if torch.linalg.vector_norm(world_pos - self.goal).item() < 1:
                cost = 0
            else:
                cost = 10
        else:
            world_pos = state[:,:2]
            goal_mask = torch.linalg.vector_norm(world_pos - self.goal, dim=1) < 0.5
            cost = torch.zeros(world_pos.shape[0]).cuda()
            cost[~goal_mask] = 10

        return cost

    def world_to_grid(self, world_pos):
        '''Converts the world position (x,y) into indices that can be used to access the costmap.
        
        Args:
            world_pos:
                Tensor(2,) or Tensor(k,2) representing the world position being queried in the costmap.
        Returns:
            grid_pos:
                Tensor(2,) or Tensor(k,2) representing the indices in the grid that correspond to the world position
        '''
        world_pos = world_pos.cuda()
        res = self.map_params['resolution']
        origin = torch.Tensor(self.map_params['origin']).cuda()

        grid_pos = ((world_pos - origin)/res).to(torch.int32)

        # Obtain mask of invalid grid locations in pixel space
        grid_min = torch.Tensor([0, 0]).cuda()
        grid_max = torch.Tensor(list(self.costmap.shape)).cuda()
        invalid_mask = (grid_pos < grid_min).any(dim=-1) | (grid_pos >= grid_max).any(dim=-1)

        return grid_pos.cuda(), invalid_mask.cuda()

def main():
    pass


if __name__=="__main__":
    
    ## 1. Instantiate cost function
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

    ## 3. Initialize CostFunction object
    goal   = torch.Tensor([5.0, 5.0])
    cost_fn = CostFunction(costmap, map_params, goal)

    ## 3. Verify that world_to_grid works:
    world_pts = torch.Tensor([[1.0, 0.0], [1.0, 1.0], [2.0, 3.2]])
    # grid_pos, invalid_mask = cost_fn.world_to_grid(world_pts)

    # print("=====")
    # print("Checking grid_pos: ")
    # print(f"Input is \n{world_pts}")
    # print(f"Output is \n{grid_pos}")

    ## 4. Verify that stagecost works
    # state  = torch.Tensor([[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0], [5.0, 1.0, 0.0, 0.0], [5.0, 5.0, 0.0, 0.0]])
    state = torch.Tensor([1.0, 0.0, 0.0, 0.0])
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