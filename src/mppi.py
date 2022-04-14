import torch
import numpy as np
from model import KinematicBycicle
from cost import CostFunction
import matplotlib.pyplot as plt

class MPPI:
    '''MPPI controller based on Algorithm 2 in [1].
    
    [1] Williams, Grady, et al. "Information theoretic MPC for model-based reinforcement learning." 2017 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2017.

    Args:
        model:
            Model instance. Look at model.py to see interface definition.
        cost_fn:
            Cost function instance. Look at cost.py to see interface definition.
        num_samples:
            Number of samples that MPPI will evaluate per timestep.
        num_timesteps:
            Horizon of control for MPPI controller.
        control_params:
            Dictionary of control parameters defined as follows:
            {sys_noise, temperature}
    '''
    def __init__(self, model, cost_fn, num_samples, num_timesteps, control_params):
        self.model = model
        self.cost_fn = cost_fn
        self.K = num_samples
        self.T = num_timesteps

        self.n = self.model.state_dim()
        self.m = self.model.control_dim()

        self.last_controls = [torch.zeros(self.m) for t in range(self.T)]

        self.sys_noise = control_params["sys_noise"]  # Equivalent to sigma in [1]
        self.temperature = control_params["temperature"]  # Equivalent to lambda in [1]

    def get_control(self, x):
        '''Returns action u at current state x

        Args:
            x: 
                Current state. Expects tensor of size self.model.state_dim()
        
        Returns:
            u: 
                Action from MPPI controller. Returns Tensor(self.m)

            cost:
                Cost from executing the optimal action sequence. Float.
        '''
        ## 1. Simulate K rollouts and get costs for each rollout
        costs = []
        control_noise = torch.normal(mean=0.0, std=self.sys_noise, size=(self.K, self.T, self.m))
        for k in range(self.K):
            # Sample noise to be added to initial control sequence
            noise = control_noise[k]

            cost = 0
            x_hist = []
            x_hist.append(x)
            sigma = self.sys_noise * torch.eye(self.m)
            for t in range(1, self.T):
                u_t = self.last_controls[t-1]
                w_t = noise[t-1]
                x_t = self.model.forward(x_hist[t-1], u_t + w_t)
                x_hist.append(x_t)
                cost += (self.cost_fn.stagecost(x_t, u_t) + self.temperature*torch.matmul(u_t.view(1,-1), torch.matmul(sigma, w_t.view(-1,1))).item())

            cost +=  self.cost_fn.termcost(x_hist[-1])

            costs.append(cost)
        costs = torch.Tensor(costs)

        ## 2. Get minimum cost and obtain normalization constant
        beta = torch.min(costs)
        eta  = torch.sum(torch.exp(-1/self.temperature*(costs - beta)))

        ## 3. Get importance sampling weight
        sampling_weights = []
        for k in range(self.K):
            weight = 1/eta * torch.exp(-1/self.temperature * (costs[k] - beta))
            sampling_weights.append(weight)
        sampling_weights = torch.Tensor(sampling_weights)

        ## 4. Get action sequence using weighted average
        for t in range(self.T):
            self.last_controls[t] += torch.sum(torch.mul(sampling_weights.view(-1,1), control_noise[:,t,:])) # TODO: check dimensions

        ## 5. Return first action in sequence and update self.last_controls
        u = self.last_controls[0]
        self.step()

        return u
        
    def step(self):
        '''Updates self.last_controls to warm start next action sequence.'''
        # Shift controls by one
        self.last_controls = self.last_controls[1:]
        # Initialize last control to be the same as the last in the sequence
        self.last_controls.append(self.last_controls[-1])

    def viz():
        pass

def main():

    # Initialize model and model parameters
    L = 1.0
    dt = 0.01
    u_max = torch.Tensor([10, torch.pi/2])
    u_min = torch.Tensor([0, -torch.pi/2])

    model = KinematicBycicle(L, dt, u_max, u_min)

    # Initialize cost function
    height = 10
    width  = 10
    resolution  = 0.05
    origin = [0.0, 0.0]

    map_params = {
        'height': height,
        'width': width,
        'resolution': resolution,
        'origin': origin
        }

    map_size = (int(height/resolution), int(width/resolution))

    costmap = np.zeros((map_size))

    # Add high cost in the middle of cost function to do "barrel test"
    map_height_third = int(height/(3*resolution))
    map_width_third  = int(width/(3*resolution))
    costmap[map_height_third:2*map_height_third, map_width_third:2*map_width_third] = 3 + np.random.randn(map_height_third, map_width_third)

    costmap = torch.from_numpy(costmap)
    
    goal = torch.Tensor([2.5, 2.5])
    # Initialize CostFunction object
    cost_fn = CostFunction(costmap, map_params, goal)

    # Initialize controller
    num_samples = 100
    num_timesteps = 50
    control_params = {
        'sys_noise': 0.01,
        'temperature': 1
    }
    controller = MPPI(model, cost_fn, num_samples, num_timesteps, control_params)

    max_iters = 10000
    iter = 0
    x_hist = []
    u_hist = []

    x = torch.Tensor([0.0, 0.0, 0.0, 0.0])
    x_hist.append(x)
    fig = plt.figure()
    fig.suptitle('MPPI demo')
    path_subplot = fig.add_subplot(1,1,1)

    while iter < max_iters:
        print(f"Iteration: {iter}")
        if iter % 10 == 0:
            fig.suptitle(f'MPPI demo. Iteration {iter}')
            print("---")
            print(f"Iteration: {iter}")
        u = controller.get_control(x)
        if iter % 10 == 0:
            print(f"State is: ")
            print(x)
            print(f"Chosen control is: ")
            print(u)
        x = model.forward(x, u)

        x_hist.append(x)
        u_hist.append(u)

        pos_x = [x_hist[iter][0] for i in range(len(x_hist))]
        pos_y = [x_hist[iter][1] for i in range(len(x_hist))]
        grid_pos,_ = cost_fn.world_to_grid(torch.cat([torch.Tensor(pos_x).view(-1,1), torch.Tensor(pos_y).view(-1,1)], dim=1))
        x_vals = grid_pos[:,1]
        y_vals = grid_pos[:,0]

        if iter % 10 == 0:
            path_subplot.clear()
            path_subplot.grid()
            path_subplot.set_xlim([0, costmap.shape[0]])
            path_subplot.set_ylim([0, costmap.shape[1]])
            path_subplot.plot(x_vals, y_vals, c='blue')
            path_subplot.scatter(int((x_vals[0]-origin[0])/resolution), int((y_vals[0]-origin[1])/resolution), c='red', marker='o')
            path_subplot.scatter(int((x_vals[iter]-origin[0])/resolution), int((y_vals[iter]-origin[1])/resolution), c='green', marker='x')
            path_subplot.scatter(int((goal[0]-origin[0])/resolution), int((goal[1]-origin[1])/resolution), c='cyan', marker='s')
            plt.pause(dt)

        iter += 1

    

if __name__=="__main__":
    main()