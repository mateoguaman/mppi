import torch
from torch.distributions import multivariate_normal
import numpy as np
from model import KinematicBycicle
from cost import CostFunction
import matplotlib.pyplot as plt

from timeit import default_timer as timer
from datetime import timedelta

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
            {'sys_noise': Tensor(2,), 
            'temperature'" Float}
    '''
    def __init__(self, model, cost_fn, num_samples, num_timesteps, control_params):
        self.model = model
        self.cost_fn = cost_fn
        self.K = num_samples
        self.T = num_timesteps

        self.n = self.model.state_dim()
        self.m = self.model.control_dim()

        self.last_controls = torch.zeros(self.T, self.m).cuda()

        self.sys_noise = control_params["sys_noise"]  # Equivalent to sigma in [1]
        self.temperature = control_params["temperature"]  # Equivalent to lambda in [1]

        mean = torch.zeros(self.m).cuda()
        self.sigma = torch.diagflat(self.sys_noise).cuda()

        self.control_noise_dist = multivariate_normal.MultivariateNormal(loc=mean, covariance_matrix=self.sigma)

    def get_control(self, x):
        '''Returns action u at current state x

        Args:
            x: 
                Current state. Expects Tensor(self.n,)
        
        Returns:
            u: 
                Action from MPPI controller. Returns Tensor(self.m,)

            cost: 
                Cost from executing the optimal action sequence. Float.
        '''
        ## 1. Simulate K rollouts and get costs for each rollout
        control_noise = self.control_noise_dist.sample(sample_shape=(self.K, self.T)).cuda()

        x_hist = torch.zeros(self.K, self.T, self.n).cuda()
        x_hist[:,0,:] = x
        u_hist = self.last_controls.unsqueeze(0).repeat(self.K, 1, 1).cuda()

        costs = torch.zeros(self.K).cuda()

        for t in range(1, self.T):
            x_prev = x_hist[:,t-1,:]
            u_prev = u_hist[:,t-1,:]
            w_prev = control_noise[:,t-1,:]

            x_t = self.model.forward(x_prev, u_prev+w_prev)
            x_hist[:,t,:] = x_t

            control_penalty = self.temperature*(torch.matmul(u_prev.view(-1, self.m, 1).transpose(1,2), torch.linalg.solve(self.sigma.unsqueeze(0), w_prev.view(-1, self.m, 1)))).squeeze().cuda()

            costs += (self.cost_fn.stagecost(x_t, u_prev) + control_penalty)

        costs += self.cost_fn.termcost(x_hist[:,-1,:])

        ## 2. Get minimum cost and obtain normalization constant
        beta = torch.min(costs)
        eta  = torch.sum(torch.exp(-1/self.temperature*(costs - beta)))

        ## 3. Get importance sampling weight
        sampling_weights = 1/eta * torch.exp(-1/self.temperature * (costs - beta))

        ## 4. Get action sequence using weighted average
        self.last_controls += torch.sum(torch.mul(sampling_weights.view(-1,1,1), control_noise), dim=0)

        ## 5. Return first action in sequence and update self.last_controls
        u = self.last_controls[0]
        self.step()

        return u
        
    def step(self):
        '''Updates self.last_controls to warm start next action sequence.'''
        # Shift controls by one
        self.last_controls = self.last_controls[1:]

        # Initialize last control to be the same as the last in the sequence
        self.last_controls = torch.cat([self.last_controls, self.last_controls[[-1]]], dim=0)

    def viz():
        pass

def main():

    # Initialize model and model parameters
    L = 1.0
    dt = 0.1
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
    # costmap[map_height_third:2*map_height_third, map_width_third:2*map_width_third] = 3 + np.random.randn(map_height_third, map_width_third)
    costmap[map_height_third:, map_width_third:] = 3 + np.random.randn(2*map_height_third+2, 2*map_width_third+2)

    costmap = torch.from_numpy(costmap)
    
    goal = torch.Tensor([9.0, 9.0]).cuda()
    # Initialize CostFunction object
    cost_fn = CostFunction(costmap, map_params, goal)

    # Initialize controller
    num_samples = 1024
    num_timesteps = int(2.5*1/dt)
    control_params = {
        'sys_noise': torch.Tensor([1, 0.1]),
        'temperature': 0.01
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

    benchmark_time = []

    achieved_goal = False
    while (iter < max_iters) and (not achieved_goal):
        fig.suptitle(f'MPPI demo. Step {iter}')
        print("---")
        print(f"Iteration: {iter}")
        start = timer()
        u = controller.get_control(x)
        end = timer()
        benchmark_time.append(end-start)
        print(timedelta(seconds=end-start))
        print(f"State is: ")
        print(x)
        print(f"Chosen control is: ")
        print(u)
        x = model.forward(x, u)

        x_hist.append(x)
        u_hist.append(u)

        if torch.linalg.vector_norm(x[:2] - goal) <= 0.2:
            achieved_goal=True

        pos_x = [x_hist[i][0].cpu().item() for i in range(len(x_hist))]
        pos_y = [x_hist[i][1].cpu().item() for i in range(len(x_hist))]
        grid_pos,_ = cost_fn.world_to_grid(torch.cat([torch.Tensor(pos_x).view(-1,1), torch.Tensor(pos_y).view(-1,1)], dim=1))
        x_vals = grid_pos[:,1].cpu().numpy()
        y_vals = grid_pos[:,0].cpu().numpy()

        # import pdb;pdb.set_trace()

        path_subplot.clear()
        path_subplot.grid()
        path_subplot.set_xlim([0, costmap.shape[0]])
        path_subplot.set_ylim([0, costmap.shape[1]])
        path_subplot.set_xlabel("X position")
        path_subplot.set_ylabel("Y position")
        path_subplot.imshow(costmap.cpu().numpy())
        path_subplot.plot(x_vals, y_vals, c='blue', label="Robot's path")
        path_subplot.scatter(x_vals[0], y_vals[0], c='red', marker='o', label="Start Position")
        path_subplot.scatter(x_vals[iter], y_vals[iter], c='green', marker='x', label="Current Position")
        path_subplot.scatter(int((goal[0]-origin[0])/resolution), int((goal[1]-origin[1])/resolution), c='cyan', marker='s', label="Goal Position")
        path_subplot.legend()
        if iter == 0:
            plt.pause(10)
        plt.pause(dt)

        iter += 1
    
    # Plot final result until it is manually closed
    path_subplot.clear()
    path_subplot.grid()
    path_subplot.set_xlim([0, costmap.shape[0]])
    path_subplot.set_ylim([0, costmap.shape[1]])
    path_subplot.set_xlabel("X position")
    path_subplot.set_ylabel("Y position")
    path_subplot.imshow(costmap.cpu().numpy())
    path_subplot.plot(x_vals, y_vals, c='blue', label="Robot's path")
    path_subplot.scatter(x_vals[0], y_vals[0], c='red', marker='o', label="Start Position")
    path_subplot.scatter(x_vals[iter], y_vals[iter], c='green', marker='x', label="Current Position")
    path_subplot.scatter(int((goal[0]-origin[0])/resolution), int((goal[1]-origin[1])/resolution), c='cyan', marker='s', label="Goal Position")
    path_subplot.legend()
    plt.show()


if __name__=="__main__":
    main()