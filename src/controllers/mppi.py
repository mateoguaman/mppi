from datetime import timedelta
from timeit import default_timer as timer
import torch
from torch.distributions import multivariate_normal
import matplotlib.pyplot as plt
import numpy as np


from controllers.base import Controller
from models.kbm import KinematicBycicle
from cost_functions.cost_manager import CostManager
from cost_functions.costmap import Costmap
from cost_functions.goal_cost import GoalCost

class MPPI(Controller):
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
        device:
            String of device ("cpu", "cuda") where to put tensors.
        viz_k:
            Integer, number of MPPI "whiskers" to visualize
    '''
    def __init__(self, model, cost_fn, num_samples, num_timesteps, control_params, device="cpu", viz_k=10):
        self.model = model
        self.cost_fn = cost_fn
        self.K = num_samples
        self.T = num_timesteps
        self.sys_noise = control_params["sys_noise"]  # Equivalent to sigma in [1]
        self.temperature = control_params["temperature"]  # Equivalent to lambda in [1]
        self.device = device
        self.viz_k = viz_k

        self.n = self.model.state_dim()
        self.m = self.model.control_dim()

        self.last_controls = torch.zeros(self.T, self.m).to(self.device)

        mean = torch.zeros(self.m).to(self.device)
        self.sigma = torch.diagflat(self.sys_noise).to(self.device)

        self.control_noise_dist = multivariate_normal.MultivariateNormal(loc=mean, covariance_matrix=self.sigma)

        # For visualization:
        self.viz_rollouts = torch.zeros(self.K, self.T, self.n)

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
        x = x.to(self.device)

        ## 1. Simulate K rollouts and get costs for each rollout
        control_noise = self.control_noise_dist.sample(sample_shape=(self.K, self.T)).to(self.device)

        x_hist = torch.zeros(self.K, self.T, self.n).to(self.device)
        x_hist[:,0,:] = x
        u_hist = self.last_controls.unsqueeze(0).repeat(self.K, 1, 1).to(self.device)

        costs = torch.zeros(self.K).to(self.device)

        for t in range(1, self.T):
            x_prev = x_hist[:,t-1,:]
            u_prev = u_hist[:,t-1,:]
            w_prev = control_noise[:,t-1,:]

            x_t = self.model.step(x_prev, u_prev+w_prev)
            x_hist[:,t,:] = x_t

            control_penalty = self.temperature*(torch.matmul(u_prev.view(-1, self.m, 1).transpose(1,2), torch.linalg.solve(self.sigma.unsqueeze(0), w_prev.view(-1, self.m, 1)))).squeeze().to(self.device)

            costs += (self.cost_fn.stagecost(x_t, u_prev) + control_penalty)

        costs += self.cost_fn.termcost(x_hist[:,-1,:])
        # For visualization, get MPPI "whiskers"
        sorted_indices = torch.argsort(costs)
        top_k_indices = sorted_indices[:self.viz_k]
        self.viz_rollouts = x_hist[top_k_indices, :, :]

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
    ## 1. Instantiate costmap function
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

    ## 2. Add high cost in the middle of cost function to do "barrel test"
    map_height_third = int(height/(3*resolution))
    map_width_third  = int(width/(3*resolution))
    costmap[map_height_third:2*map_height_third, map_width_third:2*map_width_third] = 3 + np.random.randn(map_height_third, map_width_third)

    costmap = torch.from_numpy(costmap)

    ## Initialize Costmap object
    device = "cuda" if torch.cuda.is_available() else "cpu"
    costmap_fn = Costmap(costmap, map_params, device)


    ## 2. Initialize CostFunction object
    goal   = torch.Tensor([9.0, 7.0])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    goalcost_fn = GoalCost(goal, threshold=0.5, fail_cost=10, device=device)


    ## 3. Initialize cost manager
    cost_functions = [costmap_fn, goalcost_fn]
    stagecost_weights = torch.Tensor([1, 0.2])
    termcost_weights = torch.Tensor([1, 5])
    cost_fn = CostManager(cost_functions, stagecost_weights, termcost_weights, device=device)

    # Initialize controller
    num_samples = 1024
    num_timesteps = int(5*1/dt)
    control_params = {
        'sys_noise': torch.Tensor([1, 0.1]),
        'temperature': 0.01
    }
    controller = MPPI(model, cost_fn, num_samples, num_timesteps, control_params, device)

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
        # import pdb;pdb.set_trace()
        whiskers = controller.viz_rollouts
        x = model.step(x, u)

        x_hist.append(x)
        u_hist.append(u)

        # TODO: Check if necessary on real robot
        world_pos = torch.Tensor([x[1], x[0]])
        if torch.linalg.vector_norm(world_pos - goal) <= goalcost_fn.threshold:
            achieved_goal=True

        pos_x = [x_hist[i][0].cpu().item() for i in range(len(x_hist))]
        pos_y = [x_hist[i][1].cpu().item() for i in range(len(x_hist))]
        grid_pos,_ = costmap_fn.world_to_grid(torch.cat([torch.Tensor(pos_x).view(-1,1), torch.Tensor(pos_y).view(-1,1)], dim=1))
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
        path_subplot.plot(x_vals, y_vals, c='blue', label="Robot's path", alpha=0.5)
        path_subplot.scatter(x_vals[0], y_vals[0], c='red', marker='o', label="Start Position")
        path_subplot.scatter(x_vals[iter], y_vals[iter], c='green', marker='x', label="Current Position")
        path_subplot.scatter(int((goal[0]-origin[0])/resolution), int((goal[1]-origin[1])/resolution), c='cyan', marker='s', label="Goal Position")
        
        if iter == 0:
            plt.pause(10)

        for k in range(whiskers.shape[0]):
            traj = whiskers[k]
            traj_x = [traj[i][0].cpu().item() for i in range(traj.shape[0])]
            traj_y = [traj[i][1].cpu().item() for i in range(traj.shape[0])]
            traj_grid_pos,_ = costmap_fn.world_to_grid(torch.cat([torch.Tensor(traj_x).view(-1,1), torch.Tensor(traj_y).view(-1,1)], dim=1))
            traj_x_vals = traj_grid_pos[:,1].cpu().numpy()
            traj_y_vals = traj_grid_pos[:,0].cpu().numpy()
            if k == 0:
                path_subplot.plot(traj_x_vals, traj_y_vals, c='green', label="Chosen path", alpha=0.5)
            else:
                path_subplot.plot(traj_x_vals, traj_y_vals, c='yellow', label=f"Sample path {k}", alpha=0.3)
        path_subplot.legend(loc="upper left")
        plt.pause(dt)

        iter += 1
    if achieved_goal:
        print("\n\n\n=====\nACHIEVED GOAL\n=====\n\n\n")
    else:
        print("\n\n\n=====\nFAILED: REACHED MAX ITERS\n=====\n\n\n")
    # Plot final result until it is manually closed
    fig.suptitle(f'MPPI demo. GOAL COMPLETED')
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
    path_subplot.legend(loc="upper left")
    plt.show()


if __name__=="__main__":
    main()