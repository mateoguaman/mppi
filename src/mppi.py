import torch

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

        self.sys_noise = control_params.sys_noise  # Equivalent to sigma in [1]
        self.temperature = control_params.temperature  # Equivalent to lambda in [1]



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
            sigma = torch.eye(self.m)
            for t in range(1, self.T):
                u_t = self.last_controls[t-1]
                w_t = noise[t-1]
                x_t = self.model.forward(x_hist[t-1], u_t + w_t)
                x_hist.append(x_t)
                cost += self.cost_fn.stage_cost(x_t, u_t) + self.temperature*u_t.view(1, -1)*sigma*w_t.view(-1, 1)

            cost +=  self.cost_fn.term_cost(x_hist[-1])

            costs.append(costs)
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
            self.last_controls[t] += torch.sum(torch.mul(sampling_weights, control_noise[:,t,:])) # TODO: check dimensions

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