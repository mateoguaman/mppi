import torch

class Model:
    def __init__(self):
        pass
    def dynamics(self, x, u):
        pass
    def discrete_dynamics(self, x, u, dt):
        pass
    def forward(self, x, u):
        pass
    def rollout(self, x, u_list):
        pass
    def state_dim(self):
        pass
    def control_dim(self):
        pass

def KinematicBycicle(Model):
    '''Kinematic model of a car with front wheel steering.

    Assumes no slip, combines rear and front wheels into a single rear and a single front wheel. Assumes center of gravity of the vehicle is in the middle of the rear axle. 

    The state is defined by [x, y, theta, delta], where x and y are the position coordinates, theta is the yaw angle (orientation), and delta is the steering angle. 

    The controls are [v, phi], where v is the forward velocity and phi is the steering angle rate.

    Args:
        L:
            Distance between the front and rear wheels. Expects Float.
        dt: 
            Simulation time step [s]. Expects Float.
        u_max:
            Maximum control limit. Expects Tensor(2,).
        u_min:
            Minimum control limit. Expects Tensor(2,).
    '''
    def __init__(self, L, dt, u_max, u_min):
        super(KinematicBycicle, self).__init__() 
        self.L = L  
        self.dt = dt
        self.u_max = u_max
        self.u_min = u_min

    def dynamics(self, x, u):
        '''Continuous dynamics.
        
        Args:
            x:
                State of the vehicle. Expects Tensor(4,).
            u:
                Input control. Expects Tensor(2,).

        Returns:
            x_dot:
                Time derivative of the state. Tensor(4,).
        '''
        px    = x[0]  # Position in x
        py    = x[1]  # Position in y
        theta = x[2]  # Yaw angle
        delta = x[3]  # Steering angle
        v     = u[0]  # forward velocity
        phi   = u[1]  # Steering angle rate

        x_dot = v*torch.cos(theta)
        y_dot = v*torch.sin(theta)
        omega = v*torch.tan(delta)/self.L
        
        x_dot = torch.Tensor([x_dot, y_dot, omega, phi]) 

        return x_dot

    def discrete_dynamics(self, x, u):
        '''Discrete dynamics for the kinematic bicycle model using RK4.

        Assumes zero-order hold on u

        Args:
            x:
                State of the vehicle. Expects Tensor(4,).
            u:
                Input control. Expects Tensor(2,).

        Returns:
            x_next:
                Next state of the vehicle. Tensor(4,).
        '''

        k1 = self.dynamics(x, u)
        k2 = self.dynamics(x + 1/2*self.dt*k1, u)
        k3 = self.dynamics(x + 1/2*self.dt*k2, u)
        k4 = self.dynamics(x + self.dt*k3, u)

        x_next = x + self.dt/6*(k1 + 2*k2 + 2*k3 + k4)

        return x_next

    def forward(self, x, u):
        '''Simulates one step of the model.

        Enforces control constraints.

        Args:
            x:
                State of the vehicle. Expects Tensor(4,).
            u:
                Input control. Expects Tensor(2,).

        Returns:
            x_next:
                Next state of the vehicle. Tensor(4,).
        '''
        u = torch.max(torch.min(u, self.u_max), self.u_min)

        x_next = self.discrete_dynamics(x, u)

        return x_next

    def rollout(self, x, u_list):
        '''Simulates an open-loop rollout.

        Enforces control constraints.

        Args:
            x:
                Initial state of the vehicle. Expects Tensor(4,).
            u_list:
                List of N input controls. Expects List(Tensor(2,)).
        Returns:
            x_hist:
                List of simulated states of the vehicle.
        '''
        x_hist = []
        x_hist.append(x)

        for k in range(len(u_list)):
            u = torch.max(torch.min(u_list[k], self.u_max), self.u_min)
            x_next = self.discrete_dynamics(x_hist[k], u)
            x_hist.append(x_next)
        
        return x_hist
    
    def state_dim(self):
        '''Returns size of state: 4'''
        return 4

    def control_dim(self):
        '''Returns size of control: 2'''
        return 2