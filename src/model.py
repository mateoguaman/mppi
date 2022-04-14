import torch
import numpy as np
import matplotlib.pyplot as plt

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

class KinematicBycicle(Model):
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
        self.u_max = u_max.cuda()
        self.u_min = u_min.cuda()

    def dynamics(self, x, u):
        '''Continuous dynamics.
        
        Args:
            x:
                State of the vehicle. Expects Tensor(4,) or Tensor(K, 4).
            u:
                Input control. Expects Tensor(2,) or Tensor(K, 2).

        Returns:
            x_dot:
                Time derivative of the state. Tensor(4,) or Tensor(K, 4).
        '''
        if len(x.shape) == 1:
            px    = x[0]  # Position in x
            py    = x[1]  # Position in y
            theta = x[2]  # Yaw angle
            delta = x[3]  # Steering angle
            v     = u[0]  # forward velocity
            phi   = u[1]  # Steering angle rate

            px_dot = v*torch.cos(theta)
            py_dot = v*torch.sin(theta)
            omega = v*torch.tan(delta)/self.L
            
            x_dot = torch.Tensor([px_dot, py_dot, omega, phi]).cuda()
        else:
            px_dot = u[:,[0]]*torch.cos(x[:,[2]])
            py_dot = u[:,[0]]*torch.sin(x[:,[2]])
            omega  = u[:,[0]]*torch.tan(x[:,[3]])/self.L

            x_dot  = torch.cat([px_dot, py_dot, omega, u[:,[1]]], dim=1).cuda()

        return x_dot

    def discrete_dynamics(self, x, u):
        '''Discrete dynamics for the kinematic bicycle model using RK4.

        Assumes zero-order hold on u

        Args:
            x:
                State of the vehicle. Expects Tensor(4,) or Tensor(K, 4).
            u:
                Input control. Expects Tensor(2,) or Tensor(K, 2).

        Returns:
            x_next:
                Next state of the vehicle. Tensor(4,) or Tensor(K, 4).
        '''
        x = x.cuda()
        u = u.cuda()

        k1 = self.dynamics(x, u)
        k2 = self.dynamics(x + 1/2*self.dt*k1, u)
        k3 = self.dynamics(x + 1/2*self.dt*k2, u)
        k4 = self.dynamics(x + self.dt*k3, u)

        x_next = (x + self.dt/6*(k1 + 2*k2 + 2*k3 + k4)).cuda()

        return x_next


    def forward(self, x, u):
        '''Simulates one step of the model.

        Enforces control constraints.

        Args:
            x:
                State of the vehicle. Expects Tensor(4,) or Tensor(K, 4).
            u:
                Input control. Expects Tensor(2,) or Tensor(K, 2).

        Returns:
            x_next:
                Next state of the vehicle. Tensor(4,) or Tensor(K, 4).
        '''
        u = torch.max(torch.min(u, self.u_max), self.u_min).cuda()

        x_next = self.discrete_dynamics(x, u).cuda()

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

def main():
    env = np.zeros((100, 100))

    x0 = torch.Tensor([2, 0, 1.57, 1])

    # Initialize model and model parameters
    L = 1.0
    dt = 0.01
    u_max = torch.Tensor([10, torch.pi/2])
    u_min = torch.Tensor([0, -torch.pi/2])

    model = KinematicBycicle(L, dt, u_max, u_min)

    # Test functions
    u = torch.Tensor([4, -0.1])
    x_dot = model.dynamics(x0, u)
    print("=====")
    print(f"Testing model dynamics: ")
    print(f"Input. x: {x0}, u: {u}")
    print(f"x_dot: {x_dot}")


    x_batch = torch.Tensor([
        [0.0, 0.0, 1.0, 0.1],
        [2.0, 0.0, 1.57, 1.0],
        [3.0, 2.0, 0.1, 0.2],
        [5.0, 7.0, 1.3, 0.5],
        [3.1, 2.2, 1.1, 0.3]
    ])

    u_batch = torch.Tensor([
        [1.2, 0.0],
        [4.0, -0.1],
        [2.2, 1.0],
        [0.0, 1.1],
        [1.2, 3.4]
    ])

    x_dot = model.dynamics(x_batch, u_batch)
    print("=====")
    print(f"Testing batched model dynamics: ")
    print("Input state:")
    print(x_batch)
    print("Input control:")
    print(u_batch)
    print("x_dot: ")
    print(x_dot)

    x_next = model.discrete_dynamics(x0, u)
    print("---")
    print(f"Testing discrete dynamics: ")
    print(f"x_next: {x_next}")

    x_next = model.discrete_dynamics(x_batch, u_batch)
    print("---")
    print(f"Testing batched discrete dynamics: ")
    print("Input state:")
    print(x_batch)
    print("Input control:")
    print(u_batch)
    print("x_next: ")
    print(x_next)

    x_next = model.forward(x0, u)
    print("---")
    print(f"Testing forward: ")
    print(f"x_next: {x_next}")

    x_next = model.forward(x_batch, u_batch)
    print("---")
    print("Testing batched forward")
    print(x_batch)
    print("Input control:")
    print(u_batch)
    print("x_next: ")
    print(x_next)

    N = 200
    u_list = [u]*N
    times  = torch.cumsum(torch.Tensor([dt]*N), dim=0)
    x_hist = model.rollout(x0, u_list)
    print("---")
    print("Testing rollout: ")
    print(f"x_hist: {x_hist}")

    n = model.state_dim()
    print("---")
    print("Testing state_dim: ")
    print(f"n: {n}")

    m = model.control_dim()
    print("---")
    print("Testing control_dim: ")
    print(f"m: {m}")

    pos_x = [x_hist[i][0] for i in range(len(x_hist))]
    pos_y = [x_hist[i][1] for i in range(len(x_hist))]

    fig = plt.figure()
    fig.suptitle('Kinematic Bicycle Model demo')
    path_subplot = fig.add_subplot(1,1,1)

    for i in range(1, len(pos_x)):

        x_vals = pos_x[:i]
        y_vals = pos_y[:i]

        path_subplot.clear()
        path_subplot.grid()
        path_subplot.set_xlim([-5, 5])
        path_subplot.set_ylim([-5, 5])
        path_subplot.plot(x_vals, y_vals, c='blue')
        path_subplot.scatter(x_vals[0], y_vals[0], c='red', marker='o')
        path_subplot.scatter(x_vals[i-1], y_vals[i-1], c='green', marker='x')
        plt.pause(dt)

if  __name__=="__main__":
    main()