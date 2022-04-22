from abc import ABC, abstractmethod

class Model(ABC):

    @abstractmethod
    def dynamics(self, x, u):
        pass

    @abstractmethod
    def discrete_dynamics(self, x, u, dt):
        pass

    @abstractmethod
    def step(self, x, u):
        pass

    @abstractmethod
    def rollout(self, x, u_list):
        pass

    @property
    @abstractmethod
    def state_dim(self):
        pass

    @property
    @abstractmethod
    def control_dim(self):
        pass