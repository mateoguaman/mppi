from abc import ABC, abstractmethod

class CostFunction(ABC):
    @abstractmethod
    def stagecost(self, state, action):
        pass

    @abstractmethod
    def termcost(self, state):
        pass
    