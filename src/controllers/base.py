from abc import ABC, abstractmethod

class Controller(ABC):
    @abstractmethod
    def get_control(self, x):
        pass

