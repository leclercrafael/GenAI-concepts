import abc
import typing

class BaseAgent(abc.ABC):

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def run(self)->None:
        pass

