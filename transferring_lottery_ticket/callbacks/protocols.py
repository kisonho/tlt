from torchmanager_core import abc
from torchmanager_core.typing import Protocol


class Steppable(Protocol):
    @abc.abstractmethod
    def step(self) -> None:
        pass
