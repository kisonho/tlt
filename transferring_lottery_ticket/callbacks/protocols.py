from typing import Protocol

import abc


class Steppable(Protocol):
    @abc.abstractmethod
    def step(self) -> None:
        pass
