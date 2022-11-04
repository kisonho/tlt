from torchmanager_core import abc
from torchmanager_core.typing import Generic, Optional, TypeVar, Union
from torchmanager_core.view import logging

from .protocols import DynamicPruning

PruningMethod = TypeVar("PruningMethod", bound=DynamicPruning)


class _PruningScheduler(abc.ABC):
    """
    The basic pruning scheduler

    * abstract class
    * implements: `..callbacks.protocols.Steppable`

    - Properties:
        - current_step: An `int` of current step
    """

    # properties
    __current_step: int

    @property
    def current_step(self) -> int:
        return self.__current_step

    def __init__(self) -> None:
        """Constructor"""
        super().__init__()
        self.__current_step = 0

    def step(self) -> None:
        """The scheduler step method to update pruning ratio"""
        # update amount
        amount = self.update_amount()

        # compute mask
        if amount is not None:
            self.update_mask(amount)

        # increment step
        self.__current_step += 1

    @abc.abstractmethod
    def update_amount(self) -> Optional[Union[int, float]]:
        """
        Function to update pruning amount

        - Returns: An `int` or `float` of pruning amount
        """
        raise NotImplementedError

    @abc.abstractmethod
    def update_mask(self, amount: Union[int, float]) -> None:
        """
        Function to update pruning mask

        - Parameters:
            - amount: An `int` or `float` of pruning amount
        """
        raise NotImplementedError


class _DynamicPruningScheduler(_PruningScheduler, abc.ABC, Generic[PruningMethod]):
    """
    The dynamic pruning scheduler

    * abstract class
    * extends: `_PruningScheduler`

    - Properties:
        - amount: An `int` or `float` of pruning amount
        - method: A method that performs to `DynamicPruning` to update masks
    """

    __method: PruningMethod

    @property
    def amount(self) -> Union[int, float]:
        return self.method.amount

    @property
    def method(self) -> PruningMethod:
        return self.__method

    def __init__(self, method: PruningMethod) -> None:
        """
        Constructor

        - Parameters:
            - method: A pruning method that implements `DynamicPruningMethod`
        """
        super().__init__()
        self.__method = method

    def update_mask(self, amount: Union[int, float]) -> None:
        assert amount >= 0, "[Pruning Error]: Amount should be non-negative, got {}.".format(amount)
        self.method.update_mask(amount)
        logging.info(f"Adjusted pruning ratio to {amount}")


class ConstantScheduler(_DynamicPruningScheduler[PruningMethod], Generic[PruningMethod]):
    """
    The scheduler that returns same pruning ratio for each step

    * extends: `_DynamicPruningScheduler`

    - Parameters:
        - amount: An `int` or `float` of pruning amount
    """

    # parameters
    _initial_step: int

    @property
    def amount(self) -> Union[int, float]:
        return self.method.amount

    def __init__(self, method: PruningMethod, initial_step: int = 0) -> None:
        """
        Constructor

        - Parameters:
            - method: A pruning method that implements `DynamicPruningMethod`
            - initial_step: An `int` index of initial step of pruning
        """
        super().__init__(method)
        self._initial_step = initial_step

    def update_amount(self) -> Optional[Union[int, float]]:
        return self.amount if self.current_step >= self._initial_step else None


class DPFScheduler(_DynamicPruningScheduler[PruningMethod], Generic[PruningMethod]):
    """
    Pruning scheduler for DPF

    * extends: `_DynamicPruningScheduler`

    * Modified according to: https://github.com/INCHEON-CHO/Dynamic_Model_Pruning_with_Feedback/blob/f63cf144d13fee3f5f3e57d7c647b8a698c3cfd1/main.py#L219
    """

    __initial_amount: Union[int, float]
    __target_amount: Union[int, float]
    __final_step: int
    __freq: int

    def __init__(self, method: PruningMethod, target_amount: Union[int, float], final_step: int, freq: int = 1) -> None:
        super().__init__(method)
        self.__initial_amount = method.amount
        self.__target_amount = target_amount
        self.__final_step = final_step
        self.__freq = freq
        assert target_amount >= 0, f"[Schedule Error]: target_amount must be non-negative, got {target_amount}."
        assert type(self.__initial_amount) is type(self.__target_amount), f"[Schedule Error]: The initial and target pruning amount type must be the same, got {type(self.__initial_amount)} and {type(self.__target_amount)}."
        assert final_step >= 0, f"[Schedule Error]: The final_step must be non-negative, got {final_step}."
        assert freq > 0, f"[Schedule Error]: The freq must be positive, got {freq}."

    def update_amount(self) -> Optional[Union[int, float]]:
        if self.current_step % self.__freq == 0:
            if self.current_step < self.__final_step:
                return type(self.__target_amount)(self.__target_amount - (self.__target_amount - self.__initial_amount) * (1 - (self.current_step + 1) / self.__final_step) ** 3)
            else:
                return self.__target_amount
        else:
            return None
