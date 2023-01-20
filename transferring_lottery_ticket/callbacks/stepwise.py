from torchmanager.callbacks import *  # type: ignore

from torchmanager_core.typing import Any

from .protocols import Steppable


class Stepwise(Callback):
    """
    The callback to step `Steppable`

    * extends: `torchmanager.callbacks.Callback`

    - Properties:
        - scheduler: A object that performs `Steppable` protocol to update
    """

    # properties
    scheduler: Steppable

    def __init__(self, scheduler: Steppable) -> None:
        """
        Constructor

        - Parameters:
            - scheduler: A object that performs `Steppable` protocol to update
        """
        super().__init__()
        self.scheduler = scheduler

    def on_epoch_end(self, *args: Any, **kwargs: Any) -> None:
        self.scheduler.step()
