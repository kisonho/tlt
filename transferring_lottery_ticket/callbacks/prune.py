from torchmanager.callbacks import *  # type: ignore

from typing import Any

from .protocols import Steppable


class PruningRatio(Callback):
    """
    The callback to step `_PruningScheduler`

    * extends: `torchmanager.callbacks.Callback`

    - Properties:
        - scheduler: A `_PruningScheduler` to update the pruning ratio and mask
    """

    # properties
    scheduler: Steppable

    def __init__(self, scheduler: Steppable) -> None:
        """
        Constructor

        - Parameters:
            - scheduler: A pruning scheduler that performs to `Steppable` protocol to update the pruning ratio and mask
        """
        super().__init__()
        self.scheduler = scheduler

    def on_epoch_end(self, *args: Any, **kwargs: Any) -> None:
        self.scheduler.step()
