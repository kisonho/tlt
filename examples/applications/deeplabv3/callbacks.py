from typing import Any, Dict

import torch
from torchmanager.train import learning_rate
from tlt.callbacks import * # type: ignore

class WarmUpLrScheduler(LrSchedueler):
    """`LrScheduler` with linear warmup"""
    __i: int
    warmup_scheduler: torch.optim.lr_scheduler._LRScheduler
    warmup_steps: int

    def __init__(self, warmup_scheduler: torch.optim.lr_scheduler._LRScheduler, warmup_steps: int, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.__i = 0
        self.warmup_scheduler = warmup_scheduler
        self.warmup_steps = warmup_steps

    def on_batch_end(self, *args, **kwargs) -> None:
        if self.__i < self.warmup_steps:
            self.warmup_scheduler.step()
        else: super().on_batch_end(*args, **kwargs)
        self.__i += 1

    def on_epoch_end(self, epoch: int, *args, **kwargs) -> None:
        if self.__i >= self.warmup_steps:
            super().on_epoch_end(epoch, *args, **kwargs)