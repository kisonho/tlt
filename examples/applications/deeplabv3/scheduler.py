"""
Code modified from https://github.com/cmpark0126/pytorch-polynomial-lr-decay/blob/master/torch_poly_lr_decay/torch_poly_lr_decay.py
"""
import torch
from torch.optim.lr_scheduler import * # type: ignore
from torch.optim.lr_scheduler import _LRScheduler
from typing import List, Optional

class PolynomialLRDecay(_LRScheduler):
    """Polynomial learning rate decay until step reach to max_decay_step
    
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_decay_steps: after this step, we stop decreasing learning rate
        end_learning_rate: scheduler stoping learning rate decay, value of learning rate must be this value
        power: The power of the polynomial.
    """
    base_lrs: List[float]
    optimizer: torch.optim.Optimizer
    
    def __init__(self, optimizer: torch.optim.Optimizer, max_decay_steps: int, end_learning_rate: float = 1e-4, power: float = 1.0) -> None:
        if max_decay_steps <= 1.:
            raise ValueError('max_decay_steps should be greater than 1.')
        self.max_decay_steps = max_decay_steps
        self.end_learning_rate = end_learning_rate
        self.power = power
        self.last_step = 0
        super().__init__(optimizer)

    def get_last_lr(self) -> List[float]:
        return self.get_lr()
        
    def get_lr(self) -> List[float]:
        if self.last_step > self.max_decay_steps:
            return [self.end_learning_rate for _ in self.base_lrs]

        return [(base_lr - self.end_learning_rate) * 
                ((1 - self.last_step / self.max_decay_steps) ** (self.power)) + 
                self.end_learning_rate for base_lr in self.base_lrs]
    
    def step(self, step: Optional[int] = None) -> None:
        if step is None:
            step = self.last_step + 1
        self.last_step = step if step != 0 else 1
        if self.last_step <= self.max_decay_steps:
            decay_lrs = [(base_lr - self.end_learning_rate) * 
                         ((1 - self.last_step / self.max_decay_steps) ** (self.power)) + 
                         self.end_learning_rate for base_lr in self.base_lrs]
            for i, (param_group, lr) in enumerate(zip(self.optimizer.param_groups, decay_lrs)):
                param_group['lr'] = lr
                print(f"Update learning rate in group {i} to: {lr}")