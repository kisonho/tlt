# Transferring Lottery Tickets in Computer Vision Models: a Dynamic Pruning Approach

## Pre-request
- Python >= 3.8
- [torchmanager](https://github.com/kisonho/torchmanager.git)

## Transfer a model with dynamic pruning method
1. Define and initialize a PyTorch model:
```
import torch

pretrained_model: torch.nn.Module = torch.load(...)
```

2. Compile a `torchmanager.Manager` with the pretrained model:
```
import torchmanager
from typing import Dict

optimizer: torch.optim.Optimizer = ...
loss_fn: torchmanager.losses.Loss = ...
metric_fns: Dict[str, torchmanager.metrics.Metric] = ...
manager = torchmanager.Manager(pretrained_model, optimizer=optimizer, loss_fn=loss_fn, metrics=metric_fns)
```

3. Apply pruning method to the pretrained model:
```
import transferring_lottery_ticket as prune

pruning_method = prune.TransferringLotteryTicket.apply(pretrained_model, amount=0.8, start=2, end=None, is_dynamic=True)
```

4. Apply pruning scheduler to the pruning method for dynamic pruning, masks will be updated according to the schedule:
```
pruning_scheduler = prune.schedulers.ConstantScheduler(pruning_method)
```

5. Add pruning scheduler to callback list and train with `fit` method in `torchmanager.Manager`
```
from transferring_lottery_ticket import callbacks

pruning_scheduler_callback = callbacks.PruningRatio(pruning_scheduler)
callbacks = [pruning_scheduler_callback, ...]

dataset = ...
epochs = ...
manager.fit(dataset, epochs, ..., callbacks=callbacks)
```