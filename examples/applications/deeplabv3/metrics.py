import torch
from torchmanager.metrics import * # type: ignore

class SegmentationMeanAccuracy(ConfusionMetrics):
    """Segmentation mean accuracy"""
    def __init__(self, num_classes: int) -> None:
        super().__init__(num_classes)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # argmax for input
        input = input.argmax(1).to(target.dtype)
        
        # calculate mean accuracy
        hist = super().forward(input, target)
        acc_cls = torch.diag(hist) / hist.sum(1)
        acc_cls = acc_cls.nanmean()
        return acc_cls

class SegmentationOverallAccuracy(ConfusionMetrics):
    """Segmentation mean accuracy"""
    def __init__(self, num_classes: int) -> None:
        super().__init__(num_classes)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # argmax for input
        input = input.argmax(1).to(target.dtype)
        
        # calculate overall accuracy
        hist = super().forward(input, target)
        acc = torch.diag(hist).sum() / hist.sum()
        return acc