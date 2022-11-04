from enum import Enum
from typing import Optional

import torch, torchvision

class DeepLabV3(torchvision.models.segmentation.deeplabv3.DeepLabV3):
    """The `DeepLabV3` model with torchmanager compatibility wrap, which returns only output `torch.Tensor` named `out`"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = super().forward(x)
        return result['out']

class DeepLabV3Backbone(Enum):
    RESNET50 = str("deeplabv3_resnet50")

    def load(self, pretrained_model: Optional[torch.nn.Module] = None) -> torch.nn.Sequential:
        """Load backbone model"""
        if self == DeepLabV3Backbone.RESNET50:
            # backbone
            resnet50 = torchvision.models.resnet50(True, replace_stride_with_dilation=[False, False, True])
            if pretrained_model is not None:
                resnet50.load_state_dict(pretrained_model.state_dict())

            # initialize model
            model = torch.nn.Sequential()

            # add backbone modules
            for name, m in resnet50.named_children():
                if "avgpool" not in name and "fc" not in name:
                    model.add_module(name, m)
            return model
        else: raise TypeError(f"[Model Error]: Backbone '{self.value}' is not supported.")