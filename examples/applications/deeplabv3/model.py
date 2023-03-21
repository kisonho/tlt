import logging, torch, torchvision
from enum import Enum
from torch.nn import functional as F
from torchvision.models._utils import IntermediateLayerGetter
from typing import Callable, Optional


class DeepLabV3(torchvision.models.segmentation.deeplabv3.DeepLabV3):
    """Restructured DeepLabV3 that returns only the output `torch.Tensor` instead of a dictionary"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape[-2:]
        x = self.backbone(x)["out"]
        x = self.classifier(x)
        y = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        return y


class DeepLabV3Backbone(Enum):
    RESNET50 = str("deeplabv3_resnet50")

    def load(self, pretrained_model: Optional[torch.nn.Module] = None) -> torch.nn.Sequential:
        """
        Load backbone model

        - Parameters:
            - pretrained_model: An `Optional` pretrained `torch.nn.Module`
        - Returns: A `torch.nn.Sequential` pretrained model
        """
        if self == DeepLabV3Backbone.RESNET50:
            # backbone
            resnet50 = torchvision.models.resnet50(True, replace_stride_with_dilation=[False, True, True])
            if pretrained_model is not None:
                resnet50.load_state_dict(pretrained_model.state_dict())

            # initialize model
            model = torch.nn.Sequential()

            # add backbone modules
            for name, m in resnet50.named_children():
                if "avgpool" not in name and "fc" not in name:
                    model.add_module(name, m)
            return model
        else:
            raise TypeError(f"[Model Error]: Backbone '{self.value}' is not supported.")


def deeplabv3(backbone: DeepLabV3Backbone = DeepLabV3Backbone.RESNET50, pretrained_model: Optional[torch.nn.Module] = None, num_classes: int = 21) -> DeepLabV3:
    """
    Load deeplabv3 with mobilenetv3 (large) for classification

    - Parameters:
        - backbone: A required type of `DeepLabV3Bacbkone`
        - pretrained_model: An `Optional` pretrained `torch.nn.Module`
        - num_classes: An `int` of number of classes
    - Returns: A `torch.nn.Module` of deeplabv3
    """
    # initialize model
    model_fn: Callable[..., torchvision.models.segmentation.DeepLabV3] = getattr(torchvision.models.segmentation, backbone.value)
    model = model_fn(pretrained=True)

    # transfer to current num classes
    if num_classes != 21:
        classifier_layers = list(model.classifier.children())[:-1]
        classifier_layers.append(torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1)))
        model.classifier = torch.nn.Sequential(*classifier_layers)

    # load backbone
    backbone_model = backbone.load(pretrained_model)
    if backbone == DeepLabV3Backbone.RESNET50:
        backbone_model = IntermediateLayerGetter(backbone_model, {"layer4": "out"})
    model.backbone = backbone_model

    # initialize model
    model = DeepLabV3(model.backbone, model.classifier)

    # print model
    logging.info("---------------------------------------")
    logging.info(model)
    logging.info("---------------------------------------")
    return model
