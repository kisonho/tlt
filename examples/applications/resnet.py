from typing import Optional, OrderedDict, Union

import logging, torch, torchvision, torchmanager
from torchvision.models import ResNet

def resnet50(pretrained_model_path: Optional[str], num_classes: int, input_channels: int=3, init_conv1_fc: Optional[str] = None) -> ResNet:
    '''
    Load teacher model and student model for classification

    - Parameters:
        - pretrained_model_path: An optional `str` of pretrained model
        - num_classes: An `int` of number of total classes
        - input_channels: An `int` of number of input channels for the model
        - init_conv1_fc: An optional `str` directory of checkpoint with initialized conv1 and fc
    - Returns: A transfered `ResNet`
    '''
    # initialize model
    model: Union[OrderedDict, ResNet] = torchvision.models.resnet50(pretrained=True) if pretrained_model_path is None else torch.load(pretrained_model_path)

    # reset model layers
    if isinstance(model, torch.nn.Module):
        model.conv1 = torch.nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = torch.nn.Identity() # type: ignore
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    else:
        assert pretrained_model_path is not None, "[Model Error]: The pretrained model must not be None to load the checkpoint."
        ckpt = torchmanager.train.Checkpoint.from_saved(pretrained_model_path)
        assert isinstance(ckpt.model, ResNet), "[Model Error]: The loaded model is not a ResNet model."
        model = ckpt.model

    # load conv1 and fc
    if init_conv1_fc is not None:
        # load init checkpoint
        init_model: Union[OrderedDict, ResNet] = torch.load(init_conv1_fc)

        # load conv1 and fc
        if isinstance(init_model, torch.nn.Module):
            model.conv1 = init_model.conv1
            model.fc = init_model.fc
        else: 
            init_ckpt = torchmanager.train.Checkpoint.from_saved(init_conv1_fc)
            assert type(init_ckpt.model) is type(model), "[Model Error]: The initialize model type is different from target one."
            model.conv1 = init_ckpt.model.conv1
            model.fc = init_ckpt.model.fc

    # print model
    logging.info("---------------------------------------")
    logging.info(model)
    logging.info("---------------------------------------")
    return model