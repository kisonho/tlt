from enum import Enum

import logging

from . import datasets

class Datasets(Enum):
    """
    Available Datasets `Enum`
    """
    CIFAR10 = "cifar10"
    CIFAR100 = "cifar100"
    CITYSCAPES = "cityscapes"
    IMAGENET = "imagenet"
    MNIST = "mnist"
    SVHN = "svhn"
    VOC_SEG = "voc_segmentation"

    def load(self, *args, **kwargs) -> datasets.Dataset:
        """
        Load dataset

        - Returns: A target `Dataset`
        """
        if self == Datasets.CIFAR10:
            return datasets.CIFAR10(*args, **kwargs)
        elif self == Datasets.CIFAR100:
            return datasets.CIFAR100(*args, **kwargs)
        elif self == Datasets.CITYSCAPES:
            return datasets.Cityscapes(*args, **kwargs)
        elif self == Datasets.IMAGENET:
            return datasets.ImageNet(*args, **kwargs)
        elif self == Datasets.MNIST:
            return datasets.MNIST(*args, **kwargs)
        elif self == Datasets.SVHN:
            return datasets.SVHN(*args, **kwargs)
        elif self == Datasets.VOC_SEG:
            return datasets.VOCSegmentation(*args, **kwargs)
        else:
            raise ValueError("The dataset \'{}\' is currently not available.")

def load(dataset: str, batch_size: int = 128, **kwargs) -> datasets.Dataset:
    """
    Load a dataset by string

    - Parameters:
        - dataset: the `str` of dataset
        - batch_size: An `int` of batch size
    - Returns: A `Dataset`
    """
    d = Datasets(dataset)
    logging.info("Load dataset {}: batch_size={}".format(dataset, batch_size))
    return d.load(batch_size, **kwargs)