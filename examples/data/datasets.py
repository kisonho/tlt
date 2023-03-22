"""
Code edited from: https://www.programcreek.com/python/?code=MrtnMndt%2FDeep_Openset_Recognition_through_Uncertainty%2FDeep_Openset_Recognition_through_Uncertainty-master%2Flib%2FDatasets%2Fdatasets.py
"""
from typing import Optional, Tuple

import abc, os, torchvision
from torch.utils.data import Dataset as TorchDataset, DataLoader, Subset
from torchvision.datasets import * # type: ignore

from . import transforms
from .ade20k import DatasetOptions as ADE20KOptions, TrainDataset as ADE20KTrainingDataset, ValDataset as ADE20KValidationDataset

class Dataset(abc.ABC):
    """
    The basic abstract Dataset class

    - Parameters:
        - num_classes: An `int` of total classes
        - train_loader: The `DataLoader` for training dataset
        - test_loader: The `DataLoader` for testing dataset
        - val_loader: The `DataLoader` for validation dataset
    """
    _trainset: TorchDataset
    _testset: TorchDataset
    train_loader: DataLoader
    test_loader: DataLoader
    val_loader: DataLoader
    steps_per_epoch: int

    @property
    @abc.abstractmethod
    def input_channels(self) -> int:
        return NotImplemented

    @property
    @abc.abstractmethod
    def num_classes(self) -> int:
        return NotImplemented

    def __init__(self, batch_size: int, root_dir: str = os.path.normpath("~/Documents/Data"), workers: Optional[int] = os.cpu_count()) -> None:
        """
        Constructor

        - Parameters:
            - batch_size: An `int` of dataset batch size
            - normalize: A `bool` flag of if normalize the image
            - is_gpu: A `bool` of if loading dataset into GPU
            - workers: An `int` of multi-threads
        """
        workers = 4 if workers is None else workers
        self._trainset, self._testset = self._get_dataset(root_dir)
        self.train_loader, self.val_loader, self.test_loader = self._get_dataset_loader(batch_size, workers)
        self.steps_per_epoch = len(self.train_loader)

    @abc.abstractmethod
    def _get_dataset(self, dataset_dir: str = os.path.normpath("~/Documents/Data/")) -> Tuple[TorchDataset, TorchDataset]:
        """
        Get datasets

        - Returns: A `tuple` of `torchvision.datasets.DatasetFolder` for training dataset and `torchvision.datasets.DatasetFolder` for testing dataset
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _get_dataset_loader(self, batch_size: int = 128, workers: Optional[int] = os.cpu_count()) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Get the dataset loader

        - Parameters:
            - batch_size: An `int` of dataset batch size\
            - workers: An `int` of multi-threads
        - Returns: A `tuple` of three `DataLoader` objects for training, validation, and testing loader
        """
        raise NotImplementedError
    
class ADE20K(Dataset):
    input_channels: int = 3
    num_classes: int = 150

    def _get_dataset(self, dataset_dir: str = os.path.normpath("~/Documents/Data/")) -> Tuple[ADE20KTrainingDataset, ADE20KValidationDataset]:
        # initialize odgt path and options
        training_odgt = os.path.join(dataset_dir, "training.odgt")
        testing_odgt = os.path.join(dataset_dir, "testing.odgt")
        ade20k_options = ADE20KOptions((300, 375, 450, 525, 600), 1000, 8, 8)
        training_dataset = ADE20KTrainingDataset(dataset_dir, training_odgt, ade20k_options)
        testing_dataset = ADE20KValidationDataset(dataset_dir, testing_odgt, ade20k_options)
        return training_dataset, testing_dataset

    def _get_dataset_loader(self, batch_size: int = 128, workers: Optional[int] = os.cpu_count()) -> Tuple[DataLoader, DataLoader, DataLoader]:
        workers = 4 if workers is None else workers
        train_loader = DataLoader(self._trainset, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)
        val_loader = DataLoader(self._testset, batch_size=batch_size, shuffle=False, num_workers=workers)
        test_loader = DataLoader(self._testset, batch_size=batch_size, shuffle=False, num_workers=workers)
        return train_loader, val_loader, test_loader

class CIFAR10(Dataset):
    """A Cifar 10 dataset"""
    input_channels: int = 3
    num_classes: int = 10

    def _get_dataset(self, dataset_dir: str = os.path.normpath("~/Documents/Data/")) -> Tuple[torchvision.datasets.CIFAR10, torchvision.datasets.CIFAR10]:
        # initialize
        dataset_dir = os.path.join(dataset_dir, 'CIFAR10')

        # get transforms
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        train_transforms = transforms.Compose([
            transforms.RandomCrop(32, 4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
            ])
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            normalize
            ])
        
        trainset = torchvision.datasets.CIFAR10(os.path.join(dataset_dir, "train"), train=True, transform=train_transforms, target_transform=None, download=True)
        testset = torchvision.datasets.CIFAR10(os.path.join(dataset_dir, "test"), train=False, transform=test_transforms, target_transform=None, download=True)
        return trainset, testset

    def _get_dataset_loader(self, batch_size: int = 128, workers: Optional[int] = os.cpu_count()) -> Tuple[DataLoader, DataLoader, DataLoader]:
        workers = 4 if workers is None else workers
        train_set = Subset(self._trainset, list(range(45000)))
        val_set = Subset(self._trainset, list(range(45000, 50000)))
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=workers)
        test_loader = DataLoader(self._testset, batch_size=batch_size, shuffle=False, num_workers=workers)
        return train_loader, val_loader, test_loader

class CIFAR100(Dataset):
    """A Cifar 100 dataset"""
    input_channels: int = 3
    num_classes: int = 100

    def _get_dataset(self, dataset_dir: str = os.path.normpath("~/Documents/Data/")) -> Tuple[torchvision.datasets.CIFAR100, torchvision.datasets.CIFAR100]:
        dataset_dir = os.path.join(dataset_dir, 'CIFAR100')
        normalize = transforms.Normalize(mean=[0.5071, 0.4866, 0.4409], std=[0.2009, 0.1984, 0.2023])
        train_transforms = transforms.Compose([
            transforms.RandomCrop(32, 4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
            ])
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            normalize
            ])
        trainset = torchvision.datasets.CIFAR100(os.path.join(dataset_dir, "train"), train=True, transform=train_transforms, target_transform=None, download=True)
        testset = torchvision.datasets.CIFAR100(os.path.join(dataset_dir, "test"), train=False, transform=test_transforms, target_transform=None, download=True)
        return trainset, testset

    def _get_dataset_loader(self, batch_size: int = 128, workers: Optional[int] = os.cpu_count()) -> Tuple[DataLoader, DataLoader, DataLoader]:
        workers = 4 if workers is None else workers
        train_set = Subset(self._trainset, list(range(45000)))
        val_set = Subset(self._trainset, list(range(45000, 50000)))
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=workers)
        test_loader = DataLoader(self._testset, batch_size=batch_size, shuffle=False, num_workers=workers)
        return train_loader, val_loader, test_loader

class Cityscapes(Dataset):
    """A VOC Segmentation Dataset"""
    input_channels: int = 3
    num_classes: int = 20
    _valset: torchvision.datasets.Cityscapes

    def _get_dataset(self, dataset_dir: str = os.path.normpath("~/Documents/Data/")) -> Tuple[TorchDataset, TorchDataset]:
        normalize = transforms.SegmentationNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        train_transforms = transforms.SegmentationCompose([
            transforms.SegmentationRandomCrop(size=(769, 769)),
            transforms.SegmentationColorJitter(brightness=0.5, contrast=0.5, saturation=0.5), # type: ignore
            transforms.SegmentationRandomHorizontalFlip(),
            transforms.SegmentationToTensor(),
            transforms.MappingLabel(),
            transforms.SegmentationReshape(),
            normalize
        ])
        test_transforms = transforms.SegmentationCompose([
            # transforms.SegmentationRandomCrop(size=(769, 769)),
            transforms.SegmentationToTensor(),
            transforms.MappingLabel(),
            transforms.SegmentationReshape(),
            normalize
            ])
        trainset = torchvision.datasets.Cityscapes(dataset_dir, transforms=train_transforms)
        self._valset = torchvision.datasets.Cityscapes(dataset_dir, split="val", transforms=test_transforms)
        testset = torchvision.datasets.Cityscapes(dataset_dir, split="val", transforms=test_transforms)
        return trainset, testset

    def _get_dataset_loader(self, batch_size: int = 16, workers: Optional[int] = os.cpu_count()) -> Tuple[DataLoader, DataLoader, DataLoader]:
        workers = 4 if workers is None else workers
        train_loader = DataLoader(self._trainset, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)
        val_loader = DataLoader(self._valset, batch_size=batch_size, shuffle=False, num_workers=workers)
        test_loader = DataLoader(self._testset, batch_size=batch_size, shuffle=False, num_workers=workers)
        return train_loader, val_loader, test_loader

class ImageNet(Dataset):
    """
    ImageNet dataset consisting of a large amount (more than a million images) images with varying
    resolution for 1000 different classes.
    Typically the smaller side of the images is rescaled to 256 and quadratic random 224x224 crops
    are taken. Dataloader implemented with torchvision.datasets.ImageNet.

    Parameters:
        args (dict): Dictionary of (command line) arguments.
            Needs to contain batch_size (int) and workers(int).
        is_gpu (bool): True if CUDA is enabled.
            Sets value of pin_memory in DataLoader.
    Attributes:
        train_transforms (torchvision.transforms): Composition of transforms
            including conversion to Tensor, horizontal flips, random
            translations of up to 10% in each direction and normalization.
        val_transforms (torchvision.transforms): Composition of transforms
            including conversion to Tensor and normalization.
        trainset (torch.utils.data.TensorDataset): Training set wrapper.
        valset (torch.utils.data.TensorDataset): Validation set wrapper.
        train_loader (torch.utils.data.DataLoader): Training set loader with shuffling.
        val_loader (torch.utils.data.DataLoader): Validation set loader.
    """
    num_classes: int = 1000
    input_channels: int = 3

    def _get_dataset(self, dataset_dir: str = os.path.normpath('~/Documents/Data/')) -> Tuple[torchvision.datasets.ImageNet, torchvision.datasets.ImageNet]:
        """
        Uses torchvision.datasets.ImageNet to load dataset.
        Downloads dataset if doesn't exist already.
        Returns:
             torch.utils.data.TensorDataset: trainset, valset
        """
        dataset_dir = os.path.join(dataset_dir, 'ImageNet')
        image_size = 224
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        train_transforms = transforms.Compose([
            transforms.Resize(image_size + 32),
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        test_transforms = transforms.Compose([
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize
        ])
        trainset = torchvision.datasets.ImageNet(os.path.join(dataset_dir, 'train'), split='train', transform=train_transforms, target_transform=None)
        valset = torchvision.datasets.ImageNet(os.path.join(dataset_dir, 'val'), split='val', transform=test_transforms, target_transform=None)
        return trainset, valset

    def _get_dataset_loader(self, batch_size: int = 128, workers: Optional[int] = os.cpu_count()) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Defines the dataset loader for wrapped dataset
        Parameters:
            batch_size (int): Defines the batch size in data loader
            workers (int): Number of parallel threads to be used by data loader
            is_gpu (bool): True if CUDA is enabled so pin_memory is set to True
        Returns:
             torch.utils.data.TensorDataset: trainset, valset
        """
        workers = 4 if workers is None else workers
        train_loader = DataLoader(self._trainset, batch_size=batch_size, shuffle=True, num_workers=workers, sampler=None)
        val_loader = DataLoader(self._testset, batch_size=batch_size, shuffle=False, num_workers=workers)
        return train_loader, val_loader, val_loader

class MNIST(Dataset):
    """A MNIST dataset"""
    # properties
    input_channels: int = 1
    num_classes: int = 10

    def _get_dataset(self, dataset_dir: str = os.path.normpath("~/Documents/Data/")) -> Tuple[torchvision.datasets.MNIST, torchvision.datasets.MNIST]:
        dataset_dir = os.path.join(dataset_dir, 'MNIST')
        train_transforms = transforms.Compose([
            transforms.Resize(size=(28, 28)),
            transforms.ToTensor(),
            # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        test_transforms = transforms.Compose([
            transforms.Resize(size=(28, 28)),
            transforms.ToTensor(),
            # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        trainset = torchvision.datasets.MNIST(os.path.join(dataset_dir, "train"), train=True, transform=train_transforms, target_transform=None, download=True)
        testset = torchvision.datasets.MNIST(os.path.join(dataset_dir, "test"), train=False, transform=test_transforms, target_transform=None, download=True)
        return trainset, testset

    def _get_dataset_loader(self, batch_size: int = 128, workers: Optional[int] = os.cpu_count()) -> Tuple[DataLoader, DataLoader, DataLoader]:
        workers = 4 if workers is None else workers
        train_set = Subset(self._trainset, list(range(55000)))
        val_set = Subset(self._trainset, list(range(55000, 60000)))
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=workers)
        test_loader = DataLoader(self._testset, batch_size=batch_size, shuffle=False, num_workers=workers)
        return train_loader, val_loader, test_loader

class SVHN(Dataset):
    """A SVHN dataset"""
    input_channels: int = 3
    num_classes: int = 10

    def _get_dataset(self, dataset_dir: str = os.path.normpath("~/Documents/Data/")) -> Tuple[torchvision.datasets.SVHN, torchvision.datasets.SVHN]:
        dataset_dir = os.path.join(dataset_dir, 'SVHN')
        normalize = transforms.Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.1201, 0.1231, 0.1052])
        train_transforms = transforms.Compose([
            transforms.ToTensor(),
            normalize
            ])
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            normalize
            ])
        trainset = torchvision.datasets.SVHN(dataset_dir, split="train", transform=train_transforms, target_transform=None, download=True)
        testset = torchvision.datasets.SVHN(dataset_dir, split="test", transform=test_transforms, target_transform=None, download=True)
        return trainset, testset

    def _get_dataset_loader(self, batch_size: int = 128, workers: Optional[int] = os.cpu_count()) -> Tuple[DataLoader, DataLoader, DataLoader]:
        workers = 4 if workers is None else workers
        train_set = Subset(self._trainset, list(range(68257)))
        val_set = Subset(self._trainset, list(range(68257, 73257)))
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=workers)
        test_loader = DataLoader(self._testset, batch_size=batch_size, shuffle=False, num_workers=workers)
        return train_loader, val_loader, test_loader

class VOCSegmentation(Dataset):
    """A VOC Segmentation Dataset"""
    input_channels: int = 3
    num_classes: int = 21
    _valset: torchvision.datasets.VOCSegmentation

    def _get_dataset(self, dataset_dir: str = os.path.normpath("~/Documents/Data/")) -> Tuple[TorchDataset, TorchDataset]:
        dataset_dir = os.path.join(dataset_dir, "VOC_segmentation")
        normalize = transforms.SegmentationNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        train_transforms = transforms.SegmentationCompose([
                    transforms.SegmentationRandomResizedCrop((513, 513), scale=(0.5, 1.)),
                    transforms.SegmentationRandomHorizontalFlip(),
                    transforms.SegmentationToTensor(255),
                    transforms.SegmentationReshape(),
                    normalize
                    ])
        test_transforms = transforms.SegmentationCompose([
                    transforms.SegmentationResize(513),
                    transforms.SegmentationCenterCrop(513),
                    transforms.SegmentationToTensor(255),
                    transforms.SegmentationReshape(),
                    normalize
                    ])
        trainset = torchvision.datasets.VOCSegmentation(dataset_dir, transforms=train_transforms, download=True)
        self._valset = torchvision.datasets.VOCSegmentation(dataset_dir, image_set="val", transforms=test_transforms, download=True)
        testset = torchvision.datasets.VOCSegmentation(dataset_dir, year="2007", image_set="test", transforms=test_transforms, download=True)
        return trainset, testset

    def _get_dataset_loader(self, batch_size: int = 128, workers: Optional[int] = os.cpu_count()) -> Tuple[DataLoader, DataLoader, DataLoader]:
        workers = 4 if workers is None else workers
        train_loader = DataLoader(self._trainset, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)
        val_loader = DataLoader(self._valset, batch_size=batch_size, shuffle=False, num_workers=workers)
        test_loader = DataLoader(self._testset, batch_size=batch_size, shuffle=False, num_workers=workers)
        return train_loader, val_loader, test_loader
