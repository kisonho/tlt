from typing import Callable, List, Optional, Tuple

import torch
from PIL import Image
from torchvision.transforms import functional
from torchvision.transforms import * # type: ignore

from .cityscapes import MappingLabel

class SegmentationColorJitter(ColorJitter):
    """Color Jitter for segmentation dataset"""
    def forward(self, img: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        img = super().forward(img)
        return img, label

class SegmentationNormalize(Normalize):
    """Normalize for segmentation dataset"""
    __label_mean_dim: Optional[int]

    def __init__(self, mean: List[float], std: List[float], inplace: bool=False, label_mean_dim: Optional[int] = None):
        super().__init__(mean, std, inplace)
        self.__label_mean_dim = label_mean_dim

    def forward(self, img: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return super().forward(img), label.to(torch.long) if self.__label_mean_dim is None else label.mean(self.__label_mean_dim).to(torch.long)

class SegmentationToTensor(ToTensor):
    __scale: int

    def __init__(self, scale: int = 1) -> None:
        super().__init__()
        self.__scale = scale

    """To tensor for segmentation dataset"""
    def __call__(self, pic: Image.Image, label: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return functional.to_tensor(pic).to(torch.float), (functional.to_tensor(label) * self.__scale).to(torch.long)

class SegmentationCompose(Compose):
    """Compose for segmentation dataset"""
    transforms: List[Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]]

    def __call__(self, img: torch.Tensor, label: torch.Tensor):
        for t in self.transforms:
            img, label = t(img, label)
        return img, label

class SegmentationRandomHorizontalFlip(RandomHorizontalFlip):
    """Random horizontal flip for segmentation dataset"""
    p: float

    def forward(self, img: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if torch.rand(1) < self.p:
            return functional.hflip(img), functional.hflip(label)
        return img, label

class SegmentationRandomCrop(RandomCrop):
    def forward(self, img: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            PIL Image or Tensor: Cropped image.
        """
        if self.padding is not None:
            img = functional.pad(img, self.padding, self.fill, self.padding_mode)

        width, height = functional.get_image_size(img)
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            img = functional.pad(img, padding, self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            img = functional.pad(img, padding, self.fill, self.padding_mode)

        i, j, h, w = self.get_params(img, self.size)

        return functional.crop(img, i, j, h, w), functional.crop(label, i, j, h, w)

class SegmentationRandomResizedCrop(RandomResizedCrop):
    """Random resized crop for segmentation dataset"""
    ratio: List[float]
    scale: List[float]
    size: List[int]

    def forward(self, img: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        img = functional.resized_crop(img, i, j, h, w, self.size, self.interpolation)
        label = functional.resized_crop(label, i, j, h, w, self.size, self.interpolation)
        return img, label

class SegmentationReshape(torch.nn.Module):
    def forward(self, img: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        size = img.shape[-2:]
        label = label.view((size[0], size[1]))
        return img, label

class SegmentationCenterCrop(CenterCrop):
    def forward(self, img: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return super().forward(img), super().forward(label)

class SegmentationResize(Resize):
    def forward(self, img: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return super().forward(img), super().forward(label)