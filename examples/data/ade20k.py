"""
Code modified from:
https://github.com/CSAILVision/semantic-segmentation-pytorch
"""
import abc, os, json, numpy as np, torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Any, NamedTuple, Sequence, Union


def imresize(im, size, interp='bilinear'):
    if interp == 'nearest':
        resample = Image.NEAREST
    elif interp == 'bilinear':
        resample = Image.BILINEAR
    elif interp == 'bicubic':
        resample = Image.BICUBIC
    else:
        raise Exception('resample method undefined!')

    return im.resize(size, resample)


class DatasetOptions(NamedTuple):
    img_sizes: Sequence[int]
    img_max_size: int
    padding_constant: int
    segm_downsampling_rate: int


class BaseDataset(Dataset):
    img_max_size: int
    img_sizes: Sequence[int]
    list_sample: list[dict[str, Any]]
    padding_constant: int

    def __init__(self, odgt: Union[str, list[Any]], opt: DatasetOptions, **kwargs):
        # parse options
        self.img_sizes = opt.img_sizes
        self.img_max_size = opt.img_max_size
        # max down sampling rate of network to avoid rounding during conv or pooling
        self.padding_constant = opt.padding_constant

        # parse the input list
        self.parse_input_list(odgt, **kwargs)

        # mean and std
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    
    def __len__(self) -> int:
        return len(self.list_sample)

    def parse_input_list(self, odgt: Union[str, list[Any]], max_sample: int = -1, start_idx: int = -1, end_idx = -1) -> None:
        if isinstance(odgt, list):
            self.list_sample = odgt
        elif isinstance(odgt, str):
            self.list_sample = [json.loads(x.rstrip()) for x in open(odgt, 'r')]

        if max_sample > 0:
            self.list_sample = self.list_sample[0:max_sample]
        if start_idx >= 0 and end_idx >= 0:     # divide file list
            self.list_sample = self.list_sample[start_idx:end_idx]
        assert len(self.list_sample) > 0

    def img_transform(self, img) -> torch.Tensor:
        # 0-255 to 0-1
        img = np.float32(np.array(img)) / 255.
        img = img.transpose((2, 0, 1)) # type: ignore
        img = torch.from_numpy(img).long()
        img = self.normalize(torch.from_numpy(img))
        return img

    def segm_transform(self, segm):
        # to tensor, -1 to 149
        segm = torch.from_numpy(np.array(segm)).long() - 1
        return segm

    # Round x to the nearest multiple of p and x' >= x
    def round2nearest_multiple(self, x, p):
        return ((x - 1) // p + 1) * p


class TrainDataset(BaseDataset):
    root_dataset: str
    segm_downsampling_rate: int
    if_shuffled: bool

    def __init__(self, root_dataset: str, odgt: Union[str, list[Any]], opt: DatasetOptions, **kwargs):
        super(TrainDataset, self).__init__(odgt, opt, **kwargs)
        self.root_dataset = root_dataset
        # down sampling rate of segm labe
        self.segm_downsampling_rate = opt.segm_downsampling_rate
        np.random.shuffle(self.list_sample) # type: ignore

    def __getitem__(self, index: int):
        # get sub-batch candidates
        this_sample = self.list_sample[index]

        # resize all images' short edges to the chosen size
        if isinstance(self.img_sizes, Sequence):
            this_short_size = np.random.choice(self.img_sizes)
        else:
            this_short_size = self.img_sizes

        # calculate the BATCH's height and width
        # since we concat more than one samples, the batch's h and w shall be larger than EACH sample
        img_height, img_width = this_sample['height'], this_sample['width']
        this_scale = min(
            this_short_size / min(img_height, img_width), \
            self.img_max_size / max(img_height, img_width))
        img_width = img_width * this_scale
        img_height = img_height * this_scale

        # Here we must pad both input image and segmentation map to size h' and w' so that p | h' and p | w'
        img_width = int(self.round2nearest_multiple(img_width, self.padding_constant))
        img_height = int(self.round2nearest_multiple(img_height, self.padding_constant))
        img_width //= self.segm_downsampling_rate
        img_height //= self.segm_downsampling_rate

        # load image and label
        image_path = os.path.join(self.root_dataset, this_sample['fpath_img'])
        segm_path = os.path.join(self.root_dataset, this_sample['fpath_segm'])

        img = Image.open(image_path).convert('RGB')
        segm = Image.open(segm_path)
        assert(segm.mode == "L")
        assert(img.size[0] == segm.size[0])
        assert(img.size[1] == segm.size[1])

        # random_flip
        if np.random.choice([0, 1]):
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            segm = segm.transpose(Image.FLIP_LEFT_RIGHT)

        # note that each sample within a mini batch has different scale param
        img = imresize(img, (img_width, img_height), interp='bilinear')
        segm = imresize(segm, (img_width, img_height), interp='nearest')

        # further downsample seg label, need to avoid seg label misalignment
        segm_rounded_width = self.round2nearest_multiple(segm.size[0], self.segm_downsampling_rate)
        segm_rounded_height = self.round2nearest_multiple(segm.size[1], self.segm_downsampling_rate)
        segm_rounded = Image.new('L', (segm_rounded_width, segm_rounded_height), 0)
        segm_rounded.paste(segm, (0, 0))
        segm = imresize(
            segm_rounded,
            (segm_rounded.size[0] // self.segm_downsampling_rate, \
                segm_rounded.size[1] // self.segm_downsampling_rate), \
            interp='nearest')

        # image transform, to torch float tensor 3xHxW
        img = self.img_transform(img)

        # segm transform, to torch long tensor HxW
        segm = self.segm_transform(segm)
        return img, segm


class ValDataset(BaseDataset):
    root_dataset: str

    def __init__(self, root_dataset: str, odgt: Union[str, list[Any]], opt: DatasetOptions, **kwargs):
        super(ValDataset, self).__init__(odgt, opt, **kwargs)
        self.root_dataset = root_dataset

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        this_record = self.list_sample[index]
        # load image and label
        image_path = os.path.join(self.root_dataset, this_record['fpath_img'])
        segm_path = os.path.join(self.root_dataset, this_record['fpath_segm'])
        img = Image.open(image_path).convert('RGB')
        segm = Image.open(segm_path)
        assert(segm.mode == "L")
        assert(img.size[0] == segm.size[0])
        assert(img.size[1] == segm.size[1])
        ori_width, ori_height = img.size

        # resize all images' short edges to the chosen size
        if isinstance(self.img_sizes, Sequence):
            this_short_size = self.img_sizes[0]
        else:
            this_short_size = self.img_sizes

        # calculate target height and width
        scale = min(this_short_size / float(min(ori_height, ori_width)),
                    self.img_max_size / float(max(ori_height, ori_width)))
        target_height, target_width = int(ori_height * scale), int(ori_width * scale)

        # to avoid rounding in network
        target_width = self.round2nearest_multiple(target_width, self.padding_constant)
        target_height = self.round2nearest_multiple(target_height, self.padding_constant)

        # resize images
        img_resized = imresize(img, (target_width, target_height), interp='bilinear')

        # image transform, to torch float tensor 3xHxW
        img_resized = self.img_transform(img_resized)
        img_resized = torch.unsqueeze(img_resized, 0)

        # segm transform, to torch long tensor HxW
        segm = self.segm_transform(segm)
        return img_resized, segm


class TestDataset(BaseDataset):
    def __init__(self, odgt: Union[str, list[Any]], opt: DatasetOptions, **kwargs):
        super(TestDataset, self).__init__(odgt, opt, **kwargs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, None]:
        this_record = self.list_sample[index]
        # load image
        image_path = this_record['fpath_img']
        img = Image.open(image_path).convert('RGB')

        ori_width, ori_height = img.size

        # resize all images' short edges to the chosen size
        if isinstance(self.img_sizes, Sequence):
            this_short_size = self.img_sizes[0]
        else:
            this_short_size = self.img_sizes

        # calculate target height and width
        scale = min(this_short_size / float(min(ori_height, ori_width)),
                    self.img_max_size / float(max(ori_height, ori_width)))
        target_height, target_width = int(ori_height * scale), int(ori_width * scale)

        # to avoid rounding in network
        target_width = self.round2nearest_multiple(target_width, self.padding_constant)
        target_height = self.round2nearest_multiple(target_height, self.padding_constant)

        # resize images
        img_resized = imresize(img, (target_width, target_height), interp='bilinear')

        # image transform, to torch float tensor 3xHxW
        img_resized = self.img_transform(img_resized)
        img_resized = torch.unsqueeze(img_resized, 0)
        return img_resized, None