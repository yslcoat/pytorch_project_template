import os

import torch
from torch.utils.data import Dataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class SyntheticDataset(Dataset):
    """Random images and labels, deterministic per index. Useful for smoke-testing the pipeline."""

    def __init__(self, num_samples, num_classes, in_chans, img_size, seed=0):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.shape = (in_chans, img_size, img_size)
        self.seed = seed

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        g = torch.Generator().manual_seed(self.seed + idx)
        target = int(torch.randint(self.num_classes, (1,), generator=g))
        # shift the image mean by the class so the task is learnable
        img = (
            torch.randn(self.shape, generator=g) + (target / self.num_classes - 0.5) * 2
        )
        return img, target


def _image_folder_transforms(configs, is_training):
    from torchvision import transforms

    if is_training:
        tfms = [
            transforms.RandomResizedCrop(configs.img_size),
            transforms.RandomHorizontalFlip(),
        ]
    else:
        tfms = [
            transforms.Resize(int(configs.img_size / 0.875)),
            transforms.CenterCrop(configs.img_size),
        ]
    tfms += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(tfms)


def create_dataset(configs, is_training=False):
    if configs.dataset == "synthetic":
        return SyntheticDataset(
            num_samples=(
                configs.train_num_samples if is_training else configs.val_num_samples
            ),
            num_classes=configs.num_classes,
            in_chans=configs.in_chans,
            img_size=configs.img_size,
            seed=0 if is_training else 10_000_000,
        )
    if configs.dataset == "image_folder":
        from torchvision.datasets import ImageFolder

        split = configs.train_split if is_training else configs.val_split
        return ImageFolder(
            os.path.join(configs.data_dir, split),
            transform=_image_folder_transforms(configs, is_training),
        )
    raise ValueError(f'Unknown dataset "{configs.dataset}".')
