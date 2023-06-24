# -*-coding:utf-8-*-
import random

from PIL import Image
from PIL import ImageFilter, ImageOps
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import transforms


class GaussianBlur:
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class Solarize:
    """Solarize augmentation from BYOL: https://arxiv.org/abs/2006.07733"""

    def __call__(self, x):
        return ImageOps.solarize(x)


class TwoCropsTransform:
    def __init__(self, args):
        self.base_transform1 = transforms.Compose([
            transforms.RandomResizedCrop(
                args.input_size, scale=(args.min_crop, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=1.0),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        ]
        )
        self.base_transform2 = transforms.Compose([
            transforms.RandomResizedCrop(
                args.input_size, scale=(args.min_crop, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.1),
            transforms.RandomApply([Solarize()], p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        ]
        )

    def __call__(self, x):
        im1 = self.base_transform1(x)
        im2 = self.base_transform2(x)
        return im1, im2


class MultiCropsTransform:
    def __init__(self, args):
        self.crop_num = args.multi_crop_num
        self.base_transform1 = transforms.Compose([
            transforms.RandomResizedCrop(
                args.input_size, scale=(args.global_crop, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=1.0),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        ]
        )
        self.base_transform2 = transforms.Compose([
            transforms.RandomResizedCrop(
                args.input_size, scale=(args.global_crop, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.1),
            transforms.RandomApply([Solarize()], p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        ]
        )
        self.local_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                args.multi_crop_size, scale=(args.min_crop, args.global_crop), interpolation=Image.BICUBIC),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        ]
        )

    def __call__(self, x):
        im1 = self.base_transform1(x)
        im2 = self.base_transform2(x)
        multi_im = []
        for _ in range(self.crop_num):
            multi_im.append(self.local_transform(x))
        return im1, im2, multi_im
