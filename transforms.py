import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms
import random 
from augment.randaugment import RandAugmentMC
from PIL import Image, ImageOps, ImageFilter


class TransformFixMatch(object):
    def __init__(self, mean, std):
        #TODO
        mean = (0.4836, 0.4527, 0.4011)
        std = (0.3065, 0.2728, 0.2355)
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(size = 96, scale = (0.75, 0.8), ratio= (1.0, 1.0))
                                  ])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)

def get_transforms():
    #TODO Change this
    cifar10_mean = (0.4836, 0.4527, 0.4011)
    cifar10_std = (0.3065, 0.2728, 0.2355)
    transform_labeled = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(size = 96, scale = (0.75, 0.8), ratio= (1.0, 1.0)),
            # transforms.RandomCrop(size=32,
            #                     padding=int(32*0.125),
            #                     padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
        ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    return transform_labeled, transform_val

class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img

class TransformBarlowTwins:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(96, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=1.0),
            Solarization(p=0.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4836, 0.4527, 0.4011],
                                 std=[0.3065, 0.2728, 0.2355])
        ])
        self.transform_prime = transforms.Compose([
            transforms.RandomResizedCrop(96, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.1),
            Solarization(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4836, 0.4527, 0.4011],
                                 std=[0.3065, 0.2728, 0.2355])
        ])

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return y1, y2