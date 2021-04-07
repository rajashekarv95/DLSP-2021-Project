import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms

from augment.randaugment import RandAugmentMC


class TransformFixMatch(object):
    def __init__(self, mean, std):
        #TODO
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2471, 0.2435, 0.2616)
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            # transforms.RandomCrop(size=32,
            #                       padding=int(32*0.125),
            #                       padding_mode='reflect')
                                  ])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            # transforms.RandomCrop(size=32,
            #                       padding=int(32*0.125),
            #                       padding_mode='reflect'),
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
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2471, 0.2435, 0.2616)
    transform_labeled = transforms.Compose([
            transforms.RandomHorizontalFlip(),
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