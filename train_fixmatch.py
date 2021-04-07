import argparse
import logging
import math
import os
import random
import time
from collections import OrderedDict
from matplotlib import pyplot as plt

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

from dataloader import CustomDataset
from transforms import TransformFixMatch, get_transforms

def main():
    #TODO: Get args
    dataset_folder = "./dataset" #TODO
    # print("pwd: ", os.getcwd())
    train_transform, val_transform = get_transforms()

    labeled_train_dataset = CustomDataset(root= dataset_folder, split = "train", transform = train_transform)
    unlabeled_train_dataset = CustomDataset(root= dataset_folder, 
                                            split = "unlabeled", 
                                            transform = TransformFixMatch(mean = 0, std = 0))#TODO
    val_dataset = CustomDataset(root= dataset_folder, split = "val", transform = val_transform)

    labeled_train_loader = DataLoader(labeled_train_dataset, batch_size= 1, shuffle= True)
    unlabeled_train_loader = DataLoader(unlabeled_train_dataset, batch_size= 1, shuffle= True)

    for batch in unlabeled_train_loader:
        w_img = batch[0][0]
        w_img = w_img[0]
        s_img = batch[0][1]
        s_img = s_img[0]
        print("Weak: ", batch[0][0].size())
        print("Strong: ", batch[0][1].size())
        print("Label: ", batch[1])
        plt.figure()
        plt.imshow(s_img.permute(1, 2, 0))
        plt.savefig('./test_augmentations/strong.png')
        
        plt.figure()
        plt.imshow(w_img.permute(1, 2, 0))
        plt.savefig('./test_augmentations/weak.png')
        break
    

if __name__ == '__main__':
    main()