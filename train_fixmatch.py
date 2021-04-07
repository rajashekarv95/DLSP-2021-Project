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
import torchvision

from dataloader import CustomDataset
from transforms import TransformFixMatch, get_transforms

# random.seed(10)
# torch.manual_seed(10)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(10)

# torch.backends.cudnn.deterministic=True

def main():
    #TODO: Get args
    dataset_folder = "./dataset" #TODO
    batch_size_labeled = 64 #1024
    batch_size_unlabeled = 64 #5120
    batch_size_val = 64 #5120
    n_epochs = 10
    n_steps = 10
    num_classes = 800
    threshold = 0.6
    learning_rate = 0.01
    momentum = 0.9
    lamd = 1
    tau = 0.95

    # print("pwd: ", os.getcwd())
    train_transform, val_transform = get_transforms()

    labeled_train_dataset = CustomDataset(root= dataset_folder, split = "train", transform = train_transform)
    unlabeled_train_dataset = CustomDataset(root= dataset_folder, 
                                            split = "unlabeled", 
                                            transform = TransformFixMatch(mean = 0, std = 0))#TODO
    val_dataset = CustomDataset(root= dataset_folder, split = "val", transform = val_transform)

    labeled_train_loader = DataLoader(labeled_train_dataset, batch_size= batch_size_labeled, shuffle= True)
    unlabeled_train_loader = DataLoader(unlabeled_train_dataset, batch_size= batch_size_unlabeled, shuffle= True)
    val_loader = DataLoader(val_dataset, batch_size= batch_size_val, shuffle= True)


    #Uncomment to generate some weak and strong augs
    '''
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
    '''

    labeled_iter = iter(labeled_train_loader)
    unlabeled_iter = iter(unlabeled_train_loader)

    model = torchvision.models.resnet18(pretrained= False, num_classes = num_classes)
    optimizer = torch.optim.SGD(model.parameters(), 
                                lr = learning_rate,
                                momentum= momentum,
                                nesterov= True)

    model.train()
    for epoch in tqdm(range(n_epochs)):
        for batch_idx in tqdm(range(n_steps)):
            try:
                img_lab, targets_lab = labeled_iter.next()
            except:
                labeled_iter = iter(labeled_train_loader)
                img_lab, targets_lab = labeled_iter.next()

            try:
                unlab, _ = unlabeled_iter.next()
                img_weak = unlab[0]
                img_strong = unlab[1]
            except:
                unlabeled_iter = iter(unlabeled_train_loader)
                unlab, _ = unlabeled_iter.next()
                img_weak = unlab[0]
                img_strong = unlab[1]

            # print("Weak: ", img_weak.size())
            # print("Strong: ", img_strong.size())
            # print("Lab: ", img_lab.size())
            # print(model(img_lab).size())
            # print(model(img_weak).size())
            # print(model(img_strong).size())
            img_cat = torch.cat((img_lab, img_weak, img_strong), dim = 0)
            # print(img_cat.size())
            logits_cat = model(img_cat)
            logits_lab = logits_cat[:batch_size_labeled]
            # print(logits_lab.size())
            logits_unlab = logits_cat[batch_size_labeled:]
            # print(logits_unlab)

            logits_weak, logits_strong = torch.chunk(logits_unlab, chunks= 2, dim = 0)
            # print(logits_strong.size(), logits_weak.size())
            # print(logits_weak)
            # print(logits_strong)
            pseudo_label = torch.softmax(logits_weak.detach()/tau, dim= 1)
            max_probs, targets_unlab = torch.max(pseudo_label, dim= 1)
            mask = max_probs.ge(threshold).float()
            
            # mask = 1 * torch.ge(torch.softmax(logits_weak, dim= 1), threshold)
            # mask = torch.sum(mask, dim = 1)
            # print(mask.size())
            # print(torch.sum(mask, dim = 1))
            # targets_unlab = torch.argmax(logits_weak, dim = 1)
            # print()
            # print(targets_unlab)
            # print(torch.sum(targets_unlab, dim = 1))
            # print("logits strong: ", logits_strong.size())
            # print("Targets unlab: ", targets_unlab.size())
            # print("Mask: ", mask.size())
            loss_labeled = F.cross_entropy(logits_lab, targets_lab, reduction='mean')

            # print("CE: ", F.cross_entropy(logits_strong, targets_unlab, reduction= 'none').size())

            loss_unlabeled = (F.cross_entropy(logits_strong, targets_unlab, reduction= 'none') * mask).mean()

            # print("Loss labelled, loss unlabelled: ", loss_labeled, loss_unlabeled)

            loss_total = loss_labeled + lamd * loss_unlabeled

            print("Total loss: ", loss_total)

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()


            # break

        model.eval()
        with torch.no_grad():
            val_loss = 0
            for batch in val_loader:
                logits_val = model(batch[0])
                val_loss += F.cross_entropy(logits_val, batch[1])
                break
            print("Val loss: ", val_loss)

        # break

    

if __name__ == '__main__':
    main()