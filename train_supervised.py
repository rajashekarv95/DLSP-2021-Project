import argparse
import logging
import math
import os
import random
import time
from collections import OrderedDict
from matplotlib import pyplot as plt
import argparse

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

from models.resnet import wide_resnet50_2
from models.resnet import resnet34

random.seed(10)
np.random.seed(10)
torch.manual_seed(10)
if torch.cuda.is_available():
    torch.cuda.manual_seed(10)

torch.backends.cudnn.deterministic=True

def main():
    #TODO: Get args
    # python3 train_fixmatch.py --checkpoint-path ./checkpoint_path/model.pth --batch-size 1 --num-epochs 1 --num-steps 1 --train-from-start 1 --dataset-folder ./dataset
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-path', type=str, default= "$SCRATCH/model.pth")
    parser.add_argument('--batch-size', type=int, default= 64)
    parser.add_argument('--num-epochs', type=int, default= 10)
    parser.add_argument('--num-steps', type=int, default= 10)
    parser.add_argument('--train-from-start', type= int, default= 1)
    parser.add_argument('--dataset-folder', type= str, default= "/dataset")
    parser.add_argument('--learning-rate', type = float, default= 0.01)
    parser.add_argument('--threshold', type = float, default= 0.5)
    parser.add_argument('--mu', type= int, default= 7)
    args = parser.parse_args()

    print(torchvision.__version__, flush= True)
    dataset_folder = args.dataset_folder
    batch_size_labeled = args.batch_size
    mu = args.mu
    batch_size_unlabeled = mu * args.batch_size
    batch_size_val = 256 #5120
    n_epochs = args.num_epochs
    n_steps = args.num_steps
    num_classes = 800
    threshold = args.threshold
    learning_rate = args.learning_rate
    momentum = 0.9
    lamd = 1
    tau = 0.95
    checkpoint_path = args.checkpoint_path
    train_from_start = args.train_from_start

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # print("pwd: ", os.getcwd())
    train_transform, val_transform = get_transforms()

    labeled_train_dataset = CustomDataset(root= dataset_folder, split = "train", transform = train_transform)
    val_dataset = CustomDataset(root= dataset_folder, split = "val", transform = val_transform)
    labeled_train_loader = DataLoader(labeled_train_dataset, batch_size= batch_size_labeled, shuffle= True)
    val_loader = DataLoader(val_dataset, batch_size= batch_size_val, shuffle= False)


    labeled_iter = iter(labeled_train_loader)

    # model = torchvision.models.wide_resnet50_2(pretrained= False, num_classes = num_classes)
    model = resnet34(pretrained=False, num_classes = 800)
    model = model.to(device)
    if train_from_start == 0:
        if os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path))
            print("Restoring model from checkpoint")

    optimizer = torch.optim.SGD(model.parameters(), 
                                lr = learning_rate,
                                momentum= momentum,
                                nesterov= True)

    model.train()
    for epoch in tqdm(range(n_epochs)):
        loss_epoch = 0.0
        for batch_idx in tqdm(range(n_steps)):
            try:
                img_lab, targets_lab = labeled_iter.next()
            except:
                labeled_iter = iter(labeled_train_loader)
                img_lab, targets_lab = labeled_iter.next()
            img_lab = img_lab.to(device)
            targets_lab = targets_lab.to(device)

            logits_lab = model(img_lab)
            loss_labeled = F.cross_entropy(logits_lab, targets_lab, reduction='mean')

            loss_epoch += loss_labeled

            optimizer.zero_grad()
            loss_labeled.backward()
            optimizer.step()


            # break
        print(f"Epoch number: {epoch}, loss: {loss_epoch/(n_steps)}", flush= True)
        
        torch.save(model.state_dict(), checkpoint_path)
        model.eval()
        with torch.no_grad():
            val_loss = 0
            val_size = 0
            for batch in val_loader:
                logits_val = model(batch[0].to(device))
                val_loss += F.cross_entropy(logits_val, batch[1].to(device))
                val_size += 1
                # break
            print("Val loss: ", val_loss/val_size, flush= True)

        # break

    

if __name__ == '__main__':
    main()