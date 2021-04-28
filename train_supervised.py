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
from torchvision import datasets, transforms, models

from dataloader import CustomDataset

from models.resnet_simclr import ResNetSimCLR


from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
from simclr import SimCLR


random.seed(10)
np.random.seed(10)
torch.manual_seed(10)
if torch.cuda.is_available():
    torch.cuda.manual_seed(10)

torch.backends.cudnn.deterministic=True

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

def main():
    #TODO: Get args
    # python3 train_fixmatch.py --checkpoint-path ./checkpoint_path/model.pth --batch-size 1 --num-epochs 1 --num-steps 1 --train-from-start 1 --dataset-folder ./dataset
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-path', type=str, default= "$SCRATCH/checkpoints/model.pth")
    parser.add_argument('--sup-checkpoint-path', type=str, default= "/scratch/sm9669/checkpoints/modelsup.pth")
    parser.add_argument('--batch-size', type=int, default= 64)
    parser.add_argument('--num-epochs', type=int, default= 30)
    parser.add_argument('--num-steps', type=int, default= 10)
    parser.add_argument('--train-from-start', type= int, default= 0)
    parser.add_argument('--dataset-folder', type= str, default= "/dataset")
    parser.add_argument('--learning-rate', type = float, default= 0.0003)
    parser.add_argument('--threshold', type = float, default= 0.5)
    parser.add_argument('--mu', type= int, default= 7)
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')

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
    sup_checkpoint_path = args.sup_checkpoint_path
    train_from_start = args.train_from_start

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")


    dataset = ContrastiveLearningDataset(dataset_folder)

    labeled_train_dataset = dataset.get_labeled_dataset(1)
    val_dataset = dataset.get_val_dataset(1)


    #train_transform, val_transform = get_transforms()

    #labeled_train_dataset = CustomDataset(root= dataset_folder, split = "train", transform = train_transform)
    #val_dataset = CustomDataset(root= dataset_folder, split = "val", transform = val_transform)
    labeled_train_loader = DataLoader(labeled_train_dataset, batch_size= 64, shuffle= True)
    val_loader = DataLoader(val_dataset, batch_size= batch_size_val, shuffle= False)


    #labeled_iter = iter(labeled_train_loader)

    # model = torchvision.models.wide_resnet50_2(pretrained= False, num_classes = num_classes)
    #model = resnet34(pretrained=False, num_classes = 800)
    model = ResNetSimCLR(base_model=args.arch, out_dim=800)

    model = model.to(device)
    if train_from_start == 0:
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['state_dict'])
            print("Restoring model from checkpoint")


    optimizer = torch.optim.Adam(model.parameters(), lr= learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(labeled_train_loader), eta_min=0,
                                                           last_epoch=-1)

    model.train()
    for epoch in tqdm(range(n_epochs)):
        loss_epoch = 0.0
        for batch_idx, batch in enumerate(tqdm(labeled_train_loader)):
            img_lab = torch.cat(batch[0], dim=0)
            
            img_lab = img_lab.to(device)
            targets_lab = batch[1].to(device)
            logits_lab = model(img_lab)
            loss_labeled = F.cross_entropy(logits_lab, targets_lab, reduction='mean')

            loss_epoch += loss_labeled

            optimizer.zero_grad()
            loss_labeled.backward()
            optimizer.step()
            scheduler.step()


        print(f"Epoch number: {epoch}, loss: {loss_epoch/(n_steps)}", flush= True)
        
        torch.save(model.state_dict(), sup_checkpoint_path)
        model.eval()
        with torch.no_grad():
            val_loss = 0
            val_size = 0
            for batch in val_loader:
                logits_val = model(batch[0].to(device))
                val_loss += F.cross_entropy(logits_val, batch[1].to(device))
                val_size += 1
            print("Val loss: ", val_loss/val_size, flush= True)

    

if __name__ == '__main__':
    main()