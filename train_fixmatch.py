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
from models.resnet import resnet34, resnet18

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
    parser.add_argument('--lambd', type= int, default= 1)
    parser.add_argument('--momentum', type= float, default= 0.9)
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
    momentum = args.momentum
    lamd = args.lambd
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
    unlabeled_train_dataset = CustomDataset(root= dataset_folder, 
                                            split = "unlabeled", 
                                            transform = TransformFixMatch(mean = 0, std = 0))#TODO
                                            
    val_dataset = CustomDataset(root= dataset_folder, split = "val", transform = val_transform)

    labeled_train_loader = DataLoader(labeled_train_dataset, batch_size= batch_size_labeled, shuffle= True)
    unlabeled_train_loader = DataLoader(unlabeled_train_dataset, batch_size= batch_size_unlabeled, shuffle= True)
    val_loader = DataLoader(val_dataset, batch_size= batch_size_val, shuffle= False)


    #Uncomment to generate some weak and strong augs
    #Set batch size = 1
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
    # unlabeled_train_loader = unlabeled_train_loader[:250]
    unlabeled_iter = iter(unlabeled_train_loader)

    # model = torchvision.models.wide_resnet50_2(pretrained= False, num_classes = num_classes)
    model = resnet18(pretrained=False, num_classes = 800)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

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
        loss_lab_epoch = 0.0
        loss_unlab_epoch = 0.0
        
        unlabeled_iter = iter(unlabeled_train_loader)

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
            
            img_lab = img_lab.to(device)
            targets_lab = targets_lab.to(device)
            img_weak = img_weak.to(device)
            img_strong = img_strong.to(device)

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

            # print("Total loss: ", loss_total)
            loss_epoch += loss_total
            loss_lab_epoch += loss_labeled
            loss_unlab_epoch += loss_unlabeled

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()


            # break
        print(f"Epoch number: {epoch}, loss: {loss_epoch/(n_steps)}, \
            loss lab: {loss_lab_epoch/(n_steps)},\
            loss unlab: {loss_unlab_epoch/(n_steps)}", flush= True)
        torch.save(model.state_dict(), checkpoint_path)
        model.eval()
        with torch.no_grad():
            val_loss = 0
            val_size = 0
            total = 0
            correct = 0
            for batch in val_loader:
                logits_val = model(batch[0].to(device))
                labels = batch[1].to(device)
                val_loss += F.cross_entropy(logits_val, labels)
                _, predicted = torch.max(logits_val.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                val_size += 1
                # break
            print(f"Val loss: {val_loss/val_size}, Accuracy: {(100 * correct / total):.2f}%", flush= True)

        # break

    

if __name__ == '__main__':
    main()