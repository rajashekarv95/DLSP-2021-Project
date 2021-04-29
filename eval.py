# Please do not change this file.
# We will use this eval script to benchmark your model.
# If you find a bug, post it on campuswire.

import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from dataloader import CustomDataset
from models.resnet_simclr import ResNetSimCLR


from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


team_id = 12
team_name = "Self Supervised Learners"
email_address = "rv2138@nyu.edu"

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint-path', type=str)
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
args = parser.parse_args()

# mean = (0.4836, 0.4527, 0.4011)
# std = (0.3065, 0.2728, 0.2355)

dataset_folder= '/dataset'
dataset = ContrastiveLearningDataset(dataset_folder)

evalset = dataset.get_val_dataset(1)

# eval_transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(mean=mean, std=std)
#     ])

#evalset = CustomDataset(root='/dataset', split="val", transform=eval_transform)
evalloader = torch.utils.data.DataLoader(evalset, batch_size=256, shuffle=False, num_workers=2)

net = ResNetSimCLR(base_model=args.arch, out_dim=800)
checkpoint = torch.load(args.checkpoint_path)
net.load_state_dict(checkpoint['state_dict'])
#net.load_state_dict(checkpoint)
net = net.cuda()

net.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in evalloader:
        images, labels = data
        images = torch.cat(images, dim=0)
        images = images.cuda()
        labels = labels.cuda()

        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()


print(f"Team {team_id}: {team_name} Accuracy: {(100 * correct / total):.2f}%")
print(f"Team {team_id}: {team_name} Email: {email_address}")