def normalize(x):
    x_sum = torch.sum(torch.mean(x, dim = [2,3]), dim = 0)/64
    x_sum_sq = torch.sum(torch.mean(torch.pow(x, 2), dim = [2,3]), dim = 0)/64
    # x = x - x_mean
    # x = x / x_std
    return x_sum, x_sum_sq


import os
import random
from PIL import Image
from tqdm import tqdm

import torch
from torchvision import transforms

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root, split):
        r"""
        Args:
            root: Location of the dataset folder, usually it is /dataset
            split: The split you want to used, it should be one of train, val or unlabeled.
            transform: the transform you want to applied to the images.
        """

        self.split = split
        # self.transform = transform
        self.transform = transforms.Compose([
        transforms.ToTensor()
        # transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])

        self.image_dir = os.path.join(root, split)
        label_path = os.path.join(root, f"{split}_label_tensor.pt")

        self.num_images = len(os.listdir(self.image_dir))

        if os.path.exists(label_path):
            self.labels = torch.load(label_path)
        else:
            self.labels = -1 * torch.ones(self.num_images, dtype=torch.long)
    
    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        with open(os.path.join(self.image_dir, f"{idx}.png"), 'rb') as f:
            img = Image.open(f).convert('RGB')

        return self.transform(img), self.labels[idx]


import torch
# from dataloader import CustomDataset
from transforms import TransformFixMatch, get_transforms
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

# img = torch.randn(2,3,96,96)
# s,ss = normalize(img)
# print(s.size())
# print(ss.size())


labeled_train_dataset = CustomDataset(root= "./dataset", split = "train")
labeled_train_loader = DataLoader(labeled_train_dataset, batch_size= 64, shuffle= False)

unlabeled_train_dataset = CustomDataset(root= "./dataset", 
                                            split = "unlabeled")

unlabeled_train_loader = DataLoader(unlabeled_train_dataset, batch_size= 64, shuffle= False)

mean = torch.zeros(3)
mean_sq = torch.zeros(3)
ctr = 0
for batch in tqdm(labeled_train_loader):
    # print(batch[0])
    ctr += 1
    img = batch[0]
    s,ss = normalize(img)
    # print(s.size())
    # print(s)
    # print(s.size())
    # print(ss.size())
    mean += s
    mean_sq += ss
    # break
    
for batch in tqdm(unlabeled_train_loader):
    # print(batch[0])
    ctr += 1
    img = batch[0]
    s,ss = normalize(img)
    # print(s.size())
    # print(s)
    # print(s.size())
    # print(ss.size())
    mean += s
    mean_sq += ss

print(mean/ctr, mean_sq/ctr)



