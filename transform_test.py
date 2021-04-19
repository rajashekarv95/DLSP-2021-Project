import torch
from transforms import TransformFixMatch, get_transforms, TransformBarlowTwins
from dataloader import CustomDataset
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

train_transform, _ = get_transforms()
unlabeled_train_dataset = CustomDataset(root= './dataset', 
                                            split = "unlabeled", 
                                            transform = TransformBarlowTwins())
unlabeled_train_loader = DataLoader(unlabeled_train_dataset, batch_size= 10, shuffle= True)

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