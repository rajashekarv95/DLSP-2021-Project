import torch 
from dataloader import CustomDataset

train_transform = None 
dataset_folder = '/Users/surabhiranjan/Desktop/deeplearning/project/DLSP-2021-Project/dataset/'
labeled_train_dataset = CustomDataset(root= dataset_folder, split = "train", transform = train_transform)

for batch in labeled_train_dataset:
	print(batch[0])
# train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

def normalize(x):
	x_mean = torch.mean(x, dim = [2,3], keepdim=True)
	x_std = torch.std(x, dim = [2,3], keepdim=True)
	x = x - x_mean
	x = x / x_std
	return x

