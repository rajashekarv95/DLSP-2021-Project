import argparse
import logging
import math
import os
import random
import time
from collections import OrderedDict
from matplotlib import pyplot as plt

import logging
import random

import numpy as np
import PIL
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw
from PIL import Image
from models.resnet import resnet34, resnet18


import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

from dataloader import CustomDataset
from transforms import TransformFixMatch, get_transforms, TransformBarlowTwins

from utils.misc import Average


import torch.nn as nn
import torchvision
import lightly

random.seed(10)
np.random.seed(10)
torch.manual_seed(10)
if torch.cuda.is_available():
	torch.cuda.manual_seed(10)

torch.backends.cudnn.deterministic=True

def save_checkpoint(state, checkpoint_path):
	torch.save(state, checkpoint_path)


def main():
	#TODO: Get args
	# python3 train_fixmatch.py --checkpoint-path ./checkpoint_path/model.pth --batch-size 1 --num-epochs 1 --num-steps 1 --train-from-start 1 --dataset-folder ./dataset
	parser = argparse.ArgumentParser()
	parser.add_argument('--checkpoint-path', type=str, default= "./model.pth")
	parser.add_argument('--batch-size', type=int, default= 512)
	parser.add_argument('--num-epochs', type=int, default= 10)
	parser.add_argument('--num-steps', type=int, default= 1)
	parser.add_argument('--train-from-start', type= int, default= 1)
	parser.add_argument('--dataset-folder', type= str, default= "./dataset")
	parser.add_argument('--learning-rate', type = float, default= 0.01)
	parser.add_argument('--threshold', type = float, default= 0.5)
	parser.add_argument('--mu', type= int, default= 7)
	parser.add_argument('--lambd', type= float, default= 0.005)
	parser.add_argument('--momentum', type= float, default= 0.9)
	parser.add_argument('--weight-decay', type= float, default= 1.5*1e-6)
	parser.add_argument('--num-ftrs', type= int, default= 512)
	parser.add_argument('--out-dim', type= int, default= 512)
	parser.add_argument('--proj-hidden-dim', type= int, default= 128)
	parser.add_argument('--pred-hidden-dim', type= int, default= 128)
	parser.add_argument('--num-mlp-layers', type= int, default= 2)


	args = parser.parse_args()

	dataset_folder = args.dataset_folder
	batch_size_labeled = args.batch_size
	mu = args.mu
	batch_size_unlabeled = mu * args.batch_size
	n_epochs = args.num_epochs
	n_steps = args.num_steps		
	num_classes = 800
	threshold = args.threshold
	learning_rate = args.learning_rate
	momentum = args.momentum
	lambd = args.lambd
	weight_decay = args.weight_decay
	checkpoint_path = args.checkpoint_path
	train_from_start = args.train_from_start
	
	if torch.cuda.is_available():
		device = torch.device("cuda")
	else:
		device = torch.device("cpu")

	train_transform, val_transform = get_transforms()
	unlabeled_train_dataset = CustomDataset(root= dataset_folder, split = "unlabeled", transform = TransformBarlowTwins())
# create a dataloader for training
	dataloader_train_simsiam = torch.utils.data.DataLoader(
		unlabeled_train_dataset,
		batch_size=512,
		shuffle=True)
		# collate_fn=collate_fn)
		# drop_last=True,
		# num_workers=1
	# )

	resnet = torchvision.models.resnet18()
	backbone = nn.Sequential(*list(resnet.children())[:-1])

	# create the SimSiam model using the backbone from above
	model = lightly.models.SimSiam(
		backbone,
		num_ftrs=args.num_ftrs,
		proj_hidden_dim=args.pred_hidden_dim,
		pred_hidden_dim=args.pred_hidden_dim,
		out_dim=args.out_dim,
		num_mlp_layers=args.num_mlp_layers
	)

	model = model.to(device)

	criterion = lightly.loss.SymNegCosineSimilarityLoss()

	# scale the learning rate
	lr = 0.05 * args.batch_size / 256
	# use SGD with momentum and weight decay
	optimizer = torch.optim.SGD(
		model.parameters(),
		lr=lr,
		momentum=0.9,
		weight_decay=5e-4
	)

	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	model.to(device)

	avg_loss = 0.
	avg_output_std = 0.

	start_epoch = 0 

	model.train()
	for epoch in tqdm(range(start_epoch, n_epochs)):

		for batch_idx, batch in tqdm(enumerate(dataloader_train_simsiam)):

			# move images to the gpu
			x0 = batch[0][0].to(device)
			x1 = batch[0][1].to(device)

			# run the model on both transforms of the images
			# the output of the simsiam model is a y containing the predictions
			# and projections for each input x
			y0, y1 = model(x0, x1)

			# backpropagation
			loss = criterion(y0, y1)
			loss.backward()

			optimizer.step()
			optimizer.zero_grad()

			# calculate the per-dimension standard deviation of the outputs
			# we can use this later to check whether the embeddings are collapsing
			output, _ = y0
			output = output.detach()
			output = torch.nn.functional.normalize(output, dim=1)

			output_std = torch.std(output, 0)
			output_std = output_std.mean()

			# use moving averages to track the loss and standard deviation
			w = 0.9
			avg_loss = w * avg_loss + (1 - w) * loss.item()
			avg_output_std = w * avg_output_std + (1 - w) * output_std.item()

			print(f'loss.item():', loss.item())

		# the level of collapse is large if the standard deviation of the l2
		# normalized output is much smaller than 1 / sqrt(dim)
		collapse_level = max(0., 1 - math.sqrt(args.out_dim) * avg_output_std)
		# print intermediate results
		print(f'[Epoch {epoch:3d}] '
			f'Loss = {avg_loss:.2f} | '
			f'Collapse Level: {collapse_level:.2f} / 1.00')

		save_checkpoint({
				'epoch': epoch + 1,
				'state_dict': model.state_dict(),
				'optimizer': optimizer.state_dict()
			}, checkpoint_path)

if __name__ == '__main__':
	main()