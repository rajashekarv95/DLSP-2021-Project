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

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

from dataloader import CustomDataset
from transforms import TransformFixMatch, get_transforms, TransformBarlowTwins

from utils.misc import Average

random.seed(10)
np.random.seed(10)
torch.manual_seed(10)
if torch.cuda.is_available():
	torch.cuda.manual_seed(10)

torch.backends.cudnn.deterministic=True

def exclude_bias_and_norm(p):
	return p.ndim == 1


class LARS(optim.Optimizer):
	def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001,
				 weight_decay_filter=None, lars_adaptation_filter=None):
		defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
						eta=eta, weight_decay_filter=weight_decay_filter,
						lars_adaptation_filter=lars_adaptation_filter)
		super().__init__(params, defaults)

	@torch.no_grad()
	def step(self):
		for g in self.param_groups:
			for p in g['params']:
				dp = p.grad

				if dp is None:
					continue

				if g['weight_decay_filter'] is None or not g['weight_decay_filter'](p):
					dp = dp.add(p, alpha=g['weight_decay'])

				if g['lars_adaptation_filter'] is None or not g['lars_adaptation_filter'](p):
					param_norm = torch.norm(p)
					update_norm = torch.norm(dp)
					one = torch.ones_like(param_norm)
					q = torch.where(param_norm > 0.,
									torch.where(update_norm > 0,
												(g['eta'] * param_norm / update_norm), one), one)
					dp = dp.mul(q)

				param_state = self.state[p]
				if 'mu' not in param_state:
					param_state['mu'] = torch.zeros_like(p)
				mu = param_state['mu']
				mu.mul_(g['momentum']).add_(dp)

				p.add_(mu, alpha=-g['lr'])


def save_checkpoint(state, checkpoint_path):
	torch.save(state, checkpoint_path)


def adjust_learning_rate(args, optimizer, loader, step):
	max_steps = args.num_epochs * len(loader)
	warmup_steps = 10 * len(loader)
	base_lr = args.learning_rate * args.batch_size / 256
	if step < warmup_steps:
		lr = base_lr * step / warmup_steps
	else:
		step -= warmup_steps
		max_steps -= warmup_steps
		q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
		end_lr = base_lr * 0.001
		lr = base_lr * q + end_lr * (1 - q)
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr
	return lr

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

	def off_diagonal(x):
		# return a flattened view of the off-diagonal elements of a square matrix
		n, m = x.shape
		assert n == m
		return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
		
	# dataset_folder = dataset_folder = "./dataset" 
	train_transform, val_transform = get_transforms()
	unlabeled_train_dataset = CustomDataset(root= dataset_folder, split = "unlabeled", transform = TransformBarlowTwins())
	unlabeled_train_loader = DataLoader(unlabeled_train_dataset, batch_size= 512, shuffle= True)

	model = resnet18(pretrained=False, num_classes = 800)
	optimizer = LARS(model.parameters(), lr=0, weight_decay=args.weight_decay,
					 weight_decay_filter=exclude_bias_and_norm,
					 lars_adaptation_filter=exclude_bias_and_norm)

	if torch.cuda.device_count() > 1:
		print("Let's use", torch.cuda.device_count(), "GPUs!")
		model = torch.nn.DataParallel(model)

	model = model.to(device)

	start_epoch = 0

	if train_from_start == 0:
		assert os.path.isfile(checkpoint_path), "Error: no checkpoint directory found!"
		print("Restoring model from checkpoint")
		# args.out = os.path.dirname(args.resume)
		checkpoint = torch.load(checkpoint_path)
		# best_acc = checkpoint['best_acc']
		start_epoch = checkpoint['epoch'] - 1
		model.load_state_dict(checkpoint['state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer'])

	model.train()
	losses = Average()
	losses_l = Average()
	losses_u = Average()
	mask_probs = Average()


	for epoch in tqdm(range(start_epoch, n_epochs)):

		# for batch_idx in tqdm(range(n_steps)): ## CHECK
		loss_epoch = 0.0
		loss_lab_epoch = 0.0
		loss_unlab_epoch = 0.0

		for batch_idx, batch in enumerate(tqdm(unlabeled_train_loader)):
			y_a = batch[0][0].to(device)
			y_b = batch[0][1].to(device)
			y_cat = torch.cat((y_a, y_b), dim = 0)

			z_cat = model(y_cat)

			z_a, z_b = torch.chunk(z_cat, chunks = 2, dim = 0)

			z_a_norm = (z_a - (torch.mean(z_a, dim = 1, keepdim=True))) / torch.std(z_a, dim = 1, keepdim=True)
			z_b_norm = (z_b - (torch.mean(z_b, dim = 1, keepdim=True))) / torch.std(z_b, dim = 1, keepdim=True)

			c = torch.matmul(z_a_norm.T, z_b_norm)
			c_diff = torch.pow(c - torch.eye(c.size()[0]).to(device), 2)

			loss = torch.sum(torch.mul(off_diagonal(c_diff), lambd))

			losses.update(loss.item())
			# losses_l.update(loss_labeled.item())
			# losses_u.update(loss_unlabeled.item())
			# mask_probs.update(mask.mean().item())

			lr = adjust_learning_rate(args, optimizer, unlabeled_train_loader, batch_idx)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			# scheduler.step()

			if batch_idx % 25 == 0:
				print(f"Epoch number: {epoch}, loss_avg: {losses.avg}, loss: {loss.item()}", flush= True)
		
		save_checkpoint({
				'epoch': epoch + 1,
				'state_dict': model.state_dict(),
				'optimizer': optimizer.state_dict()
			}, checkpoint_path)


if __name__ == '__main__':
	main()