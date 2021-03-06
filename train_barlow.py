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
from models.resnet_barlow import resnet34, resnet18, wide_resnet50_2

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
import lightly

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

def adjust_learning_rate(args, optimizer, loader, step):
	max_steps = args.num_epochs * len(loader)
	warmup_steps = args.warmup_epochs * len(loader)
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


def save_checkpoint(state, checkpoint_path):
	torch.save(state, checkpoint_path)


def main():
	#TODO: Get args
	# python3 train_fixmatch.py --checkpoint-path ./checkpoint_path/model.pth --batch-size 1 --num-epochs 1 --num-steps 1 --train-from-start 1 --dataset-folder ./dataset
	parser = argparse.ArgumentParser()
	parser.add_argument('--checkpoint-path', type=str, default= "./checkpoints/model_barlow_20h.pth.tar")
	parser.add_argument('--batch-size', type=int, default= 512)
	parser.add_argument('--num-epochs', type=int, default= 10)
	parser.add_argument('--num-steps', type=int, default= 1)
	parser.add_argument('--train-from-start', type= int, default= 0)
	parser.add_argument('--dataset-folder', type= str, default= "./dataset")
	parser.add_argument('--learning-rate', type = float, default= 0.01)
	parser.add_argument('--threshold', type = float, default= 0.5)
	parser.add_argument('--mu', type= int, default= 7)
	parser.add_argument('--lambd', type= float, default= 0.005)
	parser.add_argument('--momentum', type= float, default= 0.9)
	parser.add_argument('--weight-decay', type= float, default= 1.5*1e-6)
	parser.add_argument('--warmup-epochs', type= int, default= 2)
	parser.add_argument('--scale-loss', type = float, default= 1.0/32.0)
	parser.add_argument('--wide', type= int, default= 0)
	args = parser.parse_args()

	dataset_folder = args.dataset_folder
	batch_size = args.batch_size
	n_epochs = args.num_epochs
	num_classes = 800
	lambd = args.lambd
	weight_decay = args.weight_decay
	checkpoint_path = args.checkpoint_path
	train_from_start = args.train_from_start
	
	if torch.cuda.is_available():
		device = torch.device("cuda")
	else:
		device = torch.device("cpu")

	unlabeled_train_dataset = CustomDataset(root= dataset_folder, split = "unlabeled", transform = TransformBarlowTwins())
	unlabeled_train_loader = DataLoader(unlabeled_train_dataset, batch_size= batch_size, shuffle= True, num_workers= 4)

	if args.wide == 1:
		model = lightly.models.BarlowTwins(wide_resnet50_2(pretrained= False), num_ftrs= 2048)
	else:
		model = lightly.models.BarlowTwins(resnet18(pretrained= False), num_ftrs= 512)

	optimizer = LARS(model.parameters(), lr=0, weight_decay=weight_decay,
					 weight_decay_filter=exclude_bias_and_norm,
					 lars_adaptation_filter=exclude_bias_and_norm)

	criterion = lightly.loss.BarlowTwinsLoss()

	start_epoch = 0

	model.train()
	losses = Average()

	model = model.to(device)
	criterion = criterion.to(device)

	if train_from_start == 0:
		assert os.path.isfile(checkpoint_path), "Error: no checkpoint directory found!"
		print("Restoring model from checkpoint")
		# args.out = os.path.dirname(args.resume)
		checkpoint = torch.load(checkpoint_path, map_location= device)
		if args.wide == 0:
			model = torch.nn.DataParallel(model)
		# best_acc = checkpoint['best_acc']
		start_epoch = checkpoint['epoch'] - 1
		model.load_state_dict(checkpoint['state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer'])

	if torch.cuda.device_count() > 1:
		print("Let's use", torch.cuda.device_count(), "GPUs!")
		model = torch.nn.DataParallel(model)
		criterion = torch.nn.DataParallel(criterion)

	model = model.to(device)
	criterion = criterion.to(device)

	#TODO
	# scaler = torch.cuda.amp.GradScaler()
	# model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

	for epoch in tqdm(range(start_epoch, n_epochs)):

		# for batch_idx in tqdm(range(n_steps)): ## CHECK

		for batch_idx, batch in enumerate(tqdm(unlabeled_train_loader)):
			y_a = batch[0][0].to(device)
			y_b = batch[0][1].to(device)
			
			z_a, z_b = model(y_a, y_b)
			loss = criterion(z_a, z_b).mean()

			lr = adjust_learning_rate(args, optimizer, unlabeled_train_loader, epoch * len(unlabeled_train_loader) + batch_idx)
			optimizer.zero_grad()

			# scaler.scale(loss).backward()
			# scaler.step(optimizer)
			# scaler.update()
			loss.backward()
			optimizer.step()

			losses.update(loss.item())

			if batch_idx % 25 == 0:
				print(f"Epoch number: {epoch}, loss_avg: {losses.avg}, loss: {loss.item()}, lr: {lr}", flush= True)
		if torch.cuda.device_count() > 1:
			save_checkpoint({
					'epoch': epoch + 1,
					'state_dict': model.module.state_dict(),
					'optimizer': optimizer.state_dict()
				}, checkpoint_path)
		else:
			save_checkpoint({
					'epoch': epoch + 1,
					'state_dict': model.state_dict(),
					'optimizer': optimizer.state_dict()
				}, checkpoint_path)


if __name__ == '__main__':
	main()