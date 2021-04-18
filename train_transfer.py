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
import lightly

from dataloader import CustomDataset
from transforms import get_transforms

from models.resnet_barlow import resnet34, resnet18, wide_resnet50_2

from utils.misc import Average

random.seed(10)
np.random.seed(10)
torch.manual_seed(10)
if torch.cuda.is_available():
	torch.cuda.manual_seed(10)

torch.backends.cudnn.deterministic=True

def exclude_bias_and_norm(p):
	return p.ndim == 1

class Classifier(torch.nn.Module):

	def __init__(self, ip):
		super().__init__()
		self.fc1 = torch.nn.Linear(ip, 1024)
		self.fc2 = torch.nn.Linear(1024, 2048)
		self.fc3 = torch.nn.Linear(2048, 4096)
		self.fc4 = torch.nn.Linear(4096, 800)
	

	def forward(self, x):
		# TODO
		x = self.fc1(x)
		x = F.relu(x)
		x = self.fc2(x)
		x = F.relu(x)
		x = self.fc3(x)
		x = F.relu(x)
		x = self.fc4(x)

		return x

def save_checkpoint(state, checkpoint_path):
	torch.save(state, checkpoint_path)

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--checkpoint-path', type=str, default= "./checkpoints/model_transfer.pth.tar")
	parser.add_argument('--transfer-path', type=str, default= "./checkpoints/model_barlow.pth.tar")
	parser.add_argument('--batch-size', type=int, default= 512)
	parser.add_argument('--num-epochs', type=int, default= 100)
	parser.add_argument('--dataset-folder', type= str, default= "./dataset")
	parser.add_argument('--learning-rate-classifier', type = float, default= 0.001)
	parser.add_argument('--learning-rate-barlow', type = float, default= 0.001)
	parser.add_argument('--momentum', type= float, default= 0.9)
	parser.add_argument('--weight-decay', type= float, default= 0.001)
	parser.add_argument('--fine-tune', type= int, default= 0)
	parser.add_argument('--wide', type= int, default= 0 )
	args = parser.parse_args()

	dataset_folder = args.dataset_folder
	batch_size = args.batch_size
	batch_size_val = 256 #5120
	n_epochs = args.num_epochs
	weight_decay = args.weight_decay
	checkpoint_path = args.checkpoint_path

	if torch.cuda.is_available():
		device = torch.device("cuda")
	else:
		device = torch.device("cpu")

	train_transform, val_transform = get_transforms() #TODO Get new transforms file


	labeled_train_dataset = CustomDataset(root= dataset_folder, split = "train", transform = train_transform)
	val_dataset = CustomDataset(root= dataset_folder, split = "val", transform = val_transform)

	labeled_train_loader = DataLoader(labeled_train_dataset, batch_size= batch_size, shuffle= True, num_workers= 4)
	val_loader = DataLoader(val_dataset, batch_size= batch_size_val, shuffle= False, num_workers= 4)

	if args.wide == 1:
		model_barlow = lightly.models.BarlowTwins(wide_resnet50_2(pretrained= False), num_ftrs= 2048)
	else:
		model_barlow = lightly.models.BarlowTwins(resnet18(pretrained= False), num_ftrs= 512)
	
	checkpoint = torch.load(args.transfer_path, map_location= device) 

	# print(checkpoint['state_dict'].keys())
	# print("printed keys")

	# print(model_barlow.state_dict().keys())
	# print("printed model keys")

	# if args.wide == 0:
	# 	model_barlow = torch.nn.DataParallel(model_barlow)

	model_barlow.load_state_dict(checkpoint['state_dict'])
	# print(model_barlow)
	if args.wide == 0:
		model_barlow = model_barlow.backbone
	else:
		model_barlow = model_barlow.backbone

	if args.wide == 1:
		classifier = Classifier(2048)
	else:
		classifier = Classifier(512)

	if torch.cuda.device_count() > 1:
		print("Let's use", torch.cuda.device_count(), "GPUs!")
		model_barlow = torch.nn.DataParallel(model_barlow)
		classifier = torch.nn.DataParallel(classifier)

	if args.fine_tune:
		model_barlow.requires_grad_(False)

	model_barlow = model_barlow.to(device)
	classifier = classifier.to(device)

	param_groups = [dict(params=classifier.parameters(), lr=args.learning_rate_classifier)]

	if args.fine_tune:
		param_groups.append(dict(params=model_barlow.parameters(), lr=args.learning_rate_barlow))

	optimizer = optim.Adam(param_groups, weight_decay= weight_decay)
	scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

	start_epoch = 0
	losses = Average()

	criterion = torch.nn.CrossEntropyLoss().to(device)

	for epoch in tqdm(range(start_epoch, n_epochs)):
		if args.fine_tune:
			model_barlow.train()
		else:
			model_barlow.eval()

		for batch_idx, batch in enumerate(tqdm(labeled_train_loader)):
			img = batch[0].to(device)
			labels = batch[1].to(device)

			barlow_out = model_barlow(img)
			logits = classifier(barlow_out)
			loss = criterion(logits, labels)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			scheduler.step()

			losses.update(loss.item())

			if batch_idx % 25 == 0:
				print(f"Epoch number: {epoch}, loss_avg: {losses.avg}, loss: {loss.item()}", flush= True)

		save_checkpoint({
				'epoch': epoch + 1,
				'model_barlow_state_dict': model_barlow.state_dict(),
				'classifier_state_dict': classifier.state_dict(),
				'optimizer': optimizer.state_dict(),
				'scheduler': scheduler.state_dict()
			}, checkpoint_path)

		model_barlow.eval()
		with torch.no_grad():
			val_loss = 0
			val_size = 0
			total = 0
			correct = 0
			for batch in val_loader:
				logits_val = classifier(model_barlow(batch[0].to(device)))
				labels = batch[1].to(device)
				
				val_loss += F.cross_entropy(logits_val, labels)
				_, predicted = torch.max(logits_val.data, 1)
				total += labels.size(0)
				correct += (predicted == labels).sum().item()
				val_size += 1
				# break
		print(f"Val loss: {val_loss/val_size}, Accuracy: {(100 * correct / total):.2f}%", flush= True)
	
if __name__ == '__main__':
	main()