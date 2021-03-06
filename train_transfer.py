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
from models.classifier import Classifier

from utils.misc import Average


def exclude_bias_and_norm(p):
	return p.ndim == 1

def save_checkpoint(state, checkpoint_path):
	torch.save(state, checkpoint_path)

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--checkpoint-path', type=str, default= "./checkpoints/model_transfer.pth.tar")
	parser.add_argument('--transfer-path', type=str, default= "./checkpoints/model_barlow.pth.tar")
	parser.add_argument('--best-path', type= str, default= "./checkpoints/model_barlow_best.pth.tar")
	parser.add_argument('--batch-size', type=int, default= 10)
	parser.add_argument('--num-epochs', type=int, default= 100)
	parser.add_argument('--dataset-folder', type= str, default= "./dataset")
	parser.add_argument('--new-dataset-folder', type= str, default= "./dataset")
	parser.add_argument('--learning-rate-classifier', type = float, default= 0.001)
	parser.add_argument('--learning-rate-model', type = float, default= 0.001)
	parser.add_argument('--momentum', type= float, default= 0.9)
	parser.add_argument('--weight-decay', type= float, default= 0.001)
	parser.add_argument('--fine-tune', type= int, default= 0)
	parser.add_argument('--wide', type= int, default= 0)
	parser.add_argument('--model-name', type= str, default="moco")
	parser.add_argument('--dropout', type= float, default= 0)
	parser.add_argument('--new-data', type= int, default= 0)
	parser.add_argument('--seed', type = int, default= 0)
	args = parser.parse_args()

	dataset_folder = args.dataset_folder
	batch_size = args.batch_size
	batch_size_val = 256 #5120
	n_epochs = args.num_epochs
	weight_decay = args.weight_decay
	checkpoint_path = args.checkpoint_path

	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(args.seed)

	torch.backends.cudnn.deterministic=True

	print(f"Training with seed {args.seed}")

	if torch.cuda.is_available():
		device = torch.device("cuda")
	else:
		device = torch.device("cpu")

	train_transform, val_transform = get_transforms() #TODO Get new transforms file

	if args.new_data == 0:
		labeled_train_dataset = CustomDataset(root= args.dataset_folder, split = "train", transform = train_transform)
	else:
		labeled_train_dataset = CustomDataset(root= args.new_dataset_folder, split = "train_new", transform = train_transform)
	val_dataset = CustomDataset(root= args.dataset_folder, split = "val", transform = val_transform)

	labeled_train_loader = DataLoader(labeled_train_dataset, batch_size= batch_size, shuffle= True, num_workers= 4)
	val_loader = DataLoader(val_dataset, batch_size= batch_size_val, shuffle= False, num_workers= 4)

	resnet = lightly.models.ResNetGenerator('resnet-18', 1, num_splits=0)
	backbone = torch.nn.Sequential(
			*list(resnet.children())[:-1],
			torch.nn.AdaptiveAvgPool2d(1),
		)

	if args.model_name == "moco":
		model = lightly.models.MoCo(backbone, num_ftrs= 512, m=0.99, batch_shuffle=True)
	else:
		if args.wide == 1:
			model = lightly.models.BarlowTwins(wide_resnet50_2(pretrained= False), num_ftrs= 2048)
		else:
			model = lightly.models.BarlowTwins(resnet18(pretrained= False), num_ftrs= 512)
	
	checkpoint = torch.load(args.transfer_path, map_location= device) 

	# print(checkpoint['state_dict'].keys())
	# print("printed keys")

	# print(model_barlow.state_dict().keys())
	# print("printed model keys")

	# if args.wide == 0:
		# model = torch.nn.DataParallel(model)

	model.load_state_dict(checkpoint['state_dict'])
	# print(model_barlow)
	if args.wide == 0:
		model = model.backbone
	else:
		model = model.backbone

	if args.wide == 1:
		classifier = Classifier(ip = 2048, dp= args.dropout)
	else:
		classifier = Classifier(ip = 512, dp = args.dropout)

	if torch.cuda.device_count() > 1:
		print("Let's use", torch.cuda.device_count(), "GPUs!")
		model = torch.nn.DataParallel(model)
		classifier = torch.nn.DataParallel(classifier)

	if not args.fine_tune:
		model.requires_grad_(False)

	model = model.to(device)
	classifier = classifier.to(device)

	param_groups = [dict(params=classifier.parameters(), lr=args.learning_rate_classifier)]

	if args.fine_tune:
		param_groups.append(dict(params=model.parameters(), lr=args.learning_rate_model))

	optimizer = optim.Adam(param_groups, weight_decay= weight_decay)
	scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

	start_epoch = 0
	losses = Average()

	criterion = torch.nn.CrossEntropyLoss().to(device)

	best_val_accuracy = 25.0 #TODO

	for epoch in tqdm(range(start_epoch, n_epochs)):
		if args.fine_tune:
			model.train()
		else:
			model.eval()
		classifier.train()

		for batch_idx, batch in enumerate(tqdm(labeled_train_loader)):
			img = batch[0].to(device)
			labels = batch[1].to(device)

			model_out = model(img)
			if args.model_name == "moco":
				model_out = model_out.squeeze()
				model_out = torch.nn.functional.normalize(model_out, dim=1)
			logits = classifier(model_out)
			loss = criterion(logits, labels)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			scheduler.step()

			losses.update(loss.item())

			if batch_idx % 25 == 0:
				print(f"Epoch number: {epoch}, loss_avg: {losses.avg}, loss: {loss.item()}, best accuracy: {best_val_accuracy:.2f}", flush= True)

		save_checkpoint({
				'epoch': epoch + 1,
				'model_state_dict': model.state_dict(),
				'classifier_state_dict': classifier.state_dict(),
				'optimizer': optimizer.state_dict(),
				'scheduler': scheduler.state_dict()
			}, checkpoint_path)

		model.eval()
		with torch.no_grad():
			val_loss = 0
			val_size = 0
			total = 0
			correct = 0
			for batch in val_loader:
				model_out  = model(batch[0].to(device))
				if args.model_name == "moco":
					model_out = model_out.squeeze()
					model_out = torch.nn.functional.normalize(model_out, dim=1)
				logits_val = classifier(model_out)
				labels = batch[1].to(device)
				
				val_loss += F.cross_entropy(logits_val, labels)
				_, predicted = torch.max(logits_val.data, 1)
				total += labels.size(0)
				correct += (predicted == labels).sum().item()
				val_size += 1
				# break
		print(f"Val loss: {val_loss/val_size}, Accuracy: {(100 * correct / total):.2f}%", flush= True)

		if 100 * correct / total > best_val_accuracy:
			best_val_accuracy = 100 * correct / total
			best_val_loss = val_loss/val_size
			print(f"Saving the best model with {best_val_accuracy:.2f}% accuracy and {best_val_loss:.2f} loss", flush= True)
			save_checkpoint({
				'epoch': epoch + 1,
				'model_state_dict': model.state_dict(),
				'classifier_state_dict': classifier.state_dict(),
				'optimizer': optimizer.state_dict(),
				'scheduler': scheduler.state_dict(),
				'best_val_accuracy': best_val_accuracy,
				'best_val_loss': best_val_loss
			}, args.best_path)
	
if __name__ == '__main__':
	main()