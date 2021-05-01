import argparse
import logging
import math
import os
import random
import time
from collections import OrderedDict
from matplotlib import pyplot as plt
import argparse
import itertools

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

from models.resnet_barlow import wide_resnet50_2
from models.resnet_barlow import resnet34, resnet18
from models.classifier import Classifier

from utils.misc import Average

random.seed(10)
np.random.seed(10)
torch.manual_seed(10)
if torch.cuda.is_available():
	torch.cuda.manual_seed(10)

torch.backends.cudnn.deterministic=True

def get_cosine_schedule_with_warmup(optimizer,
									num_warmup_steps,
									num_training_steps,
									num_cycles=7./16.,
									last_epoch=-1):
	def _lr_lambda(current_step):
		if current_step < num_warmup_steps:
			return float(current_step) / float(max(1, num_warmup_steps))
		no_progress = float(current_step - num_warmup_steps) / \
			float(max(1, num_training_steps - num_warmup_steps))
		return max(0., math.cos(math.pi * num_cycles * no_progress))

	return LambdaLR(optimizer, _lr_lambda, last_epoch)

def save_checkpoint(state, checkpoint_path):
	torch.save(state, checkpoint_path)

def main():
	#TODO: Get args
	# python3 train_fixmatch.py --checkpoint-path ./checkpoint_path/model.pth --batch-size 1 --num-epochs 1 --num-steps 1 --train-from-start 1 --dataset-folder ./dataset
	parser = argparse.ArgumentParser()
	parser.add_argument('--checkpoint-path', type=str, default= "./checkpoints/model_fm_transfer.pth.tar")
	parser.add_argument('--transfer-path', type=str, default= "./checkpoints/model_transfer.pth.tar")
	parser.add_argument('--best-path', type= str, default= "./checkpoints/model_barlow_best.pth.tar")
	parser.add_argument('--batch-size', type=int, default= 64)
	parser.add_argument('--num-epochs', type=int, default= 10)
	parser.add_argument('--num-steps', type=int, default= 10)
	parser.add_argument('--train-from-start', type= int, default= 1)
	parser.add_argument('--dataset-folder', type= str, default= "./dataset")
	parser.add_argument('--learning-rate', type = float, default= 0.01)
	parser.add_argument('--threshold', type = float, default= 0.5)
	parser.add_argument('--mu', type= int, default= 7)
	parser.add_argument('--lambd', type= int, default= 1)
	parser.add_argument('--momentum', type= float, default= 0.9)
	parser.add_argument('--weight-decay', type= float, default= 0.001)
	parser.add_argument('--layers', type= int, default= 18)
	parser.add_argument('--fine-tune', type= int, default= 1)
	args = parser.parse_args()

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
	weight_decay = args.weight_decay
	checkpoint_path = args.checkpoint_path
	train_from_start = args.train_from_start
	n_layers = args.layers

	if torch.cuda.is_available():
		device = torch.device("cuda")
	else:
		device = torch.device("cpu")

	# print("pwd: ", os.getcwd())
	train_transform, val_transform = get_transforms()

	labeled_train_dataset = CustomDataset(root= dataset_folder, split = "train", transform = train_transform)
	unlabeled_train_dataset = CustomDataset(root= dataset_folder, 
											split = "unlabeled", 
											transform = TransformFixMatch(mean = 0, std = 0))#TODO
											
	val_dataset = CustomDataset(root= dataset_folder, split = "val", transform = val_transform)

	labeled_train_loader = DataLoader(labeled_train_dataset, batch_size= batch_size_labeled, shuffle= True, num_workers= 4)
	unlabeled_train_loader = DataLoader(unlabeled_train_dataset, batch_size= batch_size_unlabeled, shuffle= True, num_workers= 4)
	val_loader = DataLoader(val_dataset, batch_size= batch_size_val, shuffle= False, num_workers= 4)



	labeled_iter = iter(labeled_train_loader)
	unlabeled_iter = iter(unlabeled_train_loader)

	# model = torchvision.models.wide_resnet50_2(pretrained= False, num_classes = num_classes)
	# if n_layers == 18:
	# 	model = resnet18(pretrained=False, num_classes = 800)
	# 	classifier = Classifier(ip= 512, dp = 0)
	# elif n_layers == 34:
	# 	model = resnet34(pretrained=False, num_classes = 800)
	# 	classifier = Classifier(ip= 2048, dp = 0)
	# else:
	# 	raise ValueError("Wrong value passed for layers")

	model = wide_resnet50_2(pretrained=False, num_classes = 800)
	classifier = Classifier(ip= 2048, dp = 0)
	start_epoch = 0

	# checkpoint = torch.load(args.transfer_path, map_location= device)
	checkpoint = torch.load(args.transfer_path)
	model.load_state_dict(checkpoint['model_state_dict'])
	classifier.load_state_dict(checkpoint['classifier_state_dict'])

	param_groups = [dict(params=classifier.parameters(), lr=args.learning_rate)]

	if args.fine_tune:
		param_groups.append(dict(params=model.parameters(), lr=args.learning_rate))

	optimizer = torch.optim.SGD(param_groups, 
								lr = learning_rate,
								momentum= momentum,
								nesterov= True,
								weight_decay= weight_decay)

	scheduler = get_cosine_schedule_with_warmup(optimizer, 0, num_training_steps= n_epochs * n_steps)

	if torch.cuda.device_count() > 1:
		print("Let's use", torch.cuda.device_count(), "GPUs!")
		model = torch.nn.DataParallel(model)
		classifier = torch.nn.DataParallel(classifier)

	if train_from_start == 0:
		assert os.path.isfile(checkpoint_path), "Error: no checkpoint directory found!"
		print("Restoring model from checkpoint")
		# args.out = os.path.dirname(args.resume)
		checkpoint = torch.load(checkpoint_path)
		# best_acc = checkpoint['best_acc']
		start_epoch = checkpoint['epoch'] - 1
		model.load_state_dict(checkpoint['backbone_state_dict'])
		classifier.load_state_dict(checkpoint['classifier_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		scheduler.load_state_dict(checkpoint['scheduler'])

	model = model.to(device)
	classifier = classifier.to(device)
	

	model.train()
	losses = Average()
	losses_l = Average()
	losses_u = Average()
	mask_probs = Average()
	best_val_accuracy = 25.0 #TODO

	for epoch in tqdm(range(start_epoch, n_epochs)):
		if args.fine_tune:
			model.train()
			classifier.train()
		else:
			model.eval()
			classifier.train()

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

			img_cat = torch.cat((img_lab, img_weak, img_strong), dim = 0)
			logits_cat = classifier(model(img_cat))
			logits_lab = logits_cat[:batch_size_labeled]
			# print(logits_lab.size())
			logits_unlab = logits_cat[batch_size_labeled:]
			# print(logits_unlab)

			logits_weak, logits_strong = torch.chunk(logits_unlab, chunks= 2, dim = 0)

			pseudo_label = torch.softmax(logits_weak.detach()/tau, dim= 1)
			max_probs, targets_unlab = torch.max(pseudo_label, dim= 1)
			mask = max_probs.ge(threshold).float()
			
			loss_labeled = F.cross_entropy(logits_lab, targets_lab, reduction='mean')

			# print("CE: ", F.cross_entropy(logits_strong, targets_unlab, reduction= 'none').size())

			loss_unlabeled = (F.cross_entropy(logits_strong, targets_unlab, reduction= 'none') * mask).mean()

			# print("Loss labelled, loss unlabelled: ", loss_labeled, loss_unlabeled)

			loss_total = loss_labeled + lamd * loss_unlabeled

			# print("Total loss: ", loss_total)
			# loss_epoch += loss_total
			# loss_lab_epoch += loss_labeled
			# loss_unlab_epoch += loss_unlabeled
			losses.update(loss_total.item())
			losses_l.update(loss_labeled.item())
			losses_u.update(loss_unlabeled.item())
			mask_probs.update(mask.mean().item())

			optimizer.zero_grad()
			loss_total.backward()
			optimizer.step()
			scheduler.step()


			# break
			if batch_idx % 25 == 0:
				print(f"Epoch number: {epoch}, loss: {losses.avg}, loss lab: {losses_l.avg}, loss unlab: {losses_u.avg}, mask: {mask_probs.avg}, loss_here: {loss_total.item()}, best accuracy: {best_val_accuracy:.2f}", flush= True)
			# print(optimizer.param_groups[0]['lr'])
		

		save_checkpoint({
				'epoch': epoch + 1,
				'model_state_dict': model.state_dict(),
				'classifier_state_dict': model.state_dict(),
				'optimizer': optimizer.state_dict(),
				'scheduler': scheduler.state_dict(),
			}, checkpoint_path)

		model.eval()
		classifier.eval()
		with torch.no_grad():
			val_loss = 0
			val_size = 0
			total = 0
			correct = 0
			for batch in val_loader:
				logits_val = classifier(model(batch[0].to(device)))
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
		model.train()
		classifier.train()

		# break

	

if __name__ == '__main__':
	main()