import torch
import lightly
import argparse
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
import csv

from dataloader_with_fname import CustomDataset
from submission import get_model, eval_transform

from torch.utils.data import DataLoader
from models.resnet_barlow import wide_resnet50_2, resnet18
from models.classifier import Classifier

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--checkpoint-path', type=str, default= "./checkpoints/model_transfer_barlow_best.pth.tar")
	parser.add_argument('--dataset-folder', type= str, default= "./dataset")
	parser.add_argument('--lab-path', type= str, default= "./label_request/label_request.csv")
	parser.add_argument('--batch-size', type= int, default= 10)
	parser.add_argument('--wide' ,type= int, default= 0)
	parser.add_argument('--dropout', type= float, default= 0.0)
	parser.add_argument('--samples', type= int, default= 5)
	args = parser.parse_args()

	if torch.cuda.is_available():
		device = torch.device("cuda")
	else:
		device = torch.device("cpu")

	unlabeled_train_dataset = CustomDataset(root= args.dataset_folder, split = "unlabeled", transform = eval_transform)
	unlabeled_dataloader = DataLoader(unlabeled_train_dataset, batch_size= args.batch_size, shuffle= False, num_workers= 4)

	if args.wide:
		model = lightly.models.BarlowTwins(wide_resnet50_2(pretrained= False), num_ftrs= 2048)
	else:
		model = lightly.models.BarlowTwins(resnet18(pretrained= False), num_ftrs= 512)


	if args.wide == 1:
		classifier = Classifier(ip = 2048, dp= args.dropout)
	else:
		classifier = Classifier(ip = 512, dp = args.dropout)

	model = model.backbone

	checkpoint = torch.load(args.checkpoint_path, map_location= device) 
	model.load_state_dict(checkpoint['model_state_dict'])
	classifier.load_state_dict(checkpoint['classifier_state_dict'])

	model = model.to(device)
	classifier = classifier.to(device)

	# combined = torch.nn.Sequential(OrderedDict([('backbone', model), ('classifier', classifier)]))
	# print(combined)

	# print(model.state_dict())

	# print('____________________________________________________________')
	# print(combined.backbone.state_dict())

	model.eval()
	classifier.eval()
	entropy_all = torch.tensor([]).to(device)
	idx_all = torch.tensor([]).to(device)
	with torch.no_grad():
		for batch_idx, batch in enumerate(tqdm(unlabeled_dataloader)):
			img = batch[0].to(device)
			idx = batch[2].to(device)

			logits = classifier(model(img))
			log_probs = torch.nn.LogSoftmax(dim = 1)(logits)
			probs = torch.exp(log_probs)

			entropy = 0 - torch.sum(torch.mul(probs, torch.log2(probs)), dim =  1) 

			entropy_all = torch.cat((entropy_all, entropy), dim = 0)
			idx_all = torch.cat((idx_all, idx), dim = 0)
			print(entropy_all)
			print(idx_all)
		
		sort_idx = torch.argsort(entropy_all, descending= True)
		# print(idx_all[sort_idx])
		samp = idx_all[sort_idx][:args.samples]
		samp = samp.detach().cpu().numpy()
		samp = samp.astype(int)
		samp = [str(x)+".png,\n" for x in samp]
		file_ = open(args.lab_path, 'w', newline ='\n')
		with file_:
			for samp_idx, val in enumerate(samp):
				if samp_idx == len(samp) - 1:
					file_.write(val[:-2])
				else:
					file_.write(val)
		
if __name__ == '__main__':
	main()