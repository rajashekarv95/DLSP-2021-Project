import torch
import torch.nn as nn
import torchvision
import lightly

import argparse
import numpy as np
from tqdm import tqdm

from transforms import get_transforms, TransformMoCo
from dataloader import CustomDataset
from models.moco import MocoModel
from utils.misc import Average

def save_checkpoint(state, checkpoint_path):
	torch.save(state, checkpoint_path)

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--checkpoint-path', type=str, default= "./model.pth")
	parser.add_argument('--batch-size', type=int, default= 10)
	parser.add_argument('--num-epochs', type=int, default= 10)
	parser.add_argument('--train-from-start', type= int, default= 1)
	parser.add_argument('--dataset-folder', type= str, default= "./dataset")
	parser.add_argument('--learning-rate', type = float, default= 0.01)
	parser.add_argument('--momentum', type= float, default= 0.9)
	parser.add_argument('--weight-decay', type= float, default= 1.5*1e-6)
	parser.add_argument('--memory-bank-size', type= int, default= 4096)
	args = parser.parse_args()

	unlabeled_train_dataset = CustomDataset(root= args.dataset_folder, split = "unlabeled", transform = TransformMoCo())

	unlabeled_train_loader = torch.utils.data.DataLoader(
						unlabeled_train_dataset,
						batch_size= args.batch_size,
						shuffle=True,
						num_workers=4)
						
	if torch.cuda.is_available():
		device = torch.device("cuda:0")
	else:
		device = torch.device("cpu")

	resnet = lightly.models.ResNetGenerator('resnet-18', 1, num_splits=0)
	backbone = nn.Sequential(
			*list(resnet.children())[:-1],
			nn.AdaptiveAvgPool2d(1),
		)

	# create a moco based on ResNet
	moco_model = lightly.models.MoCo(backbone, num_ftrs=512, m=0.99, batch_shuffle=True)

	# create our loss with the optional memory bank
	criterion = lightly.loss.NTXentLoss(
		temperature=0.1,
		memory_bank_size=args.memory_bank_size)

	optimizer = torch.optim.SGD(moco_model.parameters(), lr=6e-2,
								momentum=0.9, weight_decay=5e-4)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs)

	start_epoch = 0

	if torch.cuda.device_count() > 1:
		print("Let's use", torch.cuda.device_count(), "GPUs!")
		moco_model = torch.nn.DataParallel(moco_model)
		multi_gpu = 1
	else:
		multi_gpu = 0

	moco_model = moco_model.to(device)

	losses = Average()

	for epoch in tqdm(range(start_epoch, args.num_epochs)):
		moco_model.train()
		for batch_idx, batch in enumerate(tqdm(unlabeled_train_loader)):
			x0 = batch[0][0].to(device)
			x1 = batch[0][1].to(device)

			y0, y1 = moco_model(x0, x1)

			loss = criterion(y0, y1)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			scheduler.step()
			
			losses.update(loss.item())
			if batch_idx % 25 == 0:
				print(f"Epoch number: {epoch}, loss_avg: {losses.avg}, loss: {loss.item()}, lr: {optimizer.param_groups[0]['lr']}", flush= True)

		if multi_gpu == 1:
			save_checkpoint({
				'epoch': epoch + 1,
				'state_dict': moco_model.module.state_dict(),
				'optimizer': optimizer.state_dict(),
				'scheduler': scheduler.state_dict()
			}, args.checkpoint_path)
		else:
			save_checkpoint({
				'epoch': epoch + 1,
				'state_dict': moco_model.state_dict(),
				'optimizer': optimizer.state_dict(),
				'scheduler': scheduler.state_dict()
			}, args.checkpoint_path)
			


if __name__ == '__main__':
	main()