import torch
import sys
import numpy as np
import os
import yaml
import matplotlib.pyplot as plt
import torchvision
from tqdm import tqdm

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets

from dataloader import CustomDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("Using device:", device)

trainset = CustomDataset(root='/dataset', split='train', transform=transforms.ToTensor())
train_loader = DataLoader(trainset, batch_size=32,
                            num_workers=0, drop_last=False, shuffle=False)

evalset = CustomDataset(root='/dataset', split='val', transform=transforms.ToTensor())
validation_loader = DataLoader(evalset, batch_size=256,
                            num_workers=0, drop_last=False, shuffle=False)

model = torchvision.models.resnet50(pretrained=False, num_classes=800).to(device)

checkpoint = torch.load("/scratch/sm9669/checkpoints/model.pth")
state_dict = checkpoint['state_dict']

#state_dict = torch.load("/scratch/sm9669/checkpoints/modelsup.pth")
# state_dict = checkpoint['state_dict']

for k in list(state_dict.keys()):

  if k.startswith('backbone.'):
    if k.startswith('backbone') and not k.startswith('backbone.fc'):
      # remove prefix
      state_dict[k[len("backbone."):]] = state_dict[k]
  del state_dict[k]

  log = model.load_state_dict(state_dict, strict=False)
#assert log.missing_keys == ['fc.weight', 'fc.bias']

# freeze all layers but the last fc
for name, param in model.named_parameters():
    if name not in ['fc.weight', 'fc.bias']:
        param.requires_grad = False

parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
assert len(parameters) == 2  # fc.weight, fc.bias

#optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=0.0008)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0008)
criterion = torch.nn.CrossEntropyLoss().to(device)

sup_checkpoint_path = "/scratch/sm9669/checkpoints/modelsup.pth"
epochs = 600
model.train()

for epoch in tqdm(range(epochs)):
	loss_epoch = 0.0
	correct = 0
	total = 0
	for counter, (x_batch, y_batch) in enumerate(train_loader):
		x_batch = x_batch.to(device)
		y_batch = y_batch.to(device)

		logits = model(x_batch)
		loss = criterion(logits, y_batch)
		loss_epoch = loss_epoch+loss
		_, predicted = torch.max(logits.data, 1)
		total += y_batch.size(0)
		correct += (predicted == y_batch).sum().item()

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	print(f"Epoch number: {epoch}, trainloss: {loss_epoch}, train accuracy: {(100 * correct / total):.2f}", flush= True)
	torch.save(model.state_dict(), sup_checkpoint_path)
	model.eval()
	with torch.no_grad():
		val_correct=0
		val_total=0
		val_loss_epoch=0
		for counter, (x_batch, y_batch) in enumerate(validation_loader):
			x_batch = x_batch.to(device)
			y_batch = y_batch.to(device)

			logits = model(x_batch)

			val_loss_epoch = val_loss_epoch +criterion(logits, y_batch)
			_, predicted = torch.max(logits.data, 1)
			val_total += y_batch.size(0)
			val_correct += (predicted == y_batch).sum().item()

		print(f"Epoch number: {epoch}, val loss: {val_loss_epoch}, val accuracy: {(100 * val_correct / val_total):.2f}", flush= True)