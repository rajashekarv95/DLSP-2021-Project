import torch
import torch.nn.functional as F

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