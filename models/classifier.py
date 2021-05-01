import torch
import torch.nn.functional as F

class Classifier(torch.nn.Module):

	def __init__(self, ip, dp):
		super().__init__()
		self.fc1 = torch.nn.Linear(ip, 8192)
		self.fc2 = torch.nn.Linear(8192, 8192)
		self.fc3 = torch.nn.Linear(8192, 8192)
		self.fc4 = torch.nn.Linear(8192, 8192)
		self.fc5 = torch.nn.Linear(8192, 8192)
		self.fc6 = torch.nn.Linear(8192, 8192)
		self.fc7 = torch.nn.Linear(8192, 800)
		self.bn0 = torch.nn.BatchNorm1d(ip)
		self.bn1 = torch.nn.BatchNorm1d(8192)
		self.bn2 = torch.nn.BatchNorm1d(8192)
		self.bn3 = torch.nn.BatchNorm1d(8192)
		self.bn4 = torch.nn.BatchNorm1d(8192)
		self.bn5 = torch.nn.BatchNorm1d(8192)
		self.bn6 = torch.nn.BatchNorm1d(8192)
		self.dropout = torch.nn.Dropout(dp)

	def forward(self, x):
		# TODO
		x = self.bn0(x)
		x = self.bn1(self.fc1(x))
		x = F.relu(x)
		x = self.dropout(x)
		x = self.bn2(self.fc2(x))
		x = F.relu(x)
		x = self.dropout(x)
		x = self.bn3(self.fc3(x))
		x = F.relu(x)
		x = self.dropout(x)
		x = self.bn4(self.fc4(x))
		x = F.relu(x)
		x = self.dropout(x)
		x = self.bn5(self.fc5(x))
		x = F.relu(x)
		x = self.dropout(x)
		x = self.bn6(self.fc6(x))
		x = F.relu(x)
		x = self.dropout(x)
		x = self.fc7(x)

		return x