# Feel free to modifiy this file.

from torchvision import models, transforms
import torch

team_id = 1
team_name = "Self Supervised Learners"
email_address = "rv2138@nyu.edu"

def get_model():
    return torch.hub.load('pytorch/vision:v0.9.0', 'wide_resnet50_2', pretrained=False, num_classes = 800)

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)

eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])