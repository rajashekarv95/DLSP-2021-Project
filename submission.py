# Feel free to modifiy this file.

from torchvision import models, transforms
import torch
from models.resnet import wide_resnet50_2
from models.resnet import resnet34, resnet18

team_id = 1
team_name = "Self Supervised Learners"
email_address = "rv2138@nyu.edu"

def get_model():
    return resnet34(pretrained=False, num_classes = 800)

mean = (0.4836, 0.4527, 0.4011)
std = (0.3065, 0.2728, 0.2355)

eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])