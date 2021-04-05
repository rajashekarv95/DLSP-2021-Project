# Feel free to modifiy this file.

from torchvision import models, transforms

team_id = 1
team_name = "Self Supervised Learners"
email_address = "rv2138@nyu.edu"

def get_model():
    return models.resnet18(num_classes=800)

eval_transform = transforms.Compose([
    transforms.ToTensor(),
])