import torch
import torch.nn.functional as F
import lightly
from collections import OrderedDict

import argparse

from models.resnet_barlow import wide_resnet50_2
from models.classifier import Classifier

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--dest-path', type=str, default= "./checkpoints/model_combined.pth.tar")
	parser.add_argument('--source-path', type=str, default= "./checkpoints/model.pth")

	args = parser.parse_args()

	if torch.cuda.is_available():
		device = torch.device("cuda")
	else:
		device = torch.device("cpu")

	
	model = lightly.models.BarlowTwins(wide_resnet50_2(pretrained= False), num_ftrs= 2048)
	model = model.backbone
	classifier = Classifier(ip = 2048, dp= 0)

	checkpoint = torch.load(args.source_path, map_location= device) 	

	model.load_state_dict(checkpoint['model_state_dict'])
	classifier.load_state_dict(checkpoint['classifier_state_dict'])
	print(f"Best val accuracy for this model is {checkpoint['best_val_accuracy']}", flush= True)
	

	model_final = torch.nn.Sequential(OrderedDict([
					('backbone', model),
					('classifier', classifier)]))
	

	torch.save(model_final.state_dict(), args.target_path)

	print("Saved model", flush= True)
if __name__ == '__main__':
	main()