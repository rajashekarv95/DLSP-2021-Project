import torch 
import lightly
import argparse
from tqdm import tqdm

from dataloader_with_fname import CustomDataset
from torch.utils.data import DataLoader
from models.resnet_barlow import wide_resnet50_2, resnet18
from submission import get_model, eval_transform

from models.classifier import Classifier

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--checkpoint-path', type=str, default= "./checkpoints/model_transfer_barlow_best.pth.tar")
	parser.add_argument('--dataset-folder', type= str, default= "./dataset")
	parser.add_argument('--out-path', type= str, default= "./representations/")
	parser.add_argument('--batch-size', type= int, default= 512)
	parser.add_argument('--wide' ,type= int, default= 0)
	parser.add_argument('--dropout', type= float, default= 0.0)
	parser.add_argument('--final', type= int, default= 0)
	args = parser.parse_args()

	if torch.cuda.is_available():
		device = torch.device("cuda")
	else:
		device = torch.device("cpu")

	unlabeled_train_dataset = CustomDataset(root= args.dataset_folder, split = "unlabeled", transform = eval_transform)
	unlabeled_dataloader = DataLoader(unlabeled_train_dataset, batch_size= args.batch_size, shuffle= False, num_workers= 4)

	labeled_train_dataset = CustomDataset(root= args.dataset_folder, split = "train", transform = eval_transform)
	labeled_dataloader = DataLoader(labeled_train_dataset, batch_size= args.batch_size, shuffle= False, num_workers= 4)

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

	model.eval()
	classifier.eval()
	label_rep_model = torch.tensor([]).to(device)
	label_rep_clf = torch.tensor([]).to(device)
	unlabel_rep_model = torch.tensor([]).to(device)
	unlabel_rep_clf = torch.tensor([]).to(device)

	with torch.no_grad():
		for batch_idx, batch in enumerate(tqdm(labeled_dataloader)):
			img = batch[0].to(device)
			
			logits_model = model(img)
			logits_classifier = classifier(logits_model)

			label_rep_model = torch.cat((label_rep_model, logits_model), dim = 0)
			label_rep_clf = torch.cat((label_rep_clf, logits_classifier), dim = 0)

		print("Writing labeled representations to file", flush= True)
		lab_path = args.out_path + "lab_rep_model.pt"
		torch.save(label_rep_model.detach(), lab_path)

		lab_path = args.out_path + "lab_rep_clf.pt"
		torch.save(label_rep_clf.detach(), lab_path)

		for batch_idx, batch in enumerate(tqdm(unlabeled_dataloader)):
			img = batch[0].to(device)

			logits_model = model(img)
			logits_classifier = classifier(logits_model)

			unlabel_rep_model = torch.cat((unlabel_rep_model, logits_model), dim = 0)
			unlabel_rep_clf = torch.cat((unlabel_rep_clf, logits_classifier), dim = 0)

		print("Writing unlabeled representations to file", flush= True)
		unlab_path = args.out_path + "unlab_rep_model.pt"
		torch.save(unlabel_rep_model.detach(), unlab_path)

		unlab_path = args.out_path + "unlab_rep_clf.pt"
		torch.save(unlabel_rep_clf.detach(), unlab_path)


if __name__ == '__main__':
	main()