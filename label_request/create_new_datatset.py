import torch
import csv
import os
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--label-request-path', type=str, default= "./label_request/")
parser.add_argument('--dataset-path', type=str, default= "./dataset/")
args = parser.parse_args()

with open(args.label_request_path + 'request_12.csv', newline='\n') as f:
    reader = csv.reader(f)
    data = list(reader)

img_ids = [x[0] for x in data]
# print(img_ids[0])

labels_obtained = list(torch.load(args.dataset_path + 'label_12.pt'))
labels_train = list(torch.load(args.dataset_path + 'train_label_tensor.pt'))
print(len(labels_train))

for i in tqdm(range(len(img_ids))):
    file_name = img_ids[i]
    lab = labels_obtained[i]

    labels_train.append(lab)

    new_file_name = args.dataset_path + f'train_new/{str(25600 + i)}.png'
    old_file_name = args.dataset_path + f'unlabeled/{file_name}'

    os.system(f'cp {old_file_name} {new_file_name}')

labels_train = torch.tensor(labels_train)
torch.save(labels_train, args.dataset_path + 'train_new_label.tensor.pt')

