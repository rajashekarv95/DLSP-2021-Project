from sklearn.cluster import KMeans
import torch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--representations-path', type=str, default= "./representations/")
parser.add_argument('--dest-path', type=str, default= "./label_request/")
args = parser.parse_args()

lab_rep_model = torch.load(args.representations_path + 'lab_rep_model.pt', map_location= 'cpu').numpy()
unlab_rep_model = torch.load(args.representations_path + 'unlab_rep_model.pt', map_location= 'cpu').numpy()

print("Loaded data")

lab_rep_model = lab_rep_model / np.linalg.norm(lab_rep_model, axis = 1)[:, np.newaxis]
print("Normalized")

km_model = KMeans(n_clusters= 5)
km_model.fit(X= lab_rep_model)

centroids = km_model.cluster_centers_
print("Fitted")
print("Inertia: ", km_model.inertia_)

print("Centroids shape: ", centroids.shape)
print("Rep shape: ", unlab_rep_model.shape)

preds = np.matmul(unlab_rep_model, centroids.T)

pred_min = preds.min(axis = 1)

max_pred_min_idx = np.argsort(- pred_min)
lab_req_kmeans = list(max_pred_min_idx[:12800])

np.save('./label_request/lab_kmeans_5.npy', lab_req_kmeans)

samp = [int(i) for i in lab_req_kmeans]
samp = [str(x)+".png,\n" for x in samp]
file_ = open(args.dest_path + "request_12.csv", 'w', newline ='\n')
with file_:
    for samp_idx, val in enumerate(samp):
        if samp_idx == len(samp) - 1:
            file_.write(val[:-2])
        else:
            file_.write(val)
print("Written to file", flush= True)