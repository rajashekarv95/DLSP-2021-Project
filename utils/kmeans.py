import numpy as np
from tqdm import tqdm

def kMeans(X, K, maxIters = 10):

    centroids = X[np.random.choice(np.arange(len(X)), K), :]
    for i in tqdm(range(maxIters)):
        # Cluster Assignment step
        C = np.array([np.argmax([np.dot(x_i, y_k)/(np.linalg.norm(x_i)*np.linalg.norm(y_k)) for y_k in centroids]) for x_i in X])
        # Move centroids step
        centroids = [X[C == k].mean(axis = 0) for k in range(K)]
    return np.array(centroids) , C