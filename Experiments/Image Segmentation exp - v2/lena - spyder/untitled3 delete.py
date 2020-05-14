# -*- coding: utf-8 -*-
"""
Created on Thu May 14 15:33:14 2020

@author: Henock
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May 14 14:34:58 2020

@author: Henock
"""

import numpy as np
import pandas as pd
import cv2 as cv
from matplotlib import pyplot as plt
import PsnrSnr as p
import kmeansAlgo
from sklearn.utils import shuffle
from sklearn.metrics import pairwise_distances


def kmeans(X, k, maxiter, seed = None):
    """
    specify the number of clusters k and
    the maximum iteration to run the algorithm
    """
    n_row, n_col = X.shape

    # randomly choose k data points as initial centroids
    if seed is not None:
        np.random.seed(seed)
    
    rand_indices = np.random.choice(n_row, size = k)
    centroids = X[rand_indices]

    for itr in range(maxiter):
        # compute distances between each data point and the set of centroids
        # and assign each data point to the closest centroid
        distances_to_centroids = pairwise_distances(X, centroids, metric = 'euclidean')
        cluster_assignment = np.argmin(distances_to_centroids, axis = 1)

        # select all data points that belong to cluster i and compute
        # the mean of these data points (each feature individually)
        # this will be our new cluster centroids
        new_centroids = np.array([X[cluster_assignment == i].mean(axis = 0) for i in range(k)])
        
        # if the updated centroid is still the same,
        # then the algorithm converged
        if np.all(centroids == new_centroids):
            break
        
        centroids = new_centroids
    
    return centroids, cluster_assignment


img = cv.imread("lena_gray.jpg", 0)
size=256
img = cv.resize(img, dsize=(size, size)) 
width=img.shape[0]
height=img.shape[1]

plt.imshow(img, cmap="gray")
plt.xticks([])
plt.yticks([])
plt.grid(False)
plt.show()



i = 0
j = 0
luv = []

for h in range(int(width / 2)):
    j = 0
    for k in range(int(height / 2)):
        luv.append([img[i, j], img[i + 1, j], img[i, j + 1], img[i + 1, j + 1]])
        j = j + 2
    i = i + 2


x = np.array(luv)

temp=np.mean(luv,axis=1)
luvtemp=np.array(luv.copy())

for i in range (luvtemp.shape[0]):
    luvtemp[i,:]=temp[i] 

x=luvtemp



import time as t

cluster = 10

clusterssize = [cluster]  # 2, 3, 4, 5, 6, 7, 8, 9,
timearray = []
error = []
segementedImg = []


for i in clusterssize:

    start = t.time()
    centers, label=kmeans(x,cluster,50,np.random.randint(0,500))  
    timearray.append(t.time() - start)

    segementedImg.append(label)

    pic2show = centers
    print('timearray',timearray)
    
    
    
lables = np.array(label)
print(np.array(label).shape)
print(centers)

clustercenters = (centers).reshape(cluster, 4)
centersp = (centers).reshape(cluster, 4)
centersp.shape


finalval = []
for i in range(lables.shape[0]):
    finalval.append(centersp[lables[i]])

finalval = np.array(finalval).reshape(int(width / 2), int(height / 2), 4)


# convert back the clustered image to the original form
# decoding

temp = np.zeros((width, height))
i = 0
j = 0
for h in range(int(width / 2)):
    j = 0
    for k in range(int(height / 2)):
        val = finalval

        temp[i, j] = val[h][k][0]
        temp[i + 1, j] = val[h][k][1]
        temp[i, j + 1] = val[h][k][2]
        temp[i + 1, j + 1] = val[h][k][3]

        j = j + 2
    i = i + 2

plt.imshow(temp, cmap="gray")
plt.xticks([])
plt.yticks([])
plt.grid(False)
plt.savefig("Kmeans_SegementdIMg_" + str(cluster) + ".jpg")
plt.show()


plt.imshow(256 - abs(img - temp), cmap="gray")
plt.xticks([])
plt.yticks([])
plt.grid(False)

plt.savefig("Kmeans_SegementdIMg_error_" + str(cluster) + ".jpg")
plt.show()


psnr = p.psnr(np.array(img).astype(int), np.array(temp).astype(int))
print('psnr',psnr)

snr = p.snr(np.array(img).astype(int), np.array(temp).astype(int))
print('snr',snr)


















