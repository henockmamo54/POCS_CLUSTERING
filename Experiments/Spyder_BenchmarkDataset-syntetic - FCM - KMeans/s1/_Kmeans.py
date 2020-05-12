# -*- coding: utf-8 -*-
"""
Created on Mon May 11 20:44:11 2020

@author: Henock
"""

# iris data Kmeans test 

import numpy as np
import random
import math
from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
import time
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.utils import shuffle 
import numpy as np
import kmeansAlgo as k
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
np.random.seed(10)
import pandas as pd
import numpy as np
from sklearn import metrics as m
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist
from scipy.spatial import distance


cluster=15
data=pd.read_csv('../../../dataset/s1.txt', header=None, sep=',')  #
datasetname='s1'
scaler=MinMaxScaler()
data=pd.DataFrame(scaler.fit_transform(data))
x= np.asarray(data.iloc[:, :]) 
data.head()  

error=[]

for i in range(10):
    
    data=shuffle(data)
    x= np.asarray(data.iloc[:, :])    

    kmeans = k.Kmeans(n_clusters=cluster, random_state=5,max_iter=50)
    kmeans.fit(x)
    
    labels=kmeans.labels
    centers=kmeans.centroids
    dst=0
    
    for j in range(len(labels)):
        dst+=distance.euclidean(list(data.iloc[j,:]),centers[labels[j]]) 
    error.append(dst)
    
    print(i)
    

print(error)
print('mean ',np.mean(error))
print('std ',np.std(error))