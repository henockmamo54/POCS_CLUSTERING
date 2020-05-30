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
# from sklearn.cluster import KMeans
import numpy as np
import kmeansAlgo as k

from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn import metrics as m
np.random.seed(0)

cluster=2
data=pd.read_csv('../../../dataset/sonar.all-data', header=None, sep=',')  #
datasetname='sonar'
x= np.asarray(data.iloc[:, :-1]) 
y= np.asarray(data.iloc[:, -1]) 
le=LabelEncoder()
y=le.fit_transform(y)
data.head() 

for i in range(10):
    
    data=shuffle(data)
    x= np.asarray(data.iloc[:, :-1]) 
    y= np.asarray(data.iloc[:, -1]) 
    le=LabelEncoder()
    y=le.fit_transform(y)
    
    
    kmeans = k.Kmeans(n_clusters = cluster,max_iter=50) #KMeans(n_clusters=cluster, random_state=5,max_iter=50)
    kmeans.fit(x) 
    
    ny=kmeans.labels   
        
    mypred=pd.DataFrame()
    mypred['Actual']=y
    mypred['Predicted']=ny
    
    mypred.to_csv('KMEANS_'+str(i)+'_pred.txt',sep=',',index=False)
    print(i)
    
    
    
    
data1=pd.read_csv('KMEANS_0_pred.txt');
data2=pd.read_csv('KMEANS_1_pred.txt');
data3=pd.read_csv('KMEANS_2_pred.txt');
data4=pd.read_csv('KMEANS_3_pred.txt');
data5=pd.read_csv('KMEANS_4_pred.txt');
data6=pd.read_csv('KMEANS_5_pred.txt');
data7=pd.read_csv('KMEANS_6_pred.txt');
data8=pd.read_csv('KMEANS_7_pred.txt');
data9=pd.read_csv('KMEANS_8_pred.txt');
data10=pd.read_csv('KMEANS_9_pred.txt');


datalist=[data1,data2,data3,data4,data5, data6,data7,data8,data9,data10]

noclasses=np.unique(data1.Actual).shape[0]
print('noclasses',noclasses)

from itertools import permutations
combcols=(list(permutations(np.arange(0,noclasses,1), noclasses)))
print('len(combcols)',len(combcols))

Accuracy=[]
Precision=[]
Recall=[]
F1_score=[]


for h in range(len(datalist)):
    
    accuracylist=[]
    
    for j in range(len(combcols)):
        currentlist=combcols[j]
        temp=datalist[h].copy()
        for i in range(temp.shape[0]):
            temp.iloc[i,1] = currentlist[temp.Predicted[i]]
        accuracylist.append(m.accuracy_score(temp.Actual,temp.Predicted))

    currentlist=combcols[np.argmax(accuracylist)]
    temp=datalist[h].copy()
    for i in range(temp.shape[0]):
        temp.iloc[i,1] = currentlist[temp.Predicted[i]]

    prf=m.precision_recall_fscore_support(temp.Actual,temp.Predicted,average='weighted')    

    Accuracy.append(accuracylist[np.argmax(accuracylist)])
    Precision.append(prf[0])
    Recall.append(prf[1])
    F1_score.append(prf[2])
    
    print(h)
    

print('Accuracy',np.average(Accuracy) , ' std= ',np.std(Accuracy))
print('Precision',np.average(Precision), ' std= ',np.std(Precision))
print('Recall',np.average(Recall), ' std= ',np.std(Recall))
print('F1_score',np.average(F1_score), ' std= ',np.std(F1_score))





