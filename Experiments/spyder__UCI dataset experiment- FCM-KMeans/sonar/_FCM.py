# -*- coding: utf-8 -*-
"""
Created on Mon May 11 20:59:33 2020

@author: Henock
"""

import random
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.datasets.samples_generator import make_blobs
from sklearn.utils.extmath import row_norms
from sklearn.utils.extmath import row_norms, squared_norm, stable_cumsum
import scipy.sparse as sp
from sklearn.metrics.pairwise import euclidean_distances
import pandas as pd
from fuzzy_cmeans import FCM 
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


def initializ_centroids(X,n_clusters): 
    random_idx = np.random.permutation(X.shape[0])
    centroids = X[random_idx[:n_clusters]]
    return centroids

    
for i in range(5):
    
    data=shuffle(data)
    x= np.asarray(data.iloc[:, :-1]) 
    y= np.asarray(data.iloc[:, -1]) 
    le=LabelEncoder()
    y=le.fit_transform(y)
    
    fuzzy_cmeans2 = FCM(n_clusters=cluster,max_iter=50, m=70, error=1e-2)
#     fuzzy_cmeans2.centers=c
    fuzzy_cmeans2.fit(x)
    centers = fuzzy_cmeans2.centers
    label=fuzzy_cmeans2.predict(x)

    label.shape

        
    mypred=pd.DataFrame()
    mypred['Actual']=y
    mypred['Predicted']=label
    
    mypred.to_csv('FCM_'+str(i)+'_pred.txt',sep=',',index=False)
    print(i)
    
    

    
data1=pd.read_csv('FCM_0_pred.txt');
data2=pd.read_csv('FCM_1_pred.txt');
data3=pd.read_csv('FCM_2_pred.txt');
data4=pd.read_csv('FCM_3_pred.txt');
data5=pd.read_csv('FCM_4_pred.txt');

# data6=pd.read_csv('FCM_5_pred.txt');
# data7=pd.read_csv('FCM_6_pred.txt');
# data8=pd.read_csv('FCM_7_pred.txt');
# data9=pd.read_csv('FCM_8_pred.txt');
# data10=pd.read_csv('FCM_9_pred.txt');


datalist=[data1,data2,data3,data4,data5]
          # data6,data7,data8,data9,data10]

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
    

print('Accuracy',np.average(Accuracy))
print('Precision',np.average(Precision))
print('Recall',np.average(Recall))
print('F1_score',np.average(F1_score))
