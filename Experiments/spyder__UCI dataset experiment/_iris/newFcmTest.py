# -*- coding: utf-8 -*-
"""
Created on Mon May 11 22:33:42 2020

@author: Henock
"""

import numpy as np
import random
import math
import operator
from matplotlib import pyplot as plt
import time


# intialize random member ship
def initializeMembershipMatrix(n):
    membership_mat = list()
    for i in range(n.shape[0]):
        random_num_list = [random.random() for i in range(cluster)]
        summation = sum(random_num_list) 
        temp_list = [x/summation for x in random_num_list]
        membership_mat.append(temp_list) 
    return np.asarray(membership_mat) 


# intialize V
def intializeCenter():
    xc=np.mean(x[:,0]) 
    yc=np.mean(x[:,1])
    v=[]
    for i in range(cluster):
        if(i%2==0):
            v.append([xc + (i+1)*e,yc + (i+1)*e])
        else:
            v.append([xc - (i+1)*e,yc - (i+1)*e])    
    return v



def updateCenterV( v,u):
    for k in range(cluster):
        num=0
        deno= 0
        for i in range(x.shape[0]):
            p=pow(u[i,k],m)
            num = num + (p*x[i])
            deno = deno+ p
            
        v[k]=num/deno
    return v

def updateMembershipValue(membership_mat, cluster_centers,val):
    n=val.shape[0]
    k=cluster
    p = float(2/(m-1))
    for i in range(n):
        x = val[i]
        distances=[]
        
        for k in range(cluster):      
            distances.append(np.linalg.norm(val[i]-v[k]) ) 
#         print(distances)
        
        for j in range(cluster):
            den=0
            for z in range(cluster):
                den= den + math.pow(float(distances[j]/distances[z]), p) 
#                 print(den,z,'*********',cluster)
#             print(den,float(1/den))    
            membership_mat[i][j] = float(1/den)  
#         print(membership_mat[i])
#         print('==><<<<<<<<<<<<<==')
    return membership_mat

def calcError(center,mem):
    dist=0
    for i in range(mem.shape[0]):
        dist+=np.linalg.norm(center - mem[i])
    return dist


# ================================================================
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics as metrics

cluster=3
data=pd.read_csv('../../../dataset/iris.data.txt', header=None, sep=',')  #
datasetname='iris'
x= np.asarray(data.iloc[:, :-1]) 
y= np.asarray(data.iloc[:, -1]) 
le=LabelEncoder()
y=le.fit_transform(y)
data.head() 

m=2 # fuzzyness
e=0.1


u=initializeMembershipMatrix(x)
v=intializeCenter()   

for l in range(20):
    
    v=updateCenterV(v,u)
    u=updateMembershipValue(u,v,x) 
    
    v=np.asarray(v)
    ny=np.argmax(u,axis=1)



noclasses=np.unique(y).shape[0]
print('noclasses',noclasses)

from itertools import permutations
combcols=(list(permutations(np.arange(0,noclasses,1), noclasses)))
print('len(combcols)',len(combcols))

Accuracy=[]
Precision=[]
Recall=[]
F1_score=[]


    
accuracylist=[]

for j in range(len(combcols)):
    currentlist=combcols[j]
    temp=ny
    for i in range(temp.shape[0]):
        temp[i] = currentlist[temp[i]]
    accuracylist.append(metrics.accuracy_score(y,temp))

currentlist=combcols[np.argmax(accuracylist)]
temp=ny.copy()

for i in range(temp.shape[0]):
    temp[i] = currentlist[temp[i]]

prf=metrics.precision_recall_fscore_support(y,temp,average='weighted')    

Accuracy.append(accuracylist[np.argmax(accuracylist)])
Precision.append(prf[0])
Recall.append(prf[1])
F1_score.append(prf[2])























