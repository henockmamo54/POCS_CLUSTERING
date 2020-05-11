# -*- coding: utf-8 -*-
"""
Created on Mon May 11 20:48:54 2020

@author: Henock
"""

import numpy as np
import random
import math
from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
import time
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.utils import shuffle
np.random.seed(5)
from sklearn.utils.extmath import row_norms
from sklearn.utils.extmath import row_norms, squared_norm, stable_cumsum
import scipy.sparse as sp
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn import metrics as m


def generateRandomCenters(c,x):
    centers=[]
    _min= (np.min(x))
    _max= (np.max(x))
    t= np.mean(x,axis=0)    
    for i in range(c):
        centers.append( t + 0.01*i)   
    return np.asarray(centers) 


# intialize random member ship
def initializeMembershipMatrix(n,cluster):
    membership_mat = list()
    for i in range(n.shape[0]):
        random_num_list = [ np.random.rand() for i in range(cluster)]
        summation = sum(random_num_list) 
        temp_list = [x/summation for x in random_num_list]
        membership_mat.append(temp_list) 
    return np.asarray(membership_mat) 


def updateMembershipValue3_2(v,val):
    distances = cdist(val, v,metric='euclidean')
    return distances



def CalculateWeightValues(cluster_center,val,q):
        
    membership=[]    
    distances = cdist(val, v,metric='euclidean')[:,q]  
    sumdistance=np.sum(distances)
    membership= distances/sumdistance 
    
    return membership



def CalculateWeightValuesByIndex(cluster_center,val,q,index):
        
    distances = cdist(val, v,metric='euclidean')[:,q]  
    sumdistance=np.sum(distances)
    membership= distances[index]/sumdistance
        
    return membership



def calcError(center,mem):
    dist=0
    for i in range(mem.shape[0]):
        dist+=np.linalg.norm(center - mem[i])
    return dist

def moveVtoTheCenter(v,u,x):
    for k in range( cluster):
        items=[]
        for i  in range(  (x.shape[0])):
            if(u[i,k]>=np.max(u[i])):
                items.append(x[i])
        v[k]=np.mean(items, axis=0)
    return v

def _k_init(X, n_clusters, x_squared_norms, random_state, n_local_trials=None):
    """Init n_clusters seeds according to k-means++
    Parameters
    ----------
    X : array or sparse matrix, shape (n_samples, n_features)
        The data to pick seeds for. To avoid memory copy, the input data
        should be double precision (dtype=np.float64).
    n_clusters : integer
        The number of seeds to choose
    x_squared_norms : array, shape (n_samples,)
        Squared Euclidean norm of each data point.
    random_state : int, RandomState instance
        The generator used to initialize the centers. Use an int to make the
        randomness deterministic.
        See :term:`Glossary <random_state>`.
    n_local_trials : integer, optional
        The number of seeding trials for each center (except the first),
        of which the one reducing inertia the most is greedily chosen.
        Set to None to make the number of trials depend logarithmically
        on the number of seeds (2+log(k)); this is the default.
    Notes
    -----
    Selects initial cluster centers for k-mean clustering in a smart way
    to speed up convergence. see: Arthur, D. and Vassilvitskii, S.
    "k-means++: the advantages of careful seeding". ACM-SIAM symposium
    on Discrete algorithms. 2007
    Version ported from http://www.stanford.edu/~darthur/kMeansppTest.zip,
    which is the implementation used in the aforementioned paper.
    """
    n_samples, n_features = X.shape

    centers = np.empty((n_clusters, n_features), dtype=X.dtype)

    assert x_squared_norms is not None, 'x_squared_norms None in _k_init'

    # Set the number of local seeding trials if none is given
    if n_local_trials is None:
        # This is what Arthur/Vassilvitskii tried, but did not report
        # specific results for other than mentioning in the conclusion
        # that it helped.
        n_local_trials = 2 + int(np.log(n_clusters))

    # Pick first center randomly
    center_id = random_state.randint(n_samples)
    if sp.issparse(X):
        centers[0] = X[center_id].toarray()
    else:
        centers[0] = X[center_id]

    # Initialize list of closest distances and calculate current potential
    closest_dist_sq = euclidean_distances(
        centers[0, np.newaxis], X, Y_norm_squared=x_squared_norms,
        squared=True)
    current_pot = closest_dist_sq.sum()

    # Pick the remaining n_clusters-1 points
    for c in range(1, n_clusters):
        # Choose center candidates by sampling with probability proportional
        # to the squared distance to the closest existing center
        rand_vals = random_state.random_sample(n_local_trials) * current_pot
        candidate_ids = np.searchsorted(stable_cumsum(closest_dist_sq),
                                        rand_vals)

        # Compute distances to center candidates
        distance_to_candidates = euclidean_distances(
            X[candidate_ids], X, Y_norm_squared=x_squared_norms, squared=True)

        # Decide which candidate is the best
        best_candidate = None
        best_pot = None
        best_dist_sq = None
        for trial in range(n_local_trials):
            # Compute potential when including center candidate
            new_dist_sq = np.minimum(closest_dist_sq,
                                     distance_to_candidates[trial])
            new_pot = new_dist_sq.sum()

            # Store result if it is the best local trial so far
            if (best_candidate is None) or (new_pot < best_pot):
                best_candidate = candidate_ids[trial]
                best_pot = new_pot
                best_dist_sq = new_dist_sq

        # Permanently add best center candidate found in local tries
        if sp.issparse(X):
            centers[c] = X[best_candidate].toarray()
        else:
            centers[c] = X[best_candidate]
        current_pot = best_pot
        closest_dist_sq = best_dist_sq

    return centers

cluster=3
data=pd.read_csv('../../../dataset/seeds_dataset.csv', header=None, sep=',')  #
datasetname='seeds_dataset'
x= np.asarray(data.iloc[:, :-1]) 
y= np.asarray(data.iloc[:, -1]) 
le=LabelEncoder()
y=le.fit_transform(y)
data.head() 

c = _k_init(x,cluster, row_norms(x, squared=True), np.random.RandomState())

for i in range(10):
    
    data=shuffle(data)
    x= np.asarray(data.iloc[:, :-1]) 
    y= np.asarray(data.iloc[:, -1]) 
    le=LabelEncoder()
    y=le.fit_transform(y)

    v=c 
    w=updateMembershipValue3_2(v,x)
    ny=np.argmin(w,axis=1)


    errorlist=[]
    start=time.time()
    for t in range(50):
        error=0
        for q in range(cluster): 

            val=x[ny == q,:] 

            if(val.shape[0]!=0):

                weight=CalculateWeightValues(v,val,q) 

                v[q]= v[q]+np.dot(weight,val-v[q])



        w=updateMembershipValue3_2(v,x)  
        ny=np.argmin(w,axis=1)


    total_time= time.time()- start
    
    mypred=pd.DataFrame()
    mypred['Actual']=y
    mypred['Predicted']=ny
    
    mypred.to_csv('PPCOS_'+str(i)+'_pred.txt',sep=',',index=False)
    print(i)
    
    
    
data1=pd.read_csv('PPCOS_0_pred.txt');
data2=pd.read_csv('PPCOS_0_pred.txt');
data3=pd.read_csv('PPCOS_0_pred.txt');
data4=pd.read_csv('PPCOS_0_pred.txt');
data5=pd.read_csv('PPCOS_0_pred.txt');

data6=pd.read_csv('PPCOS_5_pred.txt');
data7=pd.read_csv('PPCOS_6_pred.txt');
data8=pd.read_csv('PPCOS_7_pred.txt');
data9=pd.read_csv('PPCOS_8_pred.txt');
data10=pd.read_csv('PPCOS_9_pred.txt');


datalist=[data1,data2,data3,data4,data5,
          data6,data7,data8,data9,data10]

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















