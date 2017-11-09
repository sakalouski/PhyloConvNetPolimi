
# coding: utf-8

# # Generation of the distance matrice and its euclidean space transformation

# This is the first part of the pipeline, where the data is preprocessed, distance matrice is generated and transformed into a coordinate matrice with a help of an MDS procedure, implemented in R. 
# 
# The data preprocessing includes labels (Y) and features (X) generation in a form of 1d and 2d numpy arrays respectively.
# 
# The distance matrice is obtained from the performances of each of the feature subsets on the cross-validation data set. The intrasubset distances are obtained based on the feature_importance parameter of Decision Tree classifier, sklearn implementation.  

# # Importing all the necessary modules

# In[1]:


import pandas as pd
import gmql as gl
import random
import os
import sys
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.neighbors import DistanceMetric
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import numpy as np
# matplotlib properties
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

matplotlib.rcParams['figure.figsize'] = (20.0, 10.0)
matplotlib.rcParams['font.size'] = 22
matplotlib.rcParams['xtick.labelsize'] = 22
matplotlib.rcParams['ytick.labelsize'] = 22
matplotlib.rcParams['axes.labelsize'] = 30
matplotlib.rcParams['axes.titlesize'] = 30

from data_proc_dm_generation import *


# # rpy2 module allows on usage of R inside of python

# In[3]:


import rpy2
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
sma = importr('smacof')


# # Initialization of all the required parameters

# In[16]:


num_iters = 1 #number of iterations for finding the most optimal subset of features. 
#the subset is then subtracted from the list of available features and the operation is repeated 
subset_size = 5 #number of features to be chosen
top_k = None # feature selectrion procedure, where top k features based on chi2 test are taken
tads = True #set to True if you want to work with tads set of genes


# # Preprocessing data, generating features and labels for training and testing 

# In[17]:


if tads == True:
    path = "/home/nanni/Projects/ML/TAD_Trento/all_tcga_TAD.tsv"
else:
    path = "/home/nanni/Data/TCGA/all_tcga.tsv"

X_train,X_test,Y_train,Y_test = get_raw_data(path, top_k, tads)
scores, ranks, inds = cluster_generator_wrapper_subsets(X_train, Y_train, num_iters, subset_size, top_k)


# In[ ]:


dist_final = get_distance_matrix(scores, inds, ranks, X_train.shape[1])
R_mds = np.array(sma.torgerson(dist_final, p = dist_final.shape[0]-1))

np.save('R_mds_all_TADs_5feats_subset_1_iteration', R_mds)

print('the file was saved, computing the preservation...')
preservation = check_preservation_of_dims(R_mds.T, dist_final, subset_size)
#subset_size is number of nearest neighbors
print(preservation)
