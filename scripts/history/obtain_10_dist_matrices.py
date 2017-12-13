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
import scipy
import numpy as np
# matplotlib properties
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from tqdm import tqdm

matplotlib.rcParams['figure.figsize'] = (20.0, 10.0)
matplotlib.rcParams['font.size'] = 22
matplotlib.rcParams['xtick.labelsize'] = 22
matplotlib.rcParams['ytick.labelsize'] = 22
matplotlib.rcParams['axes.labelsize'] = 30
matplotlib.rcParams['axes.titlesize'] = 30


from data_proc_dm_generation import *


# ## Making the results reproducible

# In[3]:


seed = 7
np.random.seed(seed)


# # rpy2 module allows on usage of R inside of python

# In[4]:


import rpy2
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
sma = importr('smacof')


num_iters = 100 #number of iterations for finding the most optimal subset of features.
#the subset is then subtracted from the list of available features and the operation is repeated 
population_subset_size = 15 #number of features to be chosen
num_of_top_to_choose = 10
top_k = 500 # feature selectrion procedure, where top k features based on chi2 test are taken



path_X = "/home/nanni/Projects/ML/TAD_Trento/pancan_X_filtered.npy"
path_Y = "/home/nanni/Projects/ML/TAD_Trento/pancan_Y_filtered.npy"
tcga_X = np.load(path_X)
tcga_Y = np.load(path_Y)

gene_selector = SelectKBest(score_func=chi2, k=top_k)
X_train, X_test, Y_train, Y_test = train_test_split(tcga_X, tcga_Y, test_size=0.20, random_state=42)
X_train, X_cv, Y_train, Y_cv = train_test_split(X_train, Y_train, test_size=0.25, random_state=42)

gene_selector.fit(X_train,Y_train)
scores_chi2 = gene_selector.scores_
inds_to_take = np.argsort(scores_chi2)[10000:18000]

X_train = X_train[:,inds_to_take]
X_cv = X_cv[:,inds_to_take]
X_test = X_test[:,inds_to_take]

#X_cv = gene_selector.transform(X_cv)
#X_test = gene_selector.transform(X_test)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_cv = scaler.transform(X_cv)
X_test = scaler.transform(X_test)


for final_ssize in range(2,8):
    scores, ranks, inds = cluster_generator_new(X_train, Y_train, num_iters, population_subset_size, 
                                                top_k, final_ssize,num_of_top_to_choose)
    dist_final = get_distance_matrix(scores, inds, ranks, X_train.shape[1])
    R_mds = np.array(sma.torgerson(dist_final, p = 25))    
    np.save('dist_m_200_iter_'+str(final_ssize), R_mds)

