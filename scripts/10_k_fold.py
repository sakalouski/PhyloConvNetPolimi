import pandas as pd
import gmql as gl
import random
import os
import sys
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import StandardScaler
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
from keras import backend as K
import tensorflow as tf

config = tf.ConfigProto(device_count={'CPU': 1}, intra_op_parallelism_threads=15, inter_op_parallelism_threads=1)
session = tf.Session(config=config)
K.set_session(session)

import keras as k
import keras
from keras import applications
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, BatchNormalization
from keras.optimizers import RMSprop, Adam, SGD, Nadam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.models import Model
from keras.layers import Input

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from keras.layers import (Lambda, MaxPooling1D, Flatten,
                          Dropout, Dense, Input)
from keras.models import Model
from keras.backend import floatx
from phcnn.layers import PhyloConv1D, euclidean_distances
from keras.utils.np_utils import to_categorical

matplotlib.rcParams['figure.figsize'] = (20.0, 10.0)
matplotlib.rcParams['font.size'] = 22
matplotlib.rcParams['xtick.labelsize'] = 22
matplotlib.rcParams['ytick.labelsize'] = 22
matplotlib.rcParams['axes.labelsize'] = 30
matplotlib.rcParams['axes.titlesize'] = 30

from data_proc_dm_generation import *

seed = 7
np.random.seed(seed)

import rpy2
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri

rpy2.robjects.numpy2ri.activate()
sma = importr('smacof')

num_iters = 2  # number of iterations for finding the most optimal subset of features.
# the subset is then subtracted from the list of available features and the operation is repeated
population_subset_size = 100  # number of features to be chosen
num_of_top_to_choose = 1
top_k = 500  # feature selectrion procedure, where top k features based on chi2 test are taken
# tads = True #set to True if you want to work with tads set of genes



path_X = "/home/nanni/Projects/ML/TAD_Trento/pancan_X_filtered.npy"
path_Y = "/home/nanni/Projects/ML/TAD_Trento/pancan_Y_filtered.npy"
tcga_X = np.load(path_X)
tcga_Y = np.load(path_Y)

list_of_tissues = []
for i in range(0, tcga_Y.shape[0]):
    e = tcga_Y[i]
    if e in list_of_tissues:
        tcga_Y[i] = list_of_tissues.index(e)
    else:
        list_of_tissues.append(e)
        tcga_Y[i] = list_of_tissues.index(e)
tcga_Y = tcga_Y.astype(int)

gene_selector = SelectKBest(score_func=chi2, k=top_k)
X_train, X_test, Y_train, Y_test = train_test_split(tcga_X, tcga_Y, test_size=0.15, random_state=42)
X_temp, _, Y_temp, _ = train_test_split(X_train, Y_train, test_size=0.18, random_state=42)

X_train = gene_selector.fit_transform(X_train,Y_train)
X_test = gene_selector.transform(X_test)


def KFold_checker(X_train, Y_train, X_cv, Y_cv, X_test, Y_test, Random = False):

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_cv = scaler.transform(X_cv)
    X_test = scaler.transform(X_test)

    m_d_m = np.zeros((X_train.shape[1], 25, 10), float)
    for i in range(0, 10):
        m_d_m[:, :, i] = np.load('./dist_m_top_500/dist_m_100_iter_' + str(i + 2) + '.npy')

    # In[15]:


    batch_size = 100
    nb_filters = 5
    nb_neighbors = 5
    model_name = 'test_pub_data'


    MDSmat_multi = np.zeros((int(batch_size), 25, top_k, 10), float)
    for i in range(0, 10):
        MDSmat_multi[:, :, :, [i]] = mds_reshape(m_d_m[:, :, i], int(batch_size))
        if Random == True:
            temp = np.zeros_like(MDSmat_multi[:,:,:,[i]])
            temp = MDSmat_multi[:,:,:,[i]]
            temp.flatten()
            np.random.shuffle(temp)
            temp = temp.reshape(MDSmat_multi.shape[0],MDSmat_multi.shape[1],MDSmat_multi.shape[2],1)
            MDSmat_multi[:,:,:,[i]] = temp



    Y_input_test = keras.utils.to_categorical(Y_test, num_classes=None)
    Y_input_train = keras.utils.to_categorical(Y_train, num_classes=None)
    Y_input_cv = keras.utils.to_categorical(Y_cv, num_classes=None)

    X_test_inp = X_test.reshape(X_test.shape[0], X_test.shape[1], -1)
    X_train_inp = X_train.reshape(X_train.shape[0], X_train.shape[1], -1)
    X_cv_inp = X_cv.reshape(X_cv.shape[0], X_train.shape[1], -1)

    X_train_inp, Y_input_train = upsampling(X_train_inp, Y_input_train)



    model = create_model(X_train_inp, Y_input_train, MDSmat_multi, nb_filters, nb_neighbors)
    model_trained = train_model(model, X_train_inp, Y_input_train, batch_size, 15, 15, model_name,
                                MDSmat_multi, X_cv_inp, Y_input_cv, dense=False)



    model_trained = create_model(X_train_inp, Y_input_train, MDSmat_multi, nb_filters, nb_neighbors)
    model_trained.load_weights(model_name + '.h5')


    Y_pred = np.array(Y_input_test)
    for i in range(0, Y_pred.shape[0]):
        Y_pred[i, :] = model_trained.predict(x=[X_test_inp[[i], :, :], MDSmat_multi[1:2:, :, :, [1]]])

    Y_pred_1d = to_1d_labels(Y_pred)
    from sklearn.metrics import f1_score
    return f1_score(Y_test, Y_pred_1d, average='weighted')



from sklearn.model_selection import StratifiedKFold
splitter = StratifiedKFold(n_splits=5, random_state=42)
#lol = splitter.split(X_train, Y_train)

f1scores_dm = []
for train_index, cv_index in splitter.split(X_train, Y_train):
    X_train_inp, X_cv = X_train[train_index], X_train[cv_index]
    Y_train_inp, Y_cv = Y_train[train_index], Y_train[cv_index]
    f1scores_dm.append(KFold_checker(X_train_inp, Y_train_inp, X_cv, Y_cv, X_test, Y_test))
f1_dm = sum(f1scores_dm)/len(f1scores_dm)


f1scores_rand = []
for train_index, cv_index in splitter.split(X_train, Y_train):
    X_train_inp, X_cv = X_train[train_index], X_train[cv_index]
    Y_train_inp, Y_cv = Y_train[train_index], Y_train[cv_index]
    f1scores_rand.append(KFold_checker(X_train_inp, Y_train_inp, X_cv, Y_cv, X_test, Y_test, Random = True))
f1_rand = sum(f1scores_rand)/len(f1scores_rand)

print('F1 based on DMs:', f1_dm)
print('F1 based on random:', f1_rand)