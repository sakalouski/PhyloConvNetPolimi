from keras import backend as K
import tensorflow as tf
config = tf.ConfigProto(device_count = {'CPU': 1}, intra_op_parallelism_threads=7, inter_op_parallelism_threads=1)
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

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.neighbors import DistanceMetric
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

import matplotlib
matplotlib.rcParams['figure.figsize'] = (20.0, 10.0)
matplotlib.rcParams['font.size'] = 22
matplotlib.rcParams['xtick.labelsize'] = 22
matplotlib.rcParams['ytick.labelsize'] = 22
matplotlib.rcParams['axes.labelsize'] = 30
matplotlib.rcParams['axes.titlesize'] = 30
import matplotlib.pylab as plt

import pandas as pd
import numpy as np
import os
import sys

from keras.layers import (Lambda, MaxPooling1D, Flatten,
                          Dropout, Dense, Input)
from keras.models import Model
from keras.backend import floatx
from .layers import PhyloConv1D, euclidean_distances
from keras.utils.np_utils import to_categorical


def get_raw_data(tcga_path, top_k=None):

    tcga = pd.read_csv(tcga_path, sep="\t", header=[0,1,2], index_col=0)
    patients = tcga.T.reset_index()
    patients = patients.rename(columns={'level_0': 'tissue', 
                                        'level_1': 'type', 
                                        'level_2': 'patient_id'})
    patients.index.name = "patient"
    gene_names = tcga.index.tolist()
    pat_tissue = np.array(patients.values[:,0],str)
    pat_tissue = patients.values[:,0] 
    tcga_X,tcga_Y = GenerateXY(patients,pat_tissue)
    
    if top_k == None:
        X_train, X_test, Y_train, Y_test = train_test_split(tcga_X, tcga_Y, test_size=0.33, random_state=42) 
        return X_train,X_test,Y_train,Y_test
    else:
        gene_selector = SelectKBest(score_func=chi2, k=top_k)
        X_train, X_test, Y_train, Y_test = train_test_split(tcga_X, tcga_Y, test_size=0.33, random_state=42)
        X_train = gene_selector.fit_transform(X_train,Y_train)
        X_test = gene_selector.transform(X_test)
        
        scores_chi2 = gene_selector.scores_
        if (scores_chi2[0]>scores_chi2[-1]):
            top_inds_to_use = np.argsort(scores_chi2)[:kop_k] 
        else:
            top_inds_to_use = np.argsort(scores_chi2)[-kop_k:]
        np.save('chi2_chosen_genes', top_inds_to_use)

        return X_train,X_test,Y_train,Y_test


def GenerateXY(patients,pat_tissue):

    tcga_X = np.array(patients[patients.type == 'cancer'].values[:,3:],float)
    tcga_X = patients[patients.type == 'cancer'].values[:,3:]

    tcga_Y = np.zeros(len(tcga_X),int)
    list_of_tissues = ['BLCA','BRCA','COAD','HNSC','KICH','KIRC','KIRP','LIHC','LUAD','LUSC','PRAD','THCA','UCEC']
    for i in range(0,len(patients[patients.type == 'cancer'])):
        for j in range(0,len(list_of_tissues)):
            if pat_tissue[i] == list_of_tissues[j]:
                tcga_Y[i] = j

    return tcga_X,tcga_Y

def mds_load_reshape(path, batch_size):
    if path == '/home/sakalouski/TAD_Trento/distance matrices/MDSes/distance_mat_100iter_False_feats_100feats_subset_ALL_MDS.csv':
        MDSmat = pd.read_csv(path, index_col= 0).as_matrix().T
    else:
        MDSmat = pd.read_csv(path, header = None).as_matrix().T
    MDSmat = MDSmat.reshape(1,MDSmat.shape[0],MDSmat.shape[1],1)
    #print(MDSmat.shape)
    tmp = np.zeros((batch_size,MDSmat.shape[1],MDSmat.shape[2],1),float)
    tmp[:,:,:,:] = MDSmat
    MDSmat = tmp
    #print(MDSmat.shape)    
    return MDSmat

def get_class_weights(Y):
    summa = np.sum(Y)
    class_weights = {}
    for i in range(0,Y.shape[1]):
        class_weights[i] = summa/(np.sum(Y,axis = 0)[i])
    return class_weights

def to_1d_labels(Y):
    res = np.zeros(Y.shape[0],int)
    for i in range(0,Y.shape[0]):
        res[i] = np.argmax(Y[i,:])
    return res


def create_dense_model(X_train, MDSmat, nb_filters, nb_neighbors, opt = None):
    
    nb_features = X_train.shape[1]

    data = Input(shape=(nb_features, 1))
    dense_layer = Flatten()(data)
    dense_layer = Dense(int(8), activation='selu')(dense_layer)
    dense_layer = BatchNormalization()(dense_layer)
    dense_layer = Dropout(0.5)(dense_layer)

    output = Dense(units=12, activation="softmax", name='output')(dense_layer)

    model = Model(inputs = data, outputs = output)
    from keras import optimizers
    if opt == None:
         opt = 'Adam'
    else:
        opt = optimizers.SGD()
    model.compile(optimizer=opt, loss='categorical_crossentropy')
    return model


def create_model(X_train, MDSmat, nb_filters, nb_neighbors,opt = None):
    
    nb_features = X_train.shape[1]
    nb_coordinates = MDSmat.shape[1]

    data = Input(shape=(nb_features, 1), name="data", dtype=floatx())
    coordinates = Input(shape=(nb_coordinates, nb_features, 1),
                                name="coordinates", dtype=floatx())

    conv_layer = data
    # We remove the padding that we added to work around keras limitations
    conv_crd = Lambda(lambda c: c[0], output_shape=lambda s: (s[1:]))(coordinates)

    distances = euclidean_distances(conv_crd)
    conv_layer, conv_crd = PhyloConv1D(distances, nb_neighbors,
                                       nb_filters, activation='selu')([conv_layer, conv_crd])
    #max = MaxPooling1D(pool_size=2, padding="valid")(conv_layer)
    flatt = Flatten()(conv_layer)
    drop = Dropout(0.5)(flatt)
    drop = Dense(units=128, activation='selu')(drop)
    drop = BatchNormalization()(drop)
    drop = Dropout(0.5)(drop)
    output = Dense(units=12, activation="softmax", name='output')(drop)

    model = Model(inputs=[data, coordinates], outputs=output)
    from keras import optimizers
    if opt == None:
         opt = 'Adam'
    else:
        opt = optimizers.SGD()
    model.compile(optimizer=opt, loss='categorical_crossentropy')
    
    return model

def train_model(model, X_train_inp, Y_input_train, batch_size, pat_lr,pat_max, model_name, MDSmat, dense = None):
    weights = get_class_weights(Y_input_train)
    pat = 0
    pat_lr_it = 0
    loss_prev = -1 
    lr = 5e-4
    lr_decr_mult = 0.5
    loss = 0.0
    losses = []
    while (1 != 2):
        for i in range(0,int(X_train_inp.shape[0]/batch_size)):
            if dense == True:
                X = X_train_inp[i*batch_size:(i+1)*batch_size,:,[0]]
            else:
                X = [X_train_inp[i*batch_size:(i+1)*batch_size,:],MDSmat]
            Y = Y_input_train[i*batch_size:(i+1)*batch_size,:]
            history = model.fit(x=X, y=Y, 
                              epochs = 1, 
                              class_weight = weights,
                              batch_size = batch_size, 
                              verbose=0, 
                              validation_split = 0.3,
                              shuffle=True) 
            loss += float(history.history['val_loss'][0])
        loss /= float(X_train_inp.shape[0]/batch_size)
        losses.append(loss)
        print("Current loss is: ", loss)
        if loss_prev == -1:        
            loss_prev = loss
            model.save_weights(model_name) 
            print("Model Saved as: "+model_name)
            continue
        if loss >= loss_prev:
            pat += 1
            pat_lr_it += 1
            #model.load_weights(model_name)
            if pat_lr_it > pat_lr:
                lr = lr*lr_decr_mult
                #model.load_weights(model_name)
                K.set_value(model.optimizer.lr, lr)
                print("New LR is: ", float(K.get_value(model.optimizer.lr)))
                pat_lr_it = 0
            if pat > pat_max:
                #model.save(model_name) 
                #print("Model Saved as: "+model_name)
                break
        else:
            pat = 0
            pat_lr_it = 0
            loss_prev = loss
            model.save_weights(model_name) 
            print("Model Saved as: "+model_name)
    model.load_weights(model_name)
    np.save('losses'+model_name,np.asarray(losses))
    return model