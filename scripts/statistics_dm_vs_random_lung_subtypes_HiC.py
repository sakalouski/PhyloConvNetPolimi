

import pandas as pd
import os
import sys
from sklearn.metrics import f1_score
from keras import backend as K
import tensorflow as tf
config = tf.ConfigProto(device_count = {'CPU': 1}, intra_op_parallelism_threads=7, inter_op_parallelism_threads=1)
session = tf.Session(config=config)
K.set_session(session)
import keras
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import PconvNetPolimi as pcnp
from PconvNetPolimi.distances import *
import numpy as np
import gene4cancer as gc
from sklearn.preprocessing import StandardScaler
from keras.models import Model
from keras.layers import (Lambda, Flatten, Dropout, Dense, Input, BatchNormalization)
from keras.backend import floatx


def str_to_int(Y):
    list_of_tissues = []
    for i in range(Y.shape[0]):
        e = Y[i]
        if e in list_of_tissues:
            Y[i] = list_of_tissues.index(e)
        else:
            list_of_tissues.append(e)
            Y[i] = list_of_tissues.index(e)
    Y = Y.astype(int)
    return Y

def create_conv_model(X_train, Y_train, nb_filters, nb_neighbors, opt=None):
    nb_features = X_train.shape[1]
    data = Input(shape=(nb_features, 1), name="data", dtype=floatx())
    conv_layer = keras.layers.Conv1D(nb_neighbors, nb_filters, activation='relu', strides=nb_neighbors)(data)
    # conv_layer = keras.layers.Conv1D(nb_neighbors, nb_filters, activation='relu', strides = nb_neighbors)(conv_layer)
    flatt = Flatten()(conv_layer)
    #drop = Dropout(0.25)(flatt)
    #drop = Dense(units=16, activation='relu')(drop)
    #drop = BatchNormalization()(drop)
    # drop = Dropout(0.25)(drop)
    output = Dense(units=Y_train.shape[1], activation="softmax", name='output')(flatt)

    model = Model(inputs=data, outputs=output)
    from keras import optimizers
    opt = optimizers.Nadam(lr=1e-4)
    # opt = optimizers.SGD(lr = 1e-4)
    model.compile(optimizer=opt, loss='categorical_crossentropy')

    return model

def dl_test(X_dl, Y, kern_size = 5):
    Y = str_to_int(Y)
    X_train_dl, Y_train_dl, X_test_dl, Y_test_dl = gc.data_preprocessing.split(X_dl, Y)

    Y_input_test = keras.utils.to_categorical(Y_test_dl, num_classes=len(np.unique(Y_test_dl)))
    Y_input_train = keras.utils.to_categorical(Y_train_dl, num_classes=len(np.unique(Y_train_dl)))

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_dl)
    X_test = scaler.transform(X_test_dl)

    X_input_test = X_test_dl.reshape(X_test_dl.shape[0], X_test_dl.shape[1], -1)
    X_input_train = X_train_dl.reshape(X_train_dl.shape[0], X_train_dl.shape[1], -1)

    X_input_train, Y_input_train = pcnp.upsampling(X_input_train, Y_input_train)

    batch_size = 100
    model_name = 'Test'
    model = create_conv_model(X_input_train, Y_input_train, 10, kern_size)

    from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=8, min_lr=1e-7, verbose=0)
    checkpointer = ModelCheckpoint(filepath=model_name + '.h5', verbose=0, save_best_only=True)
    EarlStop = EarlyStopping(monitor='val_loss', min_delta=0, patience=12, verbose=0, mode='auto')

    history = model.fit(X_input_train, Y_input_train,
                        epochs=3000,
                        batch_size=batch_size,
                        verbose=0,
                        validation_split=0.3,
                        callbacks=[reduce_lr, EarlStop],
                        shuffle=True)

    Y_pred = np.empty_like(Y_input_test)
    for i in range(0, Y_pred.shape[0]):
        Y_pred[i, :] = model.predict(x=X_input_test[[i], :, :])

    Y_pred_1d = pcnp.to_1d_labels(Y_pred)
    score_dl = f1_score(Y_test_dl.astype(int), Y_pred_1d, average='weighted')
    return score_dl, history




















path_dm = '/home/nanni/Data/HiC/50kb/distance_matrix.npy'
d_m = np.load(path_dm)
idx_to_gene_path = '/home/nanni/Data/HiC/50kb/idx_to_gene.csv'
df_idx_to_gene = pd.read_csv(idx_to_gene_path,index_col=0,header = None)
gene_to_idx_path = '/home/nanni/Data/HiC/50kb/gene_to_idx.csv'
df_gene_to_idx = pd.read_csv(gene_to_idx_path,index_col=0,header = None)

path_data = '/home/nanni/Data/TCGA/tcga.tsv'
df_data = pd.read_csv(path_data,sep='\t')
filtered_df = df_data[df_data['sample_type'].isin(['Primary Solid Tumor','Solid Tissue Normal'])]
gene_dm = df_gene_to_idx.index.tolist()

lol = []
for i in filtered_df.index.tolist():
    if (filtered_df.tumor_type.loc[i] == 'LUSC'
        or filtered_df.tumor_type.loc[i] == 'LUAD'):
        lol.append(i)
df_sub = filtered_df.loc[lol]
filtered_df = df_sub


X_normal, Y_normal, X_tumor, Y_tumor, gene_to_idx, idx_to_gene = gc.data_preprocessing.preprocess_data(filtered_df[filtered_df.columns[:7].tolist()
                                                                                                                   + gene_dm])
X, Y, state = gc.data_preprocessing.assemble_normal_tumor(X_normal, Y_normal, X_tumor, Y_tumor)

survived_indices = filter_distance_matrix(d_m)
new_dm = d_m[survived_indices, :][:, survived_indices]
new_X = X[:,survived_indices]








num_of_rand_to_compare = 10
num_of_genes = 300
kern_size = 5
winner_dm = 0
winner_random = 0
gene_interator = 0
report = []
# ### Taking 300 random genes
while(gene_interator < new_X.shape[1]-num_of_genes):
    #indexes = list(np.random.randint(low=0, high=X.shape[1], size = (num_of_genes)))
    indexes = np.arange(gene_interator,gene_interator+num_of_genes)
    X_300 = new_X[:,indexes]
    dm_temp = new_dm[indexes,:][:,indexes]
    Y_temp = Y.astype(str)
    X_train, Y_train, X_test, Y_test = gc.data_preprocessing.split(X_300, Y_temp)

    order = []
    for i in range(X_300.shape[1]):
        sortet = np.argsort(dm_temp[i,:])
        order.append(i)
        order.extend(sortet[1:kern_size])

    single_report = {}

    X_temp = X_300[:, order]
    score_dm, history = dl_test(X_temp, Y_temp)

    single_report['real_distance_mat'] = score_dm
    single_report['epochs_real_dm'] = len(history.epoch)
    single_report['set_of_genes_begin'] = gene_interator
    single_report['set_of_genes_end'] = gene_interator + num_of_genes

    for iterator in range(num_of_rand_to_compare):
        # shuffle(order)
        rand_order = np.random.randint(0, num_of_genes, size=kern_size * num_of_genes)
        X_temp = X_300[:, rand_order]
        score_rand, history = dl_test(X_temp, Y_temp)
        single_report['shuffled_dm_' + str(iterator)] = score_rand
        single_report['epochs_shuffled_dm_' + str(iterator)] = len(history.epoch)

    report.append(single_report)
    gene_interator += num_of_genes
    print('_______________________________________')
    print(single_report)

df_report = pd.DataFrame.from_dict(report)
df_report.to_csv('report_HiC_best_genes_lung.csv')