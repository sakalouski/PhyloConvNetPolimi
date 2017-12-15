

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

from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import PconvNetPolimi as pcnp
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
    # drop = Dropout(0.25)(flatt)
    # drop = Dense(units=16, activation='relu')(drop)
    # drop = BatchNormalization()(drop)
    # drop = Dropout(0.25)(drop)
    output = Dense(units=Y_train.shape[1], activation="softmax", name='output')(flatt)

    model = Model(inputs=data, outputs=output)
    from keras import optimizers
    opt = optimizers.Nadam(lr=1e-4)
    # opt = optimizers.SGD(lr = 1e-4)
    model.compile(optimizer=opt, loss='categorical_crossentropy')

    return model


def dl_test(X_dl, Y, kern_size=5):
    Y = str_to_int(Y)
    X_train_dl, Y_train_dl, X_test_dl, Y_test_dl = gc.data_preprocessing.split(X_dl, Y)
    X_train_dl, Y_train_dl, X_cv_dl, Y_cv_dl = gc.data_preprocessing.split(X_train_dl, Y_train_dl)

    Y_input_test = keras.utils.to_categorical(Y_test_dl, num_classes=len(np.unique(Y_train_dl)))
    Y_input_train = keras.utils.to_categorical(Y_train_dl, num_classes=len(np.unique(Y_train_dl)))
    Y_input_cv = keras.utils.to_categorical(Y_cv_dl, num_classes=len(np.unique(Y_train_dl)))

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_dl)
    X_test = scaler.transform(X_test_dl)
    X_cv = scaler.transform(X_cv_dl)

    X_input_test = X_test_dl.reshape(X_test_dl.shape[0], X_test_dl.shape[1], -1)
    X_input_train = X_train_dl.reshape(X_train_dl.shape[0], X_train_dl.shape[1], -1)
    X_input_cv = X_cv_dl.reshape(X_cv_dl.shape[0], X_cv_dl.shape[1], -1)

    X_input_train, Y_input_train = pcnp.upsampling(X_input_train, Y_input_train)

    batch_size = 50
    model_name = 'Test'
    model = create_conv_model(X_input_train, Y_input_train, 5, kern_size)

    from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=8, min_lr=1e-7, verbose=0)
    checkpointer = ModelCheckpoint(filepath=model_name + '.h5', verbose=0, save_best_only=True)
    EarlStop = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='auto')

    history = model.fit(X_input_train, Y_input_train,
                        epochs=3000,
                        batch_size=batch_size,
                        verbose=1,
                        validation_split=0.3,
                        validation_data=(X_input_cv, Y_input_cv),
                        callbacks=[reduce_lr, EarlStop],
                        shuffle=True)

    Y_pred = np.empty_like(Y_input_test)
    for i in range(0, Y_pred.shape[0]):
        Y_pred[i, :] = model.predict(x=X_input_test[[i], :, :])

    Y_pred_1d = pcnp.to_1d_labels(Y_pred)
    score_dl = f1_score(Y_test_dl.astype(int), Y_pred_1d, average='weighted')
    return score_dl, history

path_staging_x = '/home/nanni/Data/TCGA/training_data/stage_BRCA_20K/X_hic.npy'
path_staging_y = '/home/nanni/Data/TCGA/training_data/stage_BRCA_20K/y_aggregated.npy'
X = np.load(path_staging_x)
Y = np.load(path_staging_y)

path_dm = '/home/nanni/Data/HiC/50kb/distance_matrix.npy'
d_m = np.load(path_dm)


selector = SelectKBest(score_func=chi2, k=6000)
X = selector.fit_transform(X,Y)
d_m = selector.transform(d_m)
d_m = selector.transform(d_m.T)

num_of_rand_to_compare = 10
num_of_genes = 300
kern_size = 5
winner_dm = 0
winner_random = 0
gene_interator = 0
report = []
# ### Taking 300 random genes
while(gene_interator < X.shape[1]-num_of_genes):
    #indexes = list(np.random.randint(low=0, high=X.shape[1], size = (num_of_genes)))
    indexes = np.arange(gene_interator,gene_interator+num_of_genes)
    X_300 = X[:,indexes]
    Y_temp = Y.astype(str)
    X_train, Y_train, X_test, Y_test = gc.data_preprocessing.split(X_300, Y_temp)


    order = []
    for i in range(X_300.shape[1]):
        sortet = np.argsort(d_m[i,:])
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
df_report.to_csv('report_staging_HiC.csv')