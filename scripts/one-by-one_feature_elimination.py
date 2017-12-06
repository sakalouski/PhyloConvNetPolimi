
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split


from keras import backend as K
import tensorflow as tf
config = tf.ConfigProto(device_count = {'CPU': 1}, intra_op_parallelism_threads=7, inter_op_parallelism_threads=1)
session = tf.Session(config=config)
K.set_session(session)
import keras

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import numpy as np
import gene4cancer as gc




path_dm = '/home/nanni/Data/HiC/50kb/distance_matrix.npy'
d_m = np.load(path_dm)
idx_to_gene = '/home/nanni/Data/HiC/50kb/idx_to_gene.csv'
df_idx_to_gene = pd.read_csv(idx_to_gene,index_col=0,header = None)
gene_to_idx = '/home/nanni/Data/HiC/50kb/gene_to_idx.csv'
df_gene_to_idx = pd.read_csv(gene_to_idx,index_col=0,header = None)


gene_dm = df_gene_to_idx.index.tolist()



path_data = '/home/nanni/Data/TCGA/tcga.tsv'
df_data = pd.read_csv(path_data,sep='\t')


df_data['sample_type'].unique()
filtered_df = df_data[df_data['sample_type'].isin(['Primary Solid Tumor','Solid Tissue Normal'])]


X_normal, Y_normal, X_tumor, Y_tumor, gene_to_idx, idx_to_gene = gc.data_preprocessing.preprocess_data(filtered_df[filtered_df.columns[:7].tolist() + gene_dm])

X, Y, state = gc.data_preprocessing.assemble_normal_tumor(X_normal, Y_normal, X_tumor, Y_tumor)
X_train, Y_train, X_test, Y_test = gc.data_preprocessing.split(X,Y)

score_base = gc.gene_extraction.random_classifier(X_train[:,1].reshape(-1, 1), Y_train, X_test[:,1].reshape(-1, 1), Y_test)

inds_to_leave = []
for i in tqdm(range(X_train.shape[1])):
    classy = LogisticRegression(n_jobs = 10)
    classy.fit(X_train[:,i].reshape(-1, 1),Y_train)
    y_pred = classy.predict(X_test[:,i].reshape(-1, 1))
    score = f1_score(Y_test, y_pred, average='weighted')
    if score<=score_base:
        inds_to_leave.append(i)

chosen_inds = np.array(inds_to_leave)
np.save('../data/chosen_inds', chosen_inds)

