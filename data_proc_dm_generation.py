import pandas as pd
import numpy as np
import gmql as gl
import random

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


#-------------------------------------------------------------------------------------------------------------------------------------
# THE FIRST PART CONTAINS FUNCTIONS NECESSARY FOR DATA PREPROCESSING AND THE DISTANCE MATRICE GENERATION
#-------------------------------------------------------------------------------------------------------------------------------------


def get_raw_data(tcga_path, top_k, tads = True):
    '''
    the function addresses the data at "tcga_path" and processes it creating features and labels. Concretely: it transforms the str labels into integers and splits the data into training and test sets. 
    
    input: 
        tcga_path - string, contains the path to the dataset
        top_k - integer, number of features to be chosen with sklearn SelectKBest routine
        tads - boolean, indicates if we work only with genes included into tads or the whole set of genes 
    output:
        X_train,X_test,Y_train,Y_test
    '''
    tcga = pd.read_csv(tcga_path, sep="\t", header=[0,1,2], index_col=0)
    patients = tcga.T.reset_index()
    patients = patients.rename(columns={'level_0': 'tissue', 
                                        'level_1': 'type', 
                                        'level_2': 'patient_id'})
    patients.index.name = "patient"
    #gene_names = tcga.index.tolist()
    tcga_X,tcga_Y = GenerateXY(patients)
    
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


def GenerateXY(patients):
    '''
    the function takes all the str labels, which are cancer types and creates 1d Y array made of integer labels
    
    input: 
        patients - DataFrame, set of all the patients with all the genes and cancer type assigned to the cancerous ones
    output: 
        tcga_X,tcga_Y - numpy arrays. Whole available data divided into features and labels
    '''
    tcga_X = np.array(patients[patients.type == 'cancer'].values[:,3:],float)
    tcga_X = patients[patients.type == 'cancer'].values[:,3:]
    tcga_Y = np.zeros(len(tcga_X),int)
    
    list_of_tissues = ['zero']
    for i in range(0,len(patients[patients.type == 'cancer'])):
        e = pat_tissue[i]
        if e in list_of_tissues:
            tcga_Y[i] = list_of_tissues.index(e)
        else:
            list_of_tissues.append(e)
            tcga_Y[i] = list_of_tissues.index(e)

    return tcga_X,tcga_Y


def run_classifier(X,Y):
    '''
    the input is a subset of features. The classifier is created and the score on a cv dataset is obtained. the scores are used for building a distance matrice.
    
    input:
        X - numpy array 2d, features
        Y - numpy array 1d, labels
    output:
        score - float. performance of the whole cluster
        importance - numpy 1d array, importances of all the features in the cluster
    '''
    classy = DecisionTreeClassifier(min_samples_split = 10, min_samples_leaf = 5, random_state = 1)
    #classy = KNeighborsClassifier(n_neighbors=3, weights='uniform', n_jobs = 10, metric = 'chebyshev')
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=42)
    classy = classy.fit(X_train,Y_train)
    Y_pred = classy.predict(X_test)
    score = f1_score(Y_test, Y_pred, average = 'macro')
    importance = classy.feature_importances_
    return score, importance

def write_to_file(score, rank, index, subset_size, num_iters,k_top,tads = True):
    '''
    Writes all the generated feature subsets to files together with the scores step-by-step. designed for higher iteration numbers to prevent any result losses. The function is not used in the current implementation.
    
    input:
        score - float. Score of a current cluster
        rank - numpy array 1d. all the ranks of features of the cluster
        index - np array 1d. indices of the genes in the cluster. 
        subset_size - integer, size of the cluster
        num_iters - integer, number of attepts to get the optimal cluster for the considered set of features
        k_top - number of genes chosen for the whole process
        tads - boolean, identifies if we work with all the genes or only included into tads 
    '''
    if tads == True:
        snippet = '_from_tads_'
    else:
        snippet = '_full_'

    f_handle = open('scores_chosen_'+str(k_top)+snippet+str(subset_size)+'__'+str(num_iters)+'.dat', 'ab')
    np.savetxt(f_handle, np.array([score]), fmt = '%.5f', delimiter=',')
    f_handle.close()

    f_handle = open('indexes_chosen_'+str(k_top)+snippet+str(subset_size)+'__'+str(num_iters)+'.dat', 'ab')
    np.savetxt(f_handle, index, fmt = '%i', delimiter=',')
    f_handle.close()

    f_handle = open('ranks_chosen_'+str(k_top)+snippet+str(subset_size)+'__'+str(num_iters)+'.dat', 'ab')
    np.savetxt(f_handle, rank, fmt = '%.5f', delimiter=',')
    f_handle.close()

    
def index_generator(index_num, list_of_gene_indx):
    '''
    chooses a subset of features (from the list of indexes - list_of_gene_indx) of the index_num size 
    input:
        index_num - integer, length of the subset
        list_of_gene_indx - list, indies of available genes
    output:
        res - list, randomly chosen subset of genes of length index_num  
    '''
    random.shuffle(list_of_gene_indx)
    res = list(list_of_gene_indx[:index_num])
    return res


def init_files(num_iters,subset_size,k_top,tads = True):
    '''
    creates fresh files for the distance matrice generation process. The function is not used in the current implementation.
    input:
        used only for naming files
    '''
    if tads == True:
        snippet = '_from_tads_'
    else:
        snippet = '_full_'
    f_handle = open('scores_chosen_'+str(k_top)+snippet+str(subset_size)+'__'+str(num_iters)+'.dat', 'w')
    f_handle.close()
    f_handle = open('indexes_chosen_'+str(k_top)+snippet+str(subset_size)+'__'+str(num_iters)+'.dat', 'w')
    f_handle.close()
    f_handle = open('ranks_chosen_'+str(k_top)+snippet+str(subset_size)+'__'+str(num_iters)+'.dat', 'w')
    f_handle.close()

    
def cluster_generator_wrapper_subsets(tcga_X, tcga_Y, num_iters, subset_size, k_top, tads = True):
    '''
    the whole process of dividing the features set into clusters and assigning scores to them is implemented here. num_iters clusters are randomly generated from the whole set of features, the optimal one (with the highest score) is takem. The chosen features are subtracted from the whole set of features which is addressed on the next step for next num_iters iterations. 
    
    input: 
        tcga_X - np array 2d. features. only training on input!
        tcga_Y - np array 1d. labels. only training on input!
        subset_size - integer, size of the cluster
        num_iters - integer, number of attepts to get the optimal cluster for the considered set of features
        k_top - number of genes chosen for the whole process
        tads - boolean, identifies if we work with all the genes or only included into tads
    output:
        out_scores - list of floats. scores for all the clusters
        out_ranks - list of lists. ranks of features inside the clusters
        out_inds - list of lists. indices for each cluster      
    '''
    list_of_genes = list(range(0,tcga_X.shape[1]))
    
    out_scores = []
    out_ranks = []
    out_inds = []
    
    init_files(num_iters,subset_size,k_top,tads)    
    step = subset_size
    it = 0
    while(len(list_of_genes)>=step):
        it += 1
        #print(it,len(list_of_genes))
        scores = np.zeros(num_iters,float)
        importance = np.zeros((num_iters,subset_size),float)
        subset_inds = np.zeros((num_iters,subset_size),int)
        for i in range(0,num_iters):
            subset_inds[i,:] = index_generator(subset_size, list_of_genes)
            masked_X = tcga_X[:,subset_inds[i,:]]
            scores[i], importance[i,:] = run_classifier(masked_X,tcga_Y)
        sorted_scores_inds = np.argsort(subset_inds,axis = 1)
        sorted_scores = np.argsort(scores)
        
        out_scores.append(scores[sorted_scores[-1]])
        out_ranks.append(list(importance[sorted_scores[-1],:]))
        out_inds.append(list(subset_inds[sorted_scores[-1],:]))

        #write_to_file(scores[sorted_scores[-1]], importance[sorted_scores[-1],:], 
        #              subset_inds[sorted_scores[-1],:], subset_size, num_iters,k_top)   
        list_of_genes = list(set(list_of_genes)-set(list(subset_inds[sorted_scores[-1],:])))
    if (len(list_of_genes) == 0):
        return out_scores, out_ranks, out_inds
    masked_X = tcga_X[:,list_of_genes]
    score_rest, importance_rest = run_classifier(masked_X,tcga_Y)
    
    out_scores.append(score_rest)
    out_ranks.append(list(importance_rest))
    out_inds.append(list(list_of_genes))
    
    #write_to_file(score_rest, importance_rest, list_of_genes,subset_size, num_iters,k_top)
    return out_scores, out_ranks, out_inds



def subset_scores_load(num_iters, subset_size,k_top,tads = True):
    '''
    Loading all the scores and clusters from the saved step-by-step files. The function is not used in the current implementation.
    
    input:
        subset_size - integer, size of the cluster
        num_iters - integer, number of attepts to get the optimal cluster for the considered set of features
        k_top - number of genes chosen for the whole process
        tads - boolean, identifies if we work with all the genes or only included into tads
    output:
        scores - np array 1d. scores for each cluster
        ranks - np array 2d. ranks of all the features inside each cluster
        inds - np array 2d. indices of features for each cluster
    '''
    if tads == True:
        snippet = '_from_tads_'
    else:
        snippet = '_full_'
    scores = []
    inds = []
    i_t = []
    ranks = []
    r_t = []
    ite = 0
    
    fname = 'scores_chosen_'+str(k_top)+snippet+str(subset_size)+'__'+str(num_iters)+'.dat'
    reader = open(fname,'r')
    for row in reader:
        row = row.replace('\n','') 
        scores.append(float(row))
      
    
    fname = 'ranks_chosen_'+str(k_top)+snippet+str(subset_size)+'__'+str(num_iters)+'.dat'
    reader = open(fname,'r')
    for row in reader:
        row = row.replace('\n','') 
        if ite < subset_size:           
            r_t.append(float(row))
            ite+=1
        else:       
            ranks.append(r_t)
            r_t = []
            r_t.append(float(row))
            ite = 1
    if(len(scores)>len(inds)):
            ranks.append(r_t)
    
    ite = 0        
    
    fname = 'indexes_chosen_'+str(k_top)+snippet+str(subset_size)+'__'+str(num_iters)+'.dat'  
    reader = open(fname,'r')
    for row in reader:
        row = row.replace('\n','') 
        if ite < subset_size:
            i_t.append(int(row))
            ite+=1
        else:
            inds.append(i_t)
            i_t = []
            i_t.append(int(row))
            ite = 1                
    if(len(scores)>len(inds)):
        inds.append(i_t)
    return scores, ranks, inds

def get_distance_matrix(scores,indexes,ranks,num_feats):
    
    '''
    Distances are generated as differences of performances between the clusters. The inside-cluster distances are obtained from feature_importance parameter of Decision Tree. The inter-cluster distances are scaled to be at least 2 times bigger than any intra-cluster distance.  
    input:
        scores - np array 1d. scores for each cluster
        ranks - np array 2d. ranks of all the features inside each cluster
        inds - np array 2d. indices of features for each cluster
        num_feats - integer. number of features in cluster
    output:
        dist_final - 2d array. distance matrice constructed of intracluster and intercluster distances.
    '''
    
    dist_inner = np.zeros((num_feats,num_feats),float)
    dist_clust = np.zeros((num_feats,num_feats),float)
    dist_final = np.zeros((num_feats,num_feats),float)
    for i in range(0,len(scores)):
        for j in range(i+1,len(scores)):
            dist = abs(scores[i]-scores[j])
            for feat1 in range(0,len(indexes[i])):
                for feat2 in range(0,len(indexes[j])):
                    dist_clust[indexes[i][feat1],indexes[j][feat2]] = dist
                    dist_clust[indexes[j][feat2],indexes[i][feat1]] = dist
    for i in range(0,len(scores)):
            for feat1 in range(0,len(indexes[i])):
                for feat2 in range(feat1+1,len(indexes[i])):
                    dist = abs(ranks[i][feat1]-ranks[i][feat2])
                    if dist == 0.0 and i != j:
                        dist = 1e-7
                    dist_inner[indexes[i][feat1],indexes[i][feat2]] = dist
                    dist_inner[indexes[i][feat2],indexes[i][feat1]] = dist
    
    dist_inner[dist_inner == 0] = np.max(dist_inner)*2.0
    dist_inner[dist_inner == 1e-9] = 0
    dist_final = dist_inner + dist_clust
    return dist_final

def get_interclust_dists(MDSres):
    '''
    the fuctions generates the distance matrice from the MDSresult matrics.
    
    input:
        MDSres - 2d np array. result of MDS 
    output:
        dist_m - 2d np array. reconstructed distance matrice
    '''
    dist_m = np.zeros((MDSres.shape[1],MDSres.shape[1]),float)
    for i in range(0,MDSres.shape[1]):
        for j in range(i + 1,MDSres.shape[1]):
            distIJ = np.sqrt(np.sum((MDSres[:,i]-MDSres[:,j])**2))
            dist_m[i,j] = distIJ
            dist_m[j,i] = distIJ
    return dist_m

def get_k_nearest_inds(dist_m,i,k):
    '''
    the function returns the k nearest neighbors of the feature i from the distance matrice dist_m 
    input:
        dist_m - 2d np array. distance matrice
        i - integer. index of feature to get the neighbors for
        k - integer. number of neighbors to get
    output:
        all_dists[:k] - indices of k nearest neighbors
    '''
    all_dists = np.argsort(dist_m[i,:])
    return all_dists[:k]

def check_preservation_of_dims(MDSres, dist_input, subset_size):
    '''
    the fraction of subset_size preserved nearest neighbors is returned. Implemented for varification of MDS procedure accuracy.
    input:
        MDSres - 2d np array. result of MDS
        dist_input - 2d np array. distance matrice, which was an input to MDS
        subset_size - integer. number of nearest neighbors to varify. 
    output:
        fraction for how many features the k nearest neighbors were preserved
    '''
    k = subset_size
    preserved = 0.0
    distroyed = 0.0
    print("Computing distances in MDSres")
    dist_m = get_interclust_dists(MDSres)
    print("Finding nearest neighbors")
    for i in range(0, dist_input.shape[1]):
        k_input = get_k_nearest_inds(dist_input,i,k)
        k_trans = get_k_nearest_inds(dist_m,i,k)
        if set(k_trans) == set(k_input):
            preserved += 1.0
        else:
            distroyed += 1.0        
    #print(preserved/(preserved+distroyed))
    return preserved/(preserved+distroyed)






#-------------------------------------------------------------------------------------------------------------------------------------
# NEXT PART CONTAINS FUNCTIONS NECESSARY FOR THE SECOND PART OF THE PIPELINE - BUILING THE NETWORK AND EVALUATING ITS PROPERTIES
#-------------------------------------------------------------------------------------------------------------------------------------







def mds_reshape(MDSmat, batch_size):
    '''
    The function reshapes the MDS representation in a form suitable for the DL setup
    
    input:
        MDSmat - 2d np array, mds coordinates matrice
        batch_size - integer, size of a training batch
    output:
        MDSmat - 4d np array, MDS representation, suitable for the DL setup
    '''
    MDSmat = MDSmat.reshape(1,MDSmat.shape[0],MDSmat.shape[1],1).T
    tmp = np.zeros((batch_size,MDSmat.shape[1],MDSmat.shape[2],1),float)
    tmp[:,:,:,:] = MDSmat
    MDSmat = tmp 
    return MDSmat


def get_class_weights(Y):
    '''
    input:
        Y - 2d np array of labels, categorical
    output:
        class_weights - 1d np array. 
    '''
    summa = np.sum(Y)
    class_weights = {}
    for i in range(0,Y.shape[1]):
        class_weights[i] = summa/(np.sum(Y,axis = 0)[i])
    return class_weights

def to_1d_labels(Y):
    '''
    Conversion from categorical to 1d labels array
    
    input:
        Y - 2d np categorical array
    output:
        res - 1d np array of labels
    '''
    res = np.zeros(Y.shape[0],int)
    for i in range(0,Y.shape[0]):
        res[i] = np.argmax(Y[i,:])
    return res


def create_dense_model(X_train, Y_train, opt = None):
    '''
    Creating a model made of fully connected layers
    
    inputs:
        X_train - 3d array, last layer is for channels
        Y_train - 2d categorical array - to define the output layer size
        opt - either Adam (None) or SGD (rest) optimizer 
    output:
        model - compiled model
    '''
    nb_features = X_train.shape[1]

    data = Input(shape=(nb_features, 1))
    
    dense_layer = Flatten()(data)
    dense_layer = Dense(int(8), activation='selu')(dense_layer)
    dense_layer = BatchNormalization()(dense_layer)
    dense_layer = Dropout(0.5)(dense_layer)

    output = Dense(units=Y_train.shape[1], activation="softmax", name='output')(dense_layer)

    model = Model(inputs = data, outputs = output)
    from keras import optimizers
    if opt == None:
         opt = 'Adam'
    else:
        opt = optimizers.SGD()
    model.compile(optimizer=opt, loss='categorical_crossentropy')
    return model


def create_model(X_train, Y_train, MDSmat, nb_filters, nb_neighbors,opt = None):
    '''
    Creating a model containing PhyloConv layers
    
    inputs:
        X_train - 3d array, last layer is for channels
        Y_train - 2d categorical array - to define the output layer size
        opt - either Adam (None) or SGD (rest) optimizer 
        MDSmat - 2d array. MDS representation
        nb_filters - integer
        nb_neighbors - integer
    output:
        model - compiled model
    '''
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
    output = Dense(units=Y_train.shape[1], activation="softmax", name='output')(drop)

    model = Model(inputs=[data, coordinates], outputs=output)
    from keras import optimizers
    if opt == None:
         opt = 'Adam'
    else:
        opt = optimizers.SGD()
    model.compile(optimizer=opt, loss='categorical_crossentropy')
    
    return model

def train_model(model, X_train_inp, Y_input_train, batch_size, pat_lr, pat_max, model_name, MDSmat, dense = None):
    '''
    The model is trained batch by batch. Checking of the losses and decrease of the learning rate is done manually. The labels are not balanced inside of the training batches at all, rather the opposite is true. There were two ways of doing training - either train_on_batch or fit with one epoch. The latter one was chosen for the beginning as the implementation is simplier. 
    
    input:
        model - compiled model to be trained
        X_train_inp - 4d array, channels last
        Y_input_train - 2d array, categorical
        batch_size - integer
        pat_lr - integer, patience for the learning rate decrease
        pat_max - integer, early stopping patience
        model_name - string, saves the model under this name
        MDSmat - MDS coordinates matrice
        dense  - boolean. No input of the coordinates matrice for the fully connected network
    output:
        model - trained model with the lowest val loss achieved
    '''
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