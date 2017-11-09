# PhyloConvNetPolimi

Follows the structure of the repository:

In [this file](./The_whole_pipeline.ipynb) you can find the full pipeline of our method, which consists in the following steps. This file exploits some functions coming from [this file](./data_proc_dm_generation.py). This last module has as dependency the code inside the [PhyloCNN layer](./phcnn) folder.

# Experiments:

Experiment 1:
    1. *Input data* TCGA training set (12 tumor types) 
    2. *Distance Matrix*: extracted from the performance measure of random subsets of genes / correlations with up to 50% preserved 3 nearest neighbors
    3. *Architecture*: like the public repository of PhyloCNN / 2-layer fully connected network - from the function "create_dense_model"
    4. *Result*: up to 0.78 F1-score for all the tries, where the worst variants were untrainable at all (<0.3 f1 score)

# Progressing:

1. Train the network for the TADs distance matrice for all the genes, which are assigned to any tad. 
2. Train same set of genes as in 1., but with performance, 1 iteration distance matrice.

# To do list:

1. Control the training batches
2. Do classes balancing
3. Change the decision trees to Random Forests

