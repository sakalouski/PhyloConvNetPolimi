# PhyloConvNetPolimi

Follows the structure of the repository:

In [this file](./The_whole_pipeline.ipynb) you can find the full pipeline of our method, which consists in the following steps. This file exploits some functions coming from [this file](./data_proc_dm_generation.py). This last module has as dependency the code inside the [PhyloCNN layer](./phcnn) folder.

# Experiments

1. Experiment 1:
    1. *Input data* TCGA training set (12 tumor types) 
    2. *Distance Matrix*: extracted from the performance measure of random subsets of genes
    3. *Architecture*: like the public repository of PhyloCNN
    4. *Result*: up to 0.78 F1-score