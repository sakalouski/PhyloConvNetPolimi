import scipy.sparse as ss
import pandas as pd
from PconvNetPolimi.distances import hic_to_distance
import os
import numpy as np


tcga_path = "/home/nanni/Data/TCGA/Xena/tcga.tsv"
gene_to_idx_path = "/home/pinoli/hic_4_dl/gene_to_idx_50kb.tsv"
idx_to_gene_path = "/home/pinoli/hic_4_dl/idx_to_gene_50kb.tsv"
matrix_path = "/home/pinoli/hic_4_dl/geneVSgene_50kb.npz"
gene_tss_coordinates_path = "/home/nanni/Data/genes/refseq_gene_TSS_coordinates_UCSC/sample"

output_path = "/home/nanni/Data/Hi-C/50kb/"

if not os.path.isdir(output_path):
    os.makedirs(output_path)

gene_to_idx = pd.read_csv(gene_to_idx_path, header=None, sep="\t", index_col=0, squeeze=True)
idx_to_gene = pd.read_csv(idx_to_gene_path, header=None, sep="\t", index_col=0, squeeze=True)
geneVsgene = ss.load_npz(matrix_path)
gene_tss_coordinates = pd.read_csv(gene_tss_coordinates_path,
                                   sep="\t", names=['chr', 'start', 'stop', 'strand', 'name'])
tcga = pd.read_csv(tcga_path, sep="\t", index_col=None)
tcga_genes = set(tcga.columns[7:])
hic_genes = set(idx_to_gene.values)
genes_intersection = sorted(tcga_genes.intersection(hic_genes))

print("# TCGA genes: {}".format(len(tcga_genes)))
print("# HiC genes: {}".format(len(hic_genes)))
print("# gene intersection: {}".format(len(genes_intersection)))

distance, new_gene_to_idx, new_idx_to_gene = hic_to_distance(geneVsgene, gene_to_idx,
                                                             genes_intersection, gene_tss_coordinates)

# creating the training set
ordered_genes = new_idx_to_gene.values
X = tcga.as_matrix(ordered_genes)

np.save(os.path.join(output_path, "distance_matrix"), distance)
np.save(os.path.join(output_path, "X"), X)
new_gene_to_idx.to_csv(os.path.join(output_path, "gene_to_idx.csv"))
new_idx_to_gene.to_csv(os.path.join(output_path, "idx_to_gene.csv"))

print("DONE")
