import pandas as pd
import numpy as np


def hic_to_distance(hic_matrix, gene_to_idx_hic, selected_genes, gene_coordinates):
    filter_gene_to_idx = gene_to_idx_hic[selected_genes]
    filtered_geneVSgene = hic_matrix[filter_gene_to_idx.values, :][:, filter_gene_to_idx.values]
    filtered_gene_to_idx = pd.Series(index=filter_gene_to_idx.index, data=np.arange(filtered_geneVSgene.shape[0]))

    sparsity = filtered_geneVSgene.count_nonzero() / (filtered_geneVSgene.shape[0]* filtered_geneVSgene.shape[1])
    print("[HiC] Sparsity: {:.2f}%".format(sparsity*100))
    filtered_geneVSgene = filtered_geneVSgene.todense()

    gene_coordinates['chr_no'] = gene_coordinates.chr.map(lambda x: x[3:])
    gene_coordinates = gene_coordinates.sort_values(['chr_no', 'start', 'stop'])
    gene_to_idx_ordered = filtered_gene_to_idx[gene_coordinates.name.values].dropna().astype(int)
    ordered_idxs = gene_to_idx_ordered.values
    ordered_genes = gene_to_idx_ordered.index

    ordered_geneVSgene = filtered_geneVSgene[ordered_idxs, :][:, ordered_idxs]
    np.fill_diagonal(ordered_geneVSgene, val=0)
    distances = adjacency_to_distance(ordered_geneVSgene)

    new_gene_to_idx = pd.Series(index=ordered_genes, data=np.arange(distances.shape[0]))
    new_idx_to_gene = pd.Series(data=ordered_genes, index=np.arange(distances.shape[0]))

    return distances, new_gene_to_idx, new_idx_to_gene

def adjacency_to_distance(ordered_geneVSgene):
    ordered_log_geneVSgene = np.log10(ordered_geneVSgene + 1)
    distances = 1 / ordered_log_geneVSgene
    return distances

