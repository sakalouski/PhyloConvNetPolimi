import numpy as np


def filter_distance_matrix(distance_matrix):
    """ Returns the indices of the rows (== columns) of the distance matrix
    to keep. These indices must be then applied to the training set as well.

    :param distance_matrix: symmetric distance matrix
    :return: set of indices

    Usage::

    >>> distance_matrix = ...
    >>> X = ...
    >>> survived_indices = filter_distance_matrix(distance_matrix)
    >>> new_distance_matrix = distance_matrix[survived_indices, :][:, survived_indices]
    >>> new_X = X[survived_indices]
    """

    # get the number of connections for each gene
    gene_connections = np.sum(~np.isinf(distance_matrix), axis=1)
    survived_idx = np.where(gene_connections > 0)[0]
    return survived_idx

