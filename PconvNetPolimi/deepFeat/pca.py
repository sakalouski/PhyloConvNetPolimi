from sklearn.decomposition import PCA
from sklearn.decomposition.base import _BasePCA
from sklearn.utils import check_array
import numpy as np
from tqdm import tqdm


class ClusterPCA(_BasePCA):

    def __init__(self, copy=True, whiten=False, svd_solver="auto",
                 tol=0.0, iterated_power="auto", random_state=None):

        self.copy = copy
        self.whiten = whiten
        self.svd_solver = svd_solver
        self.tol = tol
        self.iterated_power = iterated_power
        self.random_state = random_state
        self._fitted = None

    def fit(self, X, clusters=None):
        """ Fit the model with X dividing the features in len(clusters) clusters
        and applying PCA to each of them.

        :param X: array-like, shape (n_samples, n_features)
               Training data, where n_samples in the number of samples
               and n_features is the number of features.
        :param y: Ignored
        :param clusters: array-like, shape (n_features) containing the cluster number for
               each feature
        :return: self
        """
        res = []
        if clusters is None:
            clusters = np.repeat(1, X.shape[1])
        elif not (isinstance(clusters, np.ndarray) and clusters.shape[0] == X.shape[1]):
            raise TypeError("clusters must be a ndarray of same number of features of X")
        cluster_ids = np.unique(clusters)

        if not (isinstance(X, np.ndarray) and X.ndim == 2):
            raise TypeError("X must be a 2D ndarray")

        for c_id in tqdm(cluster_ids):
            # getting the columns of X corresponding to the cluster of features
            c_id_idxs = np.where(clusters == c_id)[0]
            X_c_id = X[:, c_id_idxs]
            pca = PCA(n_components=1, copy=self.copy, whiten=self.whiten, svd_solver=self.svd_solver,
                      tol=self.tol, iterated_power=self.iterated_power,
                      random_state=self.random_state)    # generating a new PCA object
            pca.fit(X_c_id)         # fitting the object
            res.append((c_id_idxs, pca))
            # appending to the set of transformation objects
        self._fitted = res

    def transform(self, X):
        if self._fitted is None:
            raise ValueError("You must fit before transforming")

        if not (isinstance(X, np.ndarray) and X.ndim == 2):
            raise TypeError("X must be a 2D ndarray")

        X_res = np.zeros(shape=(X.shape[0], len(self._fitted)))
        for i in tqdm(range(len(self._fitted))):
            c_id_idxs, pca = self._fitted[i]
            X_c_id = X[:, c_id_idxs]
            X_c_id_trans = pca.transform(X_c_id).flatten()
            X_res[:, i] = X_c_id_trans
        return X_res


if __name__ == '__main__':
    X = np.array([[1, 4.5, 2,    4,   5],
                  [5, 4,   1.23, 4.5, 8]], dtype=np.float32)
    clusters = np.array([1, 1, 2, 3, 3], dtype=np.float64)

    print("Original")
    print(X)

    clusterPCA = ClusterPCA()
    clusterPCA.fit(X, clusters=clusters)
    X_trans = clusterPCA.transform(X)
    print("Result")
    print(X_trans)
