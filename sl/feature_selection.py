import numpy as np

class PCA:
    """
    Perform principal components analysis on a 2 dimensional matrix of features.
    """
    def __init__(self, features: np.ndarray):

        self.features = features
        self.transform = None
        self.explained_var = None
        self.k = None
        self.mean = None
        self.std = None
        assert len(self.features.shape) == 2, "Data must be 2 dimensional."

    def fit_pca(self, k = None, target_var = 0.95):
       
        self.features = self._standarize_predictors(self.features)
        cov_matrix = self._compute_covariance_matrix(self.features)
        values, vectors = self._perform_eigendecomposition(cov_matrix)
        self.explained_var = self._explained_variance(values)

        if k is not None:
            assert 0 < k <= self.features.shape[1]

        else:
            cum_var = np.cumsum(self.explained_var)
            for j in range(len(cum_var)):
                if cum_var[j] >= target_var:
                    k = j + 1
                    break

        self.transform = vectors[:, :k]
        self.k = k
        return self.features @ vectors[:, :k]

    def transform_data(self, test_data):
        if self.transform is None:
            raise RuntimeError("You must run fit_pca first, before applying the transform on the test data")

        else:
            test_data = self._standarize_predictors(test_data)
            return test_data @ self.transform

    def get_explained_var(self):
        return np.cumsum(self.explained_var)[self.k-1]
            

    def _standarize_predictors(self, data):
        if self.mean is None:
            # Compute only once during pca fit
            self.mean = np.mean(data, axis=0)
            self.std = np.std(data, axis=0)
        res = (data - self.mean) / self.std
        return res

    def _compute_covariance_matrix(self, data):
        return (1 / (self.features.shape[0] - 1)) * data.T @ data

    def _perform_eigendecomposition(self, data):
        values, vectors = np.linalg.eigh(data)
        return values[::-1].clip(0), vectors[:,::-1]
    
    def _explained_variance(self, eigen_values):
        total_variance = np.sum(eigen_values)
        return eigen_values / total_variance
