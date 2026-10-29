import numpy as np

class LinearRegression:
    def __init__(self, X, y):

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have the same number of samples, "
                f"got {X.shape[0]} and {y.shape[0]}. Makse sure "
                f"X.shape[0] == y.shape[0]"
            )

        if np.isnan(X).sum() > 0 or np.isnan(y).sum() > 0:
            raise ValueError(
                f"X and y must not contain nan valuse, "
                f"got: {np.isnan(X).sum()} nan values for X and "
                f"{np.isnan(y).sum()} nan values for y"
            )
        ones = np.ones_like(y)
        self.X: np.ndarray = np.concatenate([ones, X], axis=-1)
        self.Z: np.ndarray | None = None # Orthogonalised X matrix
        self.Z_betas: np.ndarray | None = None
        self.y: np.ndarray = y
        self.B = np.random.randn(self.X.shape[1], 1)

    def get_predictions(self, ):
        return self._y_hat

    def get_beta_var_cov_matrix(self, ):
        return np.linalg.inv(self.X.T @ self.X) * self._get_variance()

    def _get_variance(self, ):
        sqd_err = np.sum((self.y - self._y_hat)**2)
        return 1 / (self.X.shape[0] - self.B.shape[0]) * sqd_err

    @property
    def _y_hat(self, ):
        return self.X @ self.B

    def solve(self, 
              method: str = "naive",
              return_orthogonalised: bool = False,
              efficiently: bool = False,):
        if method == "naive":
            return self._by_naive_solving()

        if method == "gram-schmidt":
            return self._by_successive_orthogonolisation(
                return_orthogonalised=return_orthogonalised,
                efficiently=efficiently
                )

    def _by_naive_solving(self,):
        self.B = np.linalg.inv(self.X.T @ self.X) @ self.X.T @ self.y
        return self.B

    def _by_successive_orthogonolisation(self, return_orthogonalised: bool, efficiently: bool):
        Z = np.empty_like(self.X)
        Z[:, 0] = self.X[:, 0].copy()
        R = np.identity(self.B.shape[0])
        z_norms = np.empty(Z.shape[-1])
        z_norms[0] = np.dot(Z[:, 0], Z[:, 0])

        for j in range(1, self.X.shape[1]):
            z = self.X[:, j].copy()
            x = self.X[:, j]
            
            for k in range(j):
                a = np.dot(Z[:, k], x) / z_norms[k] # Can compute this once
                R[k, j] = a
                z -= a * Z[:, k]

            Z[:, j] = z
            z_norms[j] = np.dot(Z[:, j], Z[:, j])
            
        numerator = (Z.T @ self.y) 
        denominator = np.diagonal(Z.T @ Z)[:, None]
        self.Z_betas = numerator / denominator
        self.Z = Z

        if return_orthogonalised:
            return self.Z_betas, self.Z

        else:

            if efficiently: # Just for testing purposes of the quick inverse
                beta_vector = np.ones_like(self.B)
                beta_vector[-1] = self.Z_betas[-1, 0]

                for j in range(beta_vector.shape[0]-2, 0, -1):
                    beta_j = self.Z_betas[j].copy()

                    for k in range(j+1, beta_vector.shape[0]):
                        beta_j -= beta_vector[k] * R[j, k]

                    beta_vector[j] = beta_j

                dot_product = np.dot(
                    beta_vector[1:].reshape(beta_vector.shape[0] - 1), np.mean(self.X, axis=0)[1:]
                    )

                beta_vector[0] = np.mean(self.y, axis=0) - dot_product

                self.B = beta_vector
                return self.B

            else:
                self.B = np.linalg.inv(R) @ self.Z_betas
                return self.B

    def _by_qr_decomposition(self, ):
        pass

    def _by_svd(self, ):
        pass

    def _by_cholesky_decomposition(self, ):
        pass

    def _ridge(self, ):
        pass

    def _lasso(self, ):
        pass


            
