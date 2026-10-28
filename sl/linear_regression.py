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

    def solve(self, method: str = "naive"):
        if method == "naive":
            return self._naive_solving()

    def _naive_solving(self,):
        self.B = np.linalg.inv(self.X.T @ self.X) @ self.X.T @ self.y
        return self.B

    def _by_successive_orthogonolisation(self, return_orthogonalised: bool = False):
        Z = self.X[:, 0].copy()
        Z = np.expand_dims(Z, axis=-1)
        R = np.identity(self.B.shape[0])

        for j in range(1, self.X.shape[1]):
            z = self.X[:, j].copy()
            x = self.X[:, j]
            
            for k in range(j):
                a = np.dot(Z[:, k], x) / np.dot(Z[:, k], Z[:, k])
                R[k, j] = a
                z -= a * Z[:, k]
            z = np.expand_dims(z, axis=-1)
            Z = np.concatenate([Z, z], axis=-1)

        numerator = (Z.T @ self.y) 
        denominator = np.diagonal(Z.T @ Z)[:, None]
        self.Z_betas = numerator / denominator
        self.Z = Z

        if return_orthogonalised:
            return self.Z_betas, self.Z

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


            
