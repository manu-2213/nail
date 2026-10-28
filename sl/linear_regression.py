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
        self.y: np.ndarray = y
        self.B = np.random.randn(self.X.shape[1], 1)

    def get_predictions(self, ):
        return self._y_hat_

    def get_beta_var_cov_matrix(self, ):
        return np.linalg.inv(self.X.T @ self.X) * self._get_variance()

    def _get_variance(self, ):
        sqd_err = np.sum((self.y - self._y_hat_)**2)
        return 1 / (self.X.shape[0] - self.B.shape[0]) * sqd_err

    @property
    def _y_hat_(self, ):
        return self.X @ self.B

    def solve(self, method: str = "naive"):
        if method == "naive":
            return self._naive_solving()

    def _naive_solving(self,):
        self.B = np.linalg.inv(self.X.T @ self.X) @ self.X.T @ self.y
        return self.B

    def _by_successive_orthogonolisation(self, ):
        pass

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


            
