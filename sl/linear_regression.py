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

    def solve(self, method: str = "naive"):
        if method == "naive":
            return self._naive_solving()

    def _naive_solving(self,):
        self.B = np.linalg.inv(self.X.T @ self.X) @ self.X.T @ self.y
        return self.B
            
