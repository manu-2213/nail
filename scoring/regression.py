import numpy as np
from nail.sl.linear_regression import LinearRegression


def mse(y: np.ndarray, y_hat: np.ndarray):
    return np.mean((y - y_hat)**2)

def rss(y: np.ndarray, y_hat: np.ndarray):
    return np.sum((y - y_hat)**2)

def f_statistic(lr_0: LinearRegression, lr_1: LinearRegression):
    """
    At this moment in time, this only makes sense in the context
    of comparing two linear regression models. 
    """
    assert lr_0.X.shape[0] == lr_1.X.shape[0], (
        "Linear regression models should be trained on the same number of samples."
    )
    p_0, p_1 = lr_0.B.shape[0], lr_1.B.shape[0]
    assert p_0 < p_1, (
        f"lr_1 must have a bigger number of coefficients than lr0 "
    )
    rss_0, rss_1 = rss(lr_0.y, lr_0._y_hat), rss(lr_1.y, lr_1._y_hat)
    assert rss_0 > rss_1, (
        "Make sure that the columns of X trained on lr_0 are a subset of the "
        "columns used to train lr_1"
    )

    return ((rss_0 - rss_1) / (p_1 - p_0)) / (rss_1 / (lr_0.X.shape[0] - p_1))

def rmse(y: np.ndarray, y_hat: np.ndarray):
    return np.sqrt(mse(y, y_hat))