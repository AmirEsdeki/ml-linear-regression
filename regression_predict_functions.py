import numpy as np

#test non-vectorized to compare the performance
def linear_regression_predict(x: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    """Compute the linear model prediction

    Args:
        x (np.ndarray): Input feature matrix of shape (m,n).
        w (np.ndarray): Weight vector of shape (n,).
        b (float): Bias term.

    Returns:
        float: Predicted value(s).
    """
    m, n = x.shape
    y = np.zeros(m)
    for i in range(m):
        f_w_b = b
        for j in range(n):
            f_w_b += x[i][j] * w[j]
        y[i] = f_w_b
    return y

def linear_regression_predict_vectorized(x: np.ndarray, w: np.ndarray, b: float) -> float | np.ndarray:
    """Compute the linear model prediction in a vectorized way up to more than 100x faster on large datasets.

    Args:
        x (np.ndarray): Input feature matrix of shape (m,n).
        w (np.ndarray): Weight vector of shape (n,).
        b (float): Bias term.

    Returns:
        float: Predicted value(s).
    """
    y = np.dot(x, w) + b
    return y

def logistic_regression_predict_vectorized(x: np.ndarray, w: np.ndarray, b: float) -> float | np.ndarray:
    f = np.dot(x, w) + b
    g = 1 / (1 + np.exp(-f))
    return g