import numpy as np
from typing import Callable

def logistic_regression_predict_vectorized(x: np.ndarray, w: np.ndarray, b: float) -> float | np.ndarray:
    f = np.dot(x, w) + b
    g = 1 / (1 + np.exp(-f))
    return g

def logistic_regression_cost_function_vectorized( regularization_lambda: float, x: np.ndarray, w: np.ndarray, b: float, y: np.ndarray) -> float:
    """Compute the binary cross-entropy cost function for logistic regression in a vectorized way.

    Args:
        regularization_lambda (float): L2 regularization strength. Pass 0 to disable.
        x (np.ndarray): Input feature matrix of shape (m, n).
        w (np.ndarray): Weight vector of shape (n,).
        b (float): Bias term.
        y (np.ndarray): Binary target values of shape (m,), each element 0 or 1.

    Returns:
        float: Binary cross-entropy cost averaged over all m training examples.
    """
    m = x.shape[0]
    f = logistic_regression_predict_vectorized(x, w, b)
    cost = -np.mean(y * np.log(f) + (1 - y) * np.log(1 - f))
    regularization = (regularization_lambda / (2 * m)) * np.dot(w, w)

    return cost + regularization