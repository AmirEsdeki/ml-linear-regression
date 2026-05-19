import numpy as np
from typing import Callable

#test non-vectorized to compare the performance
def linear_regression_predict_non_vectorized(x: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    """Compute the linear model prediction using a loop.

    Args:
        x (np.ndarray): Input feature matrix of shape (m, n).
        w (np.ndarray): Weight vector of shape (n,).
        b (float): Bias term.

    Returns:
        np.ndarray: Predicted values of shape (m,).
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
    """Compute the linear model prediction in a vectorized way.

    Args:
        x (np.ndarray): Input feature matrix of shape (m, n).
        w (np.ndarray): Weight vector of shape (n,).
        b (float): Bias term.

    Returns:
        np.ndarray: Predicted values of shape (m,).
    """
    y = np.dot(x, w) + b
    return y

# test non-vectorized to compare the performance
def linear_regression_cost_function_non_vectorized(regularization_lambda: float, x: np.ndarray, w: np.ndarray, b: float, y: np.ndarray) -> float:
    """Compute the mean squared error cost function using a loop.

    Args:
        regularization_lambda (float): L2 regularization strength. Pass 0 to disable.
        x (np.ndarray): Input feature matrix of shape (m, n).
        w (np.ndarray): Weight vector of shape (n,).
        b (float): Bias term.
        y (np.ndarray): Target values of shape (m,).

    Returns:
        float: MSE cost averaged over all m training examples.
    """
    m, n = x.shape
    cost = 0.0
    regularization = 0
    for i in range(m):
        predict = linear_regression_predict_non_vectorized(x[i:i + 1], w, b)[0]
        cost += (predict - y[i]) ** 2

    if regularization_lambda != 0:
        wj_sum = 0
        for i in range(n):
            wj_sum += w[i] ** 2
        regularization = (regularization_lambda / (2 * m)) * wj_sum

    return (1 / (2 * m) * cost) + regularization

def linear_regression_cost_function_vectorized(regularization_lambda: float, x: np.ndarray, w: np.ndarray, b: float, y: np.ndarray) -> float:
    """Compute the mean squared error cost function in a vectorized way.

    Args:
        regularization_lambda (float): L2 regularization strength. Pass 0 to disable.
        x (np.ndarray): Input feature matrix of shape (m, n).
        w (np.ndarray): Weight vector of shape (n,).
        b (float): Bias term.
        y (np.ndarray): Target values of shape (m,).

    Returns:
        float: MSE cost averaged over all m training examples.
    """
    m = x.shape[0]
    errors = linear_regression_predict_vectorized(x, w, b) - y
    cost = 1 / (2 * m) * np.dot(errors, errors)
    regularization = (regularization_lambda / (2 * m)) * np.dot(w, w)

    return cost + regularization
