import numpy as np


def linear_predict(x: np.ndarray, w: np.ndarray, b:float) -> float | np.ndarray:
    """Compute the linear model prediction: dot(x, w) + b.

    Args:
        x (np.ndarray): Input feature vector or matrix.
        w (np.ndarray): Weight vector.
        b (float): Bias term.

    Returns:
        float: Predicted value(s).
    """
    y = np.dot(x,w) + b
    return y

def compute_cost(x: np.ndarray, w: np.ndarray, b:float, y:np.ndarray) -> float:
    """Compute the mean squared error cost function.

    Args:
        x (np.ndarray): Input feature matrix of shape (m, n).
        w (np.ndarray): Weight vector of shape (n,).
        b (float): Bias term.
        y (np.ndarray): Target values of shape (m,).

    Returns:
        float: MSE cost averaged over all m training examples.
    """
    m = x.shape[0]
    cost = 0.0
    for i in range(m):
        predict = linear_predict(x[i], w, b)
        cost += (predict - y[i]) ** 2
    return 1/(2 * m) * cost
        
def compute_cost_vectorized(x: np.ndarray, w: np.ndarray, b:float, y:np.ndarray) -> float:
    """Compute the mean squared error cost function in a vectorized way up to 100x faster on large datasets.

    Args:
        x (np.ndarray): Input feature matrix of shape (m, n).
        w (np.ndarray): Weight vector of shape (n,).
        b (float): Bias term.
        y (np.ndarray): Target values of shape (m,).

    Returns:
        float: MSE cost averaged over all m training examples.
    """
    m = x.shape[0]
    errors = linear_predict(x, w, b) - y
    cost = 1/(2 * m) * np.dot(errors, errors)
    return cost
    