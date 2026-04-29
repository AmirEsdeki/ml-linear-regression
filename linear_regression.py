import numpy as np


def f_w_b(x: np.ndarray, w: np.ndarray, b:float) -> float:
    """Compute the linear model prediction: dot(x, w) + b.

    Args:
        x (np.ndarray): Input feature vector.
        w (np.ndarray): Weight vector.
        b (float): Bias term.

    Returns:
        float: Predicted value(s).
    """
    y = np.dot(x,w) + b
    return y

def j_w_b(x: np.ndarray, w: np.ndarray, b:float, y:np.ndarray) -> float:
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
        f_w_b = f_w_b(x, w, b)
        cost += (f_w_b - y[i]) ** 2
    return 1/(2 * m) * cost
        