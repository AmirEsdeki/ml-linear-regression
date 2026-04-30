import numpy as np


def linear_predict(x: np.ndarray, w: np.ndarray, b:float) -> float | np.ndarray:
    """Compute the linear model prediction

    Args:
        x (np.ndarray): Input feature matrix of shape (m,n).
        w (np.ndarray): Weight vector of shape (n,).
        b (float): Bias term.

    Returns:
        float: Predicted value(s).
    """
    m,n = x.shape
    y = np.zeros(m)
    for i in range(m):
        f_w_b = b
        for j in range(n):
            f_w_b += x[i][j] * w[j]
        y[i] = f_w_b
    return y

def linear_predict_vectorized(x: np.ndarray, w: np.ndarray, b:float) -> float | np.ndarray:
    """Compute the linear model prediction in a vectorized way up to more than 100x faster on large datasets.

    Args:
        x (np.ndarray): Input feature matrix of shape (m,n).
        w (np.ndarray): Weight vector of shape (n,).
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
        predict = linear_predict(x[i:i+1], w, b)[0] 
        cost += (predict - y[i]) ** 2
    return 1/(2 * m) * cost
        
def compute_cost_vectorized(x: np.ndarray, w: np.ndarray, b:float, y:np.ndarray) -> float:
    """Compute the mean squared error cost function in a vectorized way up to more than 100x faster on large datasets.

    Args:
        x (np.ndarray): Input feature matrix of shape (m, n).
        w (np.ndarray): Weight vector of shape (n,).
        b (float): Bias term.
        y (np.ndarray): Target values of shape (m,).

    Returns:
        float: MSE cost averaged over all m training examples.
    """
    m = x.shape[0]
    errors = linear_predict_vectorized(x, w, b) - y
    cost = 1/(2 * m) * np.dot(errors, errors)
    return cost

def compute_gradient(x: np.ndarray, w: np.ndarray, b:float, y:np.ndarray) -> tuple[np.ndarray, float]:
    """Compute the gradients of the MSE cost with respect to w and b using a loop.

    Args:
        x (np.ndarray): Input feature matrix of shape (m, n).
        w (np.ndarray): Weight vector of shape (n,).
        b (float): Bias term.
        y (np.ndarray): Target values of shape (m,).

    Returns:
        tuple[np.ndarray, float]: Gradient of w (shape (n,)) and gradient of b (scalar).
    """
    m,n= x.shape
    errors = linear_predict(x, w, b) - y # ndarray of shape (m,) having error per each data record
    d_d_w = np.zeros(n)
    d_d_b = 0
    for i in range(n):
        d_d_w[i] = 0
        for j in range(m):
            d_d_w[i] += x[j][i] * errors[j]
            if i == 0:
                d_d_b += errors[j]
        d_d_w[i] = (1/m) * d_d_w[i]
    d_d_b = (1/m) * d_d_b
    return (d_d_w, d_d_b)
    
    
def compute_gradient_vectorized(x: np.ndarray, w: np.ndarray, b:float, y:np.ndarray) -> tuple[np.ndarray, float]:
    """Compute the gradients of the MSE cost with respect to w and b in a vectorized way.

    Args:
        x (np.ndarray): Input feature matrix of shape (m, n).
        w (np.ndarray): Weight vector of shape (n,).
        b (float): Bias term.
        y (np.ndarray): Target values of shape (m,).

    Returns:
        tuple[np.ndarray, float]: Gradient of w (shape (n,)) and gradient of b (scalar).
    """
    m= x.shape[0]
    errors = linear_predict_vectorized(x, w, b) - y # ndarray of shape (m,) having error per each data record
    gradient_w = (1/m) * np.dot(np.transpose(x), errors)
    gradient_b = (1/m) * np.sum(errors)
    return (gradient_w, gradient_b)