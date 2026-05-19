import numpy as np
from utils import ProgressBar
import regression_predict_functions as rpf

class LinearRegression:
    
    def __init__(self, learning_rate: float, iterations: int, regularization_lambda: float | None = 0) -> None:
        """Initialize the LinearRegression model.

        Args:
            learning_rate (float): Step size for gradient descent updates.
            iterations (int): Number of gradient descent steps to run.
        """
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.w = None
        self.b = 0
        self.regularization_lambda = 0 if regularization_lambda is None else regularization_lambda

    @classmethod
    def default(cls):
        """Create a LinearRegression instance with default hyperparameters (learning_rate=0.01, iterations=1000).

        Returns:
            LinearRegression: A new instance with default settings.
        """
        return cls(learning_rate=0.01, iterations=1000, regularization_lambda=0)
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("Model is not trained yet. Run gradient_descent or gradient_descent_vectorized first.")
        return np.dot(x, self.w) + self.b

    def __compute_cost(self, x: np.ndarray, w: np.ndarray, b:float, y:np.ndarray) -> float:
        """Compute the mean squared error cost function.

        Args:
            x (np.ndarray): Input feature matrix of shape (m, n).
            w (np.ndarray): Weight vector of shape (n,).
            b (float): Bias term.
            y (np.ndarray): Target values of shape (m,).

        Returns:
            float: MSE cost averaged over all m training examples.
        """
        m,n = x.shape
        cost = 0.0
        regularization = 0
        for i in range(m):
            predict = rpf.linear_regression_predict(x[i:i + 1], w, b)[0]
            cost += (predict - y[i]) ** 2

        if self.regularization_lambda != 0:
            wj_sum = 0
            for i in range(n):
                wj_sum += w[i] ** 2
            regularization = (self.regularization_lambda/(2 * m)) * wj_sum

        return (1/(2 * m) * cost) + regularization
            
    def __compute_cost_vectorized(self, x: np.ndarray, w: np.ndarray, b:float, y:np.ndarray) -> float:
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
        errors = rpf.linear_regression_predict_vectorized(x, w, b) - y
        cost = 1/(2 * m) * np.dot(errors, errors)
        regularization = (self.regularization_lambda/(2 * m)) * np.dot(w,w)

        return cost + regularization

    def __compute_gradient(self, x: np.ndarray, w: np.ndarray, b:float, y:np.ndarray) -> tuple[np.ndarray, float]:
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
        errors = rpf.linear_regression_predict(x, w, b) - y # ndarray of shape (m) having error per each data record
        d_d_w = np.zeros(n)
        d_d_b = 0
        for i in range(n):
            d_d_w[i] = 0
            for j in range(m):
                d_d_w[i] += x[j][i] * errors[j]
                if i == 0:
                    d_d_b += errors[j]
            d_d_w[i] = (1/m) * d_d_w[i] + (self.regularization_lambda/m) * w[i]
        d_d_b = (1/m) * d_d_b
        return d_d_w, d_d_b
        
    def __compute_gradient_vectorized(self, x: np.ndarray, w: np.ndarray, b:float, y:np.ndarray) -> tuple[np.ndarray, float]:
        """Compute the gradients of the MSE cost with respect to w and b in a vectorized way.

        Args:
            x (np.ndarray): Input feature matrix of shape (m, n).
            w (np.ndarray): Weight vector of shape (n,).
            b (float): Bias term.
            y (np.ndarray): Target values of shape (m,).

        Returns:
            tuple[np.ndarray, float]: Gradient of w (shape (n,)) and gradient of b (scalar).
        """
        m = x.shape[0]
        errors = rpf.linear_regression_predict_vectorized(x, w, b) - y # ndarray of shape (m) having error per each data record
        gradient_w = (1/m) * np.dot(np.transpose(x), errors) + (self.regularization_lambda/m) * w
        gradient_b = (1/m) * np.sum(errors)
        return gradient_w, gradient_b

    def gradient_descent(self, x: np.ndarray, w: np.ndarray, b:float, y:np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        """Run gradient descent to minimize the MSE cost using a loop-based gradient.

        Args:
            x (np.ndarray): Input feature matrix of shape (m, n).
            w (np.ndarray): Initial weight vector of shape (n,).
            b (float): Initial bias term.
            y (np.ndarray): Target values of shape (m,).

        Returns:
            tuple[np.ndarray, float, np.ndarray]: Optimized w, optimized b, and cost history of shape (iterations,).
        """
        i = 0
        cost_history = np.zeros(self.iterations)
        new_w = w.copy()
        new_b = b
        progress = ProgressBar(self.iterations)
        while i < self.iterations:
            gradient_w, gradient_b = self.__compute_gradient(x, new_w, new_b, y)
            new_w = new_w - self.learning_rate * gradient_w
            new_b = new_b - self.learning_rate * gradient_b
            cost_history[i] = self.__compute_cost(x, new_w, new_b, y)
            i += 1
            progress.update(i)
        progress.finish()
        self.w = new_w
        self.b = new_b
        return new_w, new_b, cost_history

    def gradient_descent_vectorized(self, x: np.ndarray, w: np.ndarray, b:float, y:np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        """Run gradient descent to minimize the MSE cost using a vectorized gradient.

        Args:
            x (np.ndarray): Input feature matrix of shape (m, n).
            w (np.ndarray): Initial weight vector of shape (n,).
            b (float): Initial bias term.
            y (np.ndarray): Target values of shape (m,).

        Returns:
            tuple[np.ndarray, float, np.ndarray]: Optimized w, optimized b, and cost history of shape (iterations,).
        """
        i = 0
        cost_history = np.zeros(self.iterations)
        new_w = w.copy()
        new_b = b
        progress = ProgressBar(self.iterations)
        while i < self.iterations:
            gradient_w, gradient_b = self.__compute_gradient_vectorized(x, new_w, new_b, y)
            new_w = new_w - self.learning_rate * gradient_w
            new_b = new_b - self.learning_rate * gradient_b
            cost_history[i] = self.__compute_cost_vectorized(x, new_w, new_b, y)
            i += 1
            progress.update(i)
        progress.finish()
        self.w = new_w
        self.b = new_b
        return new_w, new_b, cost_history
