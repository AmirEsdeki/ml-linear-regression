import numpy as np
from utils import ProgressBar
import linear_regression_functions as linear_rf
import logistic_regression_functions as logistic_rf
from typing import Callable

class Regression:

    def __init__(self, predict_function: Callable[[np.ndarray,np.ndarray,float],float | np.ndarray],
                 cost_function: Callable[[float, np.ndarray,np.ndarray,float, np.ndarray],float | np.ndarray],
                 learning_rate: float,
                 iterations: int,
                 regularization_lambda: float | None = 0) -> None:
        """Initialize the Regression model.

        Args:
            predict_function (Callable): Function with signature (x, w, b) -> predictions.
            cost_function (Callable): Function with signature (regularization_lambda, x, w, b, y) -> cost.
            learning_rate (float): Step size for gradient descent updates.
            iterations (int): Number of gradient descent steps to run.
            regularization_lambda (float | None): L2 regularization strength. Pass 0 or None to disable.
        """
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.w = None
        self.b = 0
        self.regularization_lambda = 0 if regularization_lambda is None else regularization_lambda
        self.predict_function = predict_function
        self.cost_function = cost_function

    @classmethod
    def linear(cls, learning_rate: float = 0.01, iterations: int = 1000, regularization_lambda: float | None = 0):
        """Create a Regression instance configured for linear regression.

        Args:
            learning_rate (float): Step size for gradient descent updates.
            iterations (int): Number of gradient descent steps to run.
            regularization_lambda (float | None): L2 regularization strength. Pass 0 or None to disable.

        Returns:
            Regression: Instance using MSE cost and linear predict functions.
        """
        return cls(predict_function=linear_rf.linear_regression_predict_vectorized, cost_function=linear_rf.linear_regression_cost_function_vectorized,
                   learning_rate=learning_rate, iterations=iterations, regularization_lambda=regularization_lambda)

    @classmethod
    def logistic(cls, learning_rate: float = 0.01, iterations: int = 1000, regularization_lambda: float | None = 0):
        """Create a Regression instance configured for logistic regression.

        Args:
            learning_rate (float): Step size for gradient descent updates.
            iterations (int): Number of gradient descent steps to run.
            regularization_lambda (float | None): L2 regularization strength. Pass 0 or None to disable.

        Returns:
            Regression: Instance using binary cross-entropy cost and sigmoid predict functions.
        """
        return cls(predict_function=logistic_rf.logistic_regression_predict_vectorized,
                   cost_function=logistic_rf.logistic_regression_cost_function_vectorized,
                   learning_rate=learning_rate, iterations=iterations, regularization_lambda=regularization_lambda)

    def predict(self, x: np.ndarray) -> np.ndarray | float:
        """Run prediction on input features using the trained model.

        Args:
            x (np.ndarray): Input feature matrix of shape (m, n).

        Returns:
            np.ndarray | float: Predicted values. For logistic regression these are probabilities in (0, 1).

        Raises:
            RuntimeError: If called before the model has been trained.
        """
        if self.w is None:
            raise RuntimeError("Model is not trained yet. Run fit() first.")
        return self.predict_function(x, self.w, self.b)

    #test non-vectorized to compare the performance
    def __compute_gradient_non_vectorized(self, x: np.ndarray, w: np.ndarray, b:float, y:np.ndarray) -> tuple[np.ndarray, float]:
        """Compute the gradients of the cost with respect to w and b using a loop.

        Args:
            x (np.ndarray): Input feature matrix of shape (m, n).
            w (np.ndarray): Weight vector of shape (n,).
            b (float): Bias term.
            y (np.ndarray): Target values of shape (m,).

        Returns:
            tuple[np.ndarray, float]: Gradient of w (shape (n,)) and gradient of b (scalar).
        """
        m,n= x.shape
        errors = self.predict_function(x, w, b) - y
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
        """Compute the gradients of the cost with respect to w and b in a vectorized way.

        Args:
            x (np.ndarray): Input feature matrix of shape (m, n).
            w (np.ndarray): Weight vector of shape (n,).
            b (float): Bias term.
            y (np.ndarray): Target values of shape (m,).

        Returns:
            tuple[np.ndarray, float]: Gradient of w (shape (n,)) and gradient of b (scalar).
        """
        m = x.shape[0]
        errors = self.predict_function(x, w, b) - y
        gradient_w = (1/m) * np.dot(np.transpose(x), errors) + (self.regularization_lambda/m) * w
        gradient_b = (1/m) * np.sum(errors)
        return gradient_w, gradient_b

     #test non-vectorized to compare the performance
    def __gradient_descent_non_vectorized(self, x: np.ndarray, w: np.ndarray, b:float, y:np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        """Run gradient descent using a loop-based gradient.

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
            gradient_w, gradient_b = self.__compute_gradient_non_vectorized(x, new_w, new_b, y)
            new_w = new_w - self.learning_rate * gradient_w
            new_b = new_b - self.learning_rate * gradient_b
            cost_history[i] = self.cost_function(self.regularization_lambda, x, new_w, new_b, y)
            i += 1
            progress.update(i)
        progress.finish()
        self.w = new_w
        self.b = new_b
        return new_w, new_b, cost_history

    def gradient_descent(self, x: np.ndarray, w: np.ndarray, b:float, y:np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        """Run gradient descent using a vectorized gradient.

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
            cost_history[i] = self.cost_function(self.regularization_lambda, x, new_w, new_b, y)
            i += 1
            progress.update(i)
        progress.finish()
        self.w = new_w
        self.b = new_b
        return new_w, new_b, cost_history
