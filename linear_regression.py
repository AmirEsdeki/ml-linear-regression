import numpy as np
from utils import ProgressBar

class LinearRegression:
    
    def __init__(self, learning_rate: float, iterations: int):
        """Initialize the LinearRegression model.

        Args:
            learning_rate (float): Step size for gradient descent updates.
            iterations (int): Number of gradient descent steps to run.
        """
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.w = None
        self.b = 0

    @classmethod
    def default(cls):
        """Create a LinearRegression instance with default hyperparameters (learning_rate=0.01, iterations=1000).

        Returns:
            LinearRegression: A new instance with default settings.
        """
        return cls(learning_rate=0.01, iterations=1000)
    
    def __linear_predict(self, x: np.ndarray, w: np.ndarray, b:float) -> np.ndarray:
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

    def __linear_predict_vectorized(self, x: np.ndarray, w: np.ndarray, b:float) -> float | np.ndarray:
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
        m = x.shape[0]
        cost = 0.0
        for i in range(m):
            predict = self.__linear_predict(x[i:i+1], w, b)[0] 
            cost += (predict - y[i]) ** 2
        return 1/(2 * m) * cost
            
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
        errors = self.__linear_predict_vectorized(x, w, b) - y
        cost = 1/(2 * m) * np.dot(errors, errors)
        return cost

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
        errors = self.__linear_predict(x, w, b) - y # ndarray of shape (m,) having error per each data record
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
        m= x.shape[0]
        errors = self.__linear_predict_vectorized(x, w, b) - y # ndarray of shape (m,) having error per each data record
        gradient_w = (1/m) * np.dot(np.transpose(x), errors)
        gradient_b = (1/m) * np.sum(errors)
        return (gradient_w, gradient_b)

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
        return (new_w, new_b, cost_history)

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
        return (new_w, new_b, cost_history)
