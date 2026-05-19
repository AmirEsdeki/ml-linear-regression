import numpy as np
import pytest
from regression import Regression


# Shared fixture: a perfectly linear dataset y = 2*x1 + 3*x2 + 1
# Both gradient descent methods should converge close to w=[2,3], b=1
@pytest.fixture
def linear_dataset():
    np.random.seed(42)
    m = 100
    x = np.random.randn(m, 2)
    y = 2 * x[:, 0] + 3 * x[:, 1] + 1
    w_init = np.zeros(2)
    b_init = 0.0
    return x, y, w_init, b_init


class TestInit:
    def test_stores_hyperparameters(self):
        model = Regression(learning_rate=0.05, iterations=500)
        assert model.learning_rate == 0.05
        assert model.iterations == 500

    def test_default_hyperparameters(self):
        model = Regression.default()
        assert model.learning_rate == 0.01
        assert model.iterations == 1000


class TestGradientDescent:
    def test_returns_correct_shapes(self, linear_dataset):
        x, y, w_init, b_init = linear_dataset
        model = Regression(learning_rate=0.1, iterations=200)
        w, b, cost_history = model.gradient_descent_non_vectorized(x, w_init, b_init, y)
        assert w.shape == (2,)
        assert isinstance(b, float)
        assert cost_history.shape == (200,)

    def test_cost_decreases(self, linear_dataset):
        x, y, w_init, b_init = linear_dataset
        model = Regression(learning_rate=0.1, iterations=200)
        _, _, cost_history = model.gradient_descent_non_vectorized(x, w_init, b_init, y)
        assert cost_history[0] > cost_history[-1]

    def test_converges_to_correct_weights(self, linear_dataset):
        x, y, w_init, b_init = linear_dataset
        model = Regression(learning_rate=0.1, iterations=1000)
        w, b, _ = model.gradient_descent_non_vectorized(x, w_init, b_init, y)
        assert np.allclose(w, [2.0, 3.0], atol=0.1)
        assert abs(b - 1.0) < 0.1


class TestGradientDescentVectorized:
    def test_returns_correct_shapes(self, linear_dataset):
        x, y, w_init, b_init = linear_dataset
        model = Regression(learning_rate=0.1, iterations=200)
        w, b, cost_history = model.gradient_descent(x, w_init, b_init, y)
        assert w.shape == (2,)
        assert isinstance(b, (float, np.floating))
        assert cost_history.shape == (200,)

    def test_cost_decreases(self, linear_dataset):
        x, y, w_init, b_init = linear_dataset
        model = Regression(learning_rate=0.1, iterations=200)
        _, _, cost_history = model.gradient_descent(x, w_init, b_init, y)
        assert cost_history[0] > cost_history[-1]

    def test_converges_to_correct_weights(self, linear_dataset):
        x, y, w_init, b_init = linear_dataset
        model = Regression(learning_rate=0.1, iterations=1000)
        w, b, _ = model.gradient_descent(x, w_init, b_init, y)
        assert np.allclose(w, [2.0, 3.0], atol=0.1)
        assert abs(b - 1.0) < 0.1


class TestLoopVsVectorizedConsistency:
    def test_same_result(self, linear_dataset):
        x, y, w_init, b_init = linear_dataset
        model = Regression(learning_rate=0.1, iterations=100)
        w_loop, b_loop, _ = model.gradient_descent_non_vectorized(x, w_init, b_init, y)
        w_vec, b_vec, _ = model.gradient_descent(x, w_init, b_init, y)
        assert np.allclose(w_loop, w_vec, atol=1e-6)
        assert abs(b_loop - b_vec) < 1e-6
