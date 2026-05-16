import numpy as np
from itertools import combinations_with_replacement


class PolynomialFeatures:
    def __init__(self, degree: int | None):
        self._degree = degree if degree is not None else 1
        self.poly_x: np.ndarray | None = None

    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Expand x with all polynomial feature combinations up to self._degree.

        Args:
            x: Array of shape (n_samples, n_features).

        Returns:
            Expanded array of shape (n_samples, n_output_features).
        """
        n_samples, n_features = x.shape
        cols = [x]

        for d in range(2, self._degree + 1):
            for combo in combinations_with_replacement(range(n_features), d):
                col = np.prod(x[:, list(combo)], axis=1, keepdims=True)
                cols.append(col)

        return np.hstack(cols)