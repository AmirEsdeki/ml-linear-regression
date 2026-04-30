import numpy as np
from enum import Enum


class ScalingMethod(Enum):
    Z_SCORE = "z_score"
    MIN_MAX = "min_max"
    MAX = "max"
    MEAN = "mean"


class FeatureScaler:

    def __init__(self, x: np.ndarray, feature_mask: np.ndarray, method: ScalingMethod | None = None):
        """
        Args:
            x: Training data of shape (n_samples, n_features).
            feature_mask: Boolean or binary array of length n_features indicating which columns to scale.
            method: Scaling method to use. Defaults to Z_SCORE if None.
        """
        self._x = x
        self._mask = feature_mask.astype(bool)
        self._method = method if method is not None else ScalingMethod.Z_SCORE
        self._params: dict | None = None

    @classmethod
    def from_mask(cls, x: np.ndarray, feature_mask: np.ndarray, method: ScalingMethod) -> "FeatureScaler":
        """Create a FeatureScaler from a boolean or binary feature mask."""
        return cls(x, feature_mask, method)

    @classmethod
    def from_columns(cls, x: np.ndarray, all_columns: list[str], scale_columns: list[str], method: ScalingMethod | None = None) -> "FeatureScaler":
        """
        Create a FeatureScaler by specifying column names to scale.

        Args:
            x: Training data of shape (n_samples, n_features).
            all_columns: Ordered list of all column names in x.
            scale_columns: Subset of column names to scale.
            method: Scaling method to use. Defaults to Z_SCORE if None.
        """
        indices = [all_columns.index(col) for col in scale_columns]
        return cls.from_indices(x, indices, method)

    @classmethod
    def from_indices(cls, x: np.ndarray, indices: list[int], method: ScalingMethod | None = None) -> "FeatureScaler":
        """
        Create a FeatureScaler by specifying column indices to scale.

        Args:
            x: Training data of shape (n_samples, n_features).
            indices: List of column positions to scale.
            method: Scaling method to use. Defaults to Z_SCORE if None.
        """
        mask = np.zeros(x.shape[1], dtype=int)
        mask[indices] = 1
        return cls(x, mask, method)

    def fit(self) -> "FeatureScaler":
        """Compute scaling parameters from the training data. Returns self for chaining."""
        self._params = {}
        self._params["mean"] = np.mean(self._x[:, self._mask], axis=0)
        self._params["std"] = np.std(self._x[:, self._mask], axis=0)
        self._params["min"] = np.min(self._x[:, self._mask], axis=0)
        self._params["max"] = np.max(self._x[:, self._mask], axis=0)
        return self

    def __max_normalization(self, x: np.ndarray) -> np.ndarray:
        """Scale by dividing by the max: x / max."""
        return x / self._params["max"]

    def __min_max_normalization(self, x: np.ndarray) -> np.ndarray:
        """Scale to [0, 1] range: (x - min) / (max - min)."""
        x_minus_min = x - self._params["min"]
        max_minus_min = self._params["max"] - self._params["min"]
        return x_minus_min / max_minus_min

    def __mean_normalization(self, x: np.ndarray) -> np.ndarray:
        """Center around zero within [−1, 1]: (x - mean) / (max - min)."""
        x_minus_mean = x - self._params["mean"]
        max_minus_min = self._params["max"] - self._params["min"]
        return x_minus_mean / max_minus_min

    def __z_score_normalization(self, x: np.ndarray) -> np.ndarray:
        """Standardize to zero mean and unit variance: (x - mean) / std."""
        x_minus_mean = x - self._params["mean"]
        return x_minus_mean / self._params["std"]

    def __max_inverse_normalization(self, x: np.ndarray) -> np.ndarray:
        """Inverse of max normalization: x * max."""
        return x * self._params["max"]

    def __min_max_inverse_normalization(self, x: np.ndarray) -> np.ndarray:
        """Inverse of min-max normalization: x * (max - min) + min."""
        max_minus_min = self._params["max"] - self._params["min"]
        return x * max_minus_min + self._params["min"]

    def __mean_inverse_normalization(self, x: np.ndarray) -> np.ndarray:
        """Inverse of mean normalization: x * (max - min) + mean."""
        max_minus_min = self._params["max"] - self._params["min"]
        return x * max_minus_min + self._params["mean"]

    def __z_score_inverse_normalization(self, x: np.ndarray) -> np.ndarray:
        """Inverse of z-score normalization: x * std + mean."""
        return x * self._params["std"] + self._params["mean"]


    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Apply scaling to the masked columns of x.

        Args:
            x: Array of shape (n_samples, n_features) to scale.

        Returns:
            Copy of x with masked columns scaled.
        """
        if self._params is None:
            raise RuntimeError("call fit() before transform()")
        result = x.copy()
        masked_x = x[:, self._mask]
        match self._method:
            case ScalingMethod.MAX:
                result[:, self._mask] = self.__max_normalization(masked_x)
            case ScalingMethod.MIN_MAX:
                result[:, self._mask] = self.__min_max_normalization(masked_x)
            case ScalingMethod.MEAN:
                result[:, self._mask] = self.__mean_normalization(masked_x)
            case ScalingMethod.Z_SCORE:
                result[:, self._mask] = self.__z_score_normalization(masked_x)
        return result


    def fit_transform(self) -> np.ndarray:
        """Fit on the training data and return it scaled."""
        return self.fit().transform(self._x)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        """
        Reverse the scaling on the masked columns of x.

        Args:
            x: Scaled array of shape (n_samples, n_features).

        Returns:
            Copy of x with masked columns restored to original scale.
        """
        if self._params is None:
            raise RuntimeError("call fit() before inverse_transform()")
        result = x.copy()
        masked_x = x[:, self._mask]
        match self._method:
            case ScalingMethod.MAX:
                result[:, self._mask] = self.__max_inverse_normalization(masked_x)
            case ScalingMethod.MIN_MAX:
                result[:, self._mask] = self.__min_max_inverse_normalization(masked_x)
            case ScalingMethod.MEAN:
                result[:, self._mask] = self.__mean_inverse_normalization(masked_x)
            case ScalingMethod.Z_SCORE:
                result[:, self._mask] = self.__z_score_inverse_normalization(masked_x)
        return result
