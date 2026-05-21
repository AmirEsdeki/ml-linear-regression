import numpy as np


class RegressionMetrics:

    @staticmethod
    def evaluate(predictions: np.ndarray, y_true: np.ndarray) -> tuple[float, float, float]:
        """Evaluate regression predictions using MAE, RMSE, and R².

        Args:
            predictions (np.ndarray): Predicted values of shape (m,).
            y_true (np.ndarray): True target values of shape (m,).

        Returns:
            tuple[float, float, float]: MAE, RMSE, and R² score.
        """
        mae = np.mean(np.abs(predictions - y_true))
        rmse = np.sqrt(np.mean((predictions - y_true) ** 2))
        ss_res = np.sum((predictions - y_true) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - ss_res / ss_tot
        return mae, rmse, r2


class ClassificationMetrics:

    @staticmethod
    def evaluate(predictions: np.ndarray, y_true: np.ndarray, threshold: float = 0.5) -> tuple[float, float, float, float]:
        """Evaluate logistic regression predictions using accuracy, precision, recall, and F1.

        Args:
            predictions (np.ndarray): Predicted probabilities of shape (m,), each value in (0, 1).
            y_true (np.ndarray): True binary labels of shape (m,), each value 0 or 1.
            threshold (float): Probability cutoff for classifying as positive. Defaults to 0.5.

        Returns:
            tuple[float, float, float, float]: Accuracy, precision, recall, and F1 score.
        """
        labels = (predictions >= threshold).astype(int)

        tp = np.sum((labels == 1) & (y_true == 1))
        fp = np.sum((labels == 1) & (y_true == 0))
        fn = np.sum((labels == 0) & (y_true == 1))

        accuracy = np.mean(labels == y_true)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return accuracy, precision, recall, f1
