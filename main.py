# type: ignore
import kagglehub
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from regression import Regression
from feature_scaler import FeatureScaler, ScalingMethod
from polynomial_features import PolynomialFeatures
from metrics import RegressionMetrics

# Download dataset from kaggle
path = kagglehub.dataset_download("uciml/red-wine-quality-cortez-et-al-2009")
print("Path to dataset files:", path)

df = pd.read_csv(path + "/winequality-red.csv")
df_test = df.sample(frac=0.2, random_state=42)
df_train = df.drop(df_test.index)

X_train = df_train.iloc[:, :-1].values
y_train = df_train.iloc[:, -1].values

X_test = df_test.iloc[:, :-1].values
y_test = df_test.iloc[:, -1].values

# Preprocessing
poly = PolynomialFeatures(degree=5)
X_train_poly = poly.transform(X_train)

scaler = FeatureScaler(x=X_train_poly, feature_mask=np.ones(X_train_poly.shape[1], dtype=int), method=ScalingMethod.Z_SCORE)
X_train_scaled = scaler.fit_transform()

X_test_scaled = scaler.transform(poly.transform(X_test))

print(f"Feature count after polynomial expansion: {X_train_scaled.shape[1]}")

# Train
model = Regression.linear(learning_rate=0.001, iterations=100000, regularization_lambda=100, tolerance=1e-7).fit(X_train_scaled, y_train)

print(f"Final cost (MSE): {model.cost_history[-1]:.6f}")

# Evaluate
predictions = model.predict(X_test_scaled)
mae, rmse, r2 = RegressionMetrics.evaluate(predictions, y_test)

print(f"\nMAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")

# Plot predicted vs actual
fig, ax = plt.subplots()
ax.scatter(y_test, predictions, alpha=0.4, s=15)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=1.5, label='Perfect fit')
ax.set_xlabel("Actual quality")
ax.set_ylabel("Predicted quality")
ax.set_title("Predicted vs Actual (Test Set)")
ax.legend()
plt.show()
