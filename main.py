# type: ignore
import kagglehub
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linear_regression import LinearRegression
from feature_scaler import FeatureScaler, ScalingMethod
from polynomial_features import PolynomialFeatures

#Download a model from kaggle to start our ML work
path = kagglehub.dataset_download("uciml/red-wine-quality-cortez-et-al-2009")
print("Path to dataset files:", path)

df_train = pd.read_csv(path + "/winequality-red.csv")
# print(df_train.head(100))
#take 1000 of data for test reasons
samples = 1000
df_test = df_train.sample(samples, random_state=42)
df_train = df_train.drop(df_test.index)

input_features_x = df_train.iloc[:,:-1].values
output_targets_y = df_train.iloc[:,-1:].values.flatten() #flatten to have a ndarray of shape (m) instead of (m,1)

# fig, ax = plt.subplots()
# ax.scatter(input_features_x[:, 11], output_targets_y[:], alpha=0.3, s=10)
# plt.show()

poly = PolynomialFeatures(degree=5)
input_features_x_poly = poly.transform(input_features_x)

scaler = FeatureScaler(x=input_features_x_poly, feature_mask=np.ones(input_features_x_poly.shape[1], dtype=int), method=ScalingMethod.Z_SCORE) #scaling all the inputs in case
input_features_x_scaled = scaler.fit_transform()

reg_model = LinearRegression(learning_rate=0.001, iterations=100000, regularization_lambda=50)

n = input_features_x_scaled.shape[1]
print(f"Feature count after making polynomial expansion: {n}")
w_init = np.zeros(n)
b_init = 0.0

w, b, cost_history = reg_model.gradient_descent_vectorized(
    input_features_x_scaled, w_init, b_init, output_targets_y
)

# fig, ax = plt.subplots()
# ax.plot(range(len(cost_history)), cost_history)
# ax.set_xlabel("Iteration")
# ax.set_ylabel("Cost (MSE)")
# ax.set_title("Gradient Descent Cost History")
# plt.show()

print("Cost (MSE)", cost_history[-1])

test_x = df_test.iloc[:, :-1].values
test_y = df_test.iloc[:, -1:].values.flatten()
test_x_poly = poly.transform(test_x)
test_x_scaled = scaler.transform(test_x_poly)

predictions = reg_model.predict(test_x_scaled)

# print(f"\n{'#':<6} {'Predicted':>12} {'Actual':>10}")
# print("-" * 30)
correct = 0
for i, (pred, actual) in enumerate(zip(predictions, test_y)):
    rounded = round(pred)
    if rounded == int(actual):
        correct += 1
    # print(f"{i+1:<6} {rounded:>12} {actual:>10.0f}")

print(f"\nCorrect: {correct}/{samples} ({100 * correct // samples}%)") #first try: Correct: 560/1000 (56%)

