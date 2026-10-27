import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

df = pd.read_csv("Advertising.csv")
df = df.dropna()
df = df.drop_duplicates()

X = df["Radio"]
Y = df["Sales"]
intercept = 0
slope = 0
learning_rate = 0.001
iterations = 1000
n = len(X)
cost_history = []

for i in range(iterations):
    y_pred = slope * X + intercept
    error = y_pred - Y
    cost = np.mean(error ** 2)
    cost_history.append(cost)
    slope_gradient = (2/n) * np.sum(error * X)
    intercept_gradient = (2/n) * np.sum(error)
    slope = slope - learning_rate * slope_gradient
    intercept = intercept - learning_rate * intercept_gradient

y_pred = slope * X + intercept
print("\nSlope:", slope)
print("\nIntercept:", intercept)
rmse = np.sqrt(mean_squared_error(Y, y_pred))
print("\nRMSE:", rmse)

plt.plot(range(iterations), cost_history)
plt.xlabel("Iterations")
plt.ylabel("Mean Squared Error")
plt.title("Gradient Descent Cost")
plt.show()

plt.scatter(X, Y, label = "Actual Sales")
sorted_indices = np.argsort(X)
plt.plot(X[sorted_indices], y_pred[sorted_indices], color = "Red", label = "Regression Line")
plt.title("Linear Regression using Gradient Descent")
plt.legend()
plt.show()


















