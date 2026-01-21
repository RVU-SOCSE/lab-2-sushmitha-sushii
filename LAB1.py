import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# Given data
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([52, 55, 61, 70, 82])

# -------------------------------
# Model A: Simple Linear Regression
# -------------------------------
linear_model = LinearRegression()
linear_model.fit(X, y)

# Coefficients
beta_0 = linear_model.intercept_
beta_1 = linear_model.coef_[0]

print("Model A (Linear Regression):")
print(f"Equation: y = {beta_0:.2f} + {beta_1:.2f}x")

# Prediction for x = 6
x_test = np.array([[6]])
y_pred_linear = linear_model.predict(x_test)
print(f"Prediction at x = 6: {y_pred_linear[0]:.2f}")

# Training MSE
y_train_pred_linear = linear_model.predict(X)
mse_linear = mean_squared_error(y, y_train_pred_linear)
print(f"Training MSE: {mse_linear:.2f}\n")

# -----------------------------------------
# Model B: Polynomial Regression (Degree 4)
# -----------------------------------------
poly = PolynomialFeatures(degree=4)
X_poly = poly.fit_transform(X)

poly_model = LinearRegression()
poly_model.fit(X_poly, y)

print("Model B (Polynomial Regression - Degree 4):")
print("Coefficients:", poly_model.coef_)
print("Intercept:", poly_model.intercept_)

# Prediction for x = 6
x_test_poly = poly.transform(x_test)
y_pred_poly = poly_model.predict(x_test_poly)
print(f"Prediction at x = 6: {y_pred_poly[0]:.2f}")

# Training MSE
y_train_pred_poly = poly_model.predict(X_poly)
mse_poly = mean_squared_error(y, y_train_pred_poly)
print(f"Training MSE: {mse_poly:.2f}")