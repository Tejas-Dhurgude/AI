import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = {
    'Speed': np.array([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]),
    'Processor': np.array([0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]),
    'Price': np.array([3.75, 2.0, 0.25, 1.0, 0.25, 2.0, 3.75])
}
df = pd.DataFrame(data)

X_train = df[['Speed', 'Processor']].values
y_train = df['Price'].values

degree = 3

# # Polynomial regression using sklearn
# poly_features = PolynomialFeatures(degree=degree)
# X_train_poly = poly_features.fit_transform(X_train)

# Manually creating polynomial features
X_train_poly = np.column_stack((X_train, X_train[:, 0] ** 2, X_train[:, 1] ** 2, X_train[:, 0] * X_train[:, 1]))

model = LinearRegression()
model.fit(X_train_poly, y_train)

y_train_pred = model.predict(X_train_poly)

train_rmse_sklearn = np.sqrt(mean_squared_error(y_train, y_train_pred))
print(f"Train RMSE (sklearn): {train_rmse_sklearn}")

# Plotting 3D surface
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['Speed'], df['Processor'], df['Price'], color='blue', label='Original data')
ax.scatter(df['Speed'], df['Processor'], y_train_pred, color='red', label='Predicted data')
ax.set_xlabel('Speed')
ax.set_ylabel('Processor')
ax.set_zlabel('Price')
ax.set_title('Polynomial Regression')
plt.legend()
plt.show()

# Equation, slope, intercepts
print("Equation of the polynomial regression model:")
equation = "Price = "
coefficients = model.coef_
for i in range(degree + 1):
    if i == 0:
        equation += f"{coefficients[i]:.2f}"
    else:
        equation += f" + {coefficients[i]:.2f} * (Speed ** {i})"

print(equation)
print("")

# Mean Absolute Error (MAE)
mae_sklearn = np.mean(np.abs(y_train - y_train_pred))
print(f"Mean Absolute Error (MAE): {mae_sklearn}")

# Mean Squared Error (MSE)
mse_sklearn = mean_squared_error(y_train, y_train_pred)
print(f"Mean Squared Error (MSE): {mse_sklearn}")

# Root Mean Squared Error (RMSE)
rmse_sklearn = np.sqrt(mse_sklearn)
print(f"Root Mean Squared Error (RMSE): {rmse_sklearn}")

from sklearn.metrics import r2_score

# Calculate R-squared
r2 = r2_score(y_train, y_train_pred)
print(f"R-squared: {r2}")
