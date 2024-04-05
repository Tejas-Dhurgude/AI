import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# Provided data
data = {
    'Hours Studied': [5, 7, 6, 8, 4, 9, 6, 7, 5, 8, 7, 6, 5, 9, 8],
    'Previous Scores': [90, 95, 88, 92, 85, 97, 82, 89, 85, 94, 96, 87, 92, 95, 88],
    'Performance Index': [85.0, 90.0, 88.0, 95.0, 82.0, 98.0, 87.0, 91.0, 85.0, 94.0, 93.0, 89.0, 87.0, 97.0, 93.0]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Preparing the data for linear regression
X = np.column_stack((np.ones(len(df)), df['Hours Studied'], df['Previous Scores']))
y = df['Performance Index'].values

# Performing linear regression calculations
X_transpose = np.transpose(X)
X_transpose_X = X_transpose @ X
X_transpose_X_inverse = np.linalg.inv(X_transpose_X)
coefficients = X_transpose_X_inverse @ X_transpose @ y

# Predicting Performance Index values
predicted_y = X @ coefficients

# Visualizing the results
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['Hours Studied'], df['Previous Scores'], df['Performance Index'], label='Actual Performance Index', c='blue', marker='o')
ax.scatter(df['Hours Studied'], df['Previous Scores'], predicted_y, label='Predicted Performance Index', c='red', marker='x')
ax.set_xlabel('Hours Studied')
ax.set_ylabel('Previous Scores')
ax.set_zlabel('Performance Index')
plt.legend()
plt.show()

# Pair plot to visualize relationships
sns.pairplot(df)
plt.show()

# Heatmap to visualize correlations
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.show()

# Evaluating the model using metrics like Mean Absolute Error, Mean Squared Error, and Root Mean Squared Error
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(y, predicted_y)
print("\nMean Absolute Error (MAE):", mae)

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mean_squared_error(y, predicted_y))
print("Root Mean Squared Error (RMSE):", rmse)

# Printing the coefficients and intercept of the regression equation
print("\nCoefficients:")
print("Intercept:", coefficients[0])
print("Slope for Hours Studied:", coefficients[1])
print("Slope for Previous Scores:", coefficients[2])

from sklearn.metrics import r2_score

# Calculate R-squared
r_squared = r2_score(y, predicted_y)
print("R-squared (R^2):", r_squared)
