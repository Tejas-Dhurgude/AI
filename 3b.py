import numpy as np
from sklearn.metrics import classification_report, accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LogisticRegression
import pandas as pd
import matplotlib.pyplot as plt

# Updated dataset
df = [
    [1.4, 'Iris-setosa'],
    [1.3, 'Iris-setosa'],
    [1.5, 'Iris-setosa'],
    [4.7, 'Iris-versicolor'],
    [4.5, 'Iris-versicolor'],
    [4.9, 'Iris-versicolor'],
    [6.0, 'Iris-virginica'],
    [5.1, 'Iris-virginica'],
    [5.9, 'Iris-virginica'],
    [6.1, 'Iris-virginica'],
    [5.6, 'Iris-virginica'],
    [6.7, 'Iris-virginica'],
    [5.8, 'Iris-virginica'],
    [5.7, 'Iris-virginica'],
    [5.4, 'Iris-virginica'],
    [5.2, 'Iris-virginica'],
    [4.8, 'Iris-virginica'],
    [4.6, 'Iris-versicolor'],
    [4.4, 'Iris-versicolor'],
    [4.8, 'Iris-versicolor']
]

# Create DataFrame
df = pd.DataFrame(df, columns=['Petal Length (cm)', 'Species'])

# Encoding categorical variable 'Species'
df['Species'] = df['Species'].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})

# Custom implementation of logistic regression
class CustomLogisticRegression:
    def __init__(self, learning_rate=0.01, n_iter=1000):
        self.learning_rate = learning_rate
        self.n_iter = n_iter

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        self.theta = np.zeros(X.shape[1])

        for _ in range(self.n_iter):
            z = np.dot(X, self.theta)
            h = self.sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / len(y)
            self.theta -= self.learning_rate * gradient

    def predict(self, X):
        z = np.dot(X, self.theta)
        return np.round(self.sigmoid(z)).astype(int)

# Train custom logistic regression model
custom_model = CustomLogisticRegression()
custom_model.fit(df[['Petal Length (cm)']], df['Species'])
y_pred_custom = custom_model.predict(df[['Petal Length (cm)']])

# Train Scikit-learn logistic regression model
sklearn_model = LogisticRegression()
sklearn_model.fit(df[['Petal Length (cm)']], df['Species'])
y_pred_sklearn = sklearn_model.predict(df[['Petal Length (cm)']])

# Compare predictions
print("\nCustom Implementation Classification Report:")
print(classification_report(df['Species'], y_pred_custom))

print("\nScikit-learn Logistic Regression Classification Report:")
print(classification_report(df['Species'], y_pred_sklearn))

# Calculate accuracy for both models
accuracy_custom = accuracy_score(df['Species'], y_pred_custom)
accuracy_sklearn = accuracy_score(df['Species'], y_pred_sklearn)

# Calculate Mean Absolute Error for both models
mae_custom = mean_absolute_error(df['Species'], y_pred_custom)
mae_sklearn = mean_absolute_error(df['Species'], y_pred_sklearn)

# Calculate Mean Squared Error for both models
mse_custom = mean_squared_error(df['Species'], y_pred_custom)
mse_sklearn = mean_squared_error(df['Species'], y_pred_sklearn)

# Calculate Root Mean Squared Error for both models
rmse_custom = np.sqrt(mse_custom)
rmse_sklearn = np.sqrt(mse_sklearn)

# Print the metrics
print("\nMetrics for Custom Implementation:")
print("Accuracy:", accuracy_custom)
print("Mean Absolute Error:", mae_custom)
print("Mean Squared Error:", mse_custom)
print("Root Mean Squared Error:", rmse_custom)

print("\nMetrics for Scikit-learn Logistic Regression:")
print("Accuracy:", accuracy_sklearn)
print("Mean Absolute Error:", mae_sklearn)
print("Mean Squared Error:", mse_sklearn)
print("Root Mean Squared Error:", rmse_sklearn)


# Scatter plot of the data
plt.scatter(df['Petal Length (cm)'], df['Species'], color='blue')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Species')
plt.title('Scatter plot of Petal Length against Species')
plt.show()

# Plot showing the predicted values
plt.plot(df.index, df['Species'], color='red', label='Actual')
plt.plot(df.index, y_pred_custom, color='blue', linestyle='--', label='Custom Prediction')
plt.plot(df.index, y_pred_sklearn, color='green', linestyle=':', label='Sklearn Prediction')
plt.xlabel('Index')
plt.ylabel('Species')
plt.title('Predicted Values')
plt.legend()
plt.show()
