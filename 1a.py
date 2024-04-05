
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# df=pd.read_csv('filename.csv')

# Define dataset
data = {
    'distance': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    'fare': [2, 3, 5, 8, 8, 9, 10, 13, 12, 10, 13, 12, 17]
}

# Create DataFrame
df = pd.DataFrame(data)

# Split into training and testing set
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

# Use .copy() to avoid SettingWithCopyWarning when modifying dataframes later
df_train = df_train.copy()
df_test = df_test.copy()



# Calculate the necessary sums and means from the training set
sum_x = df_train['distance'].sum()
sum_y = df_train['fare'].sum()
no_rows = len(df_train)
mean_x = sum_x / no_rows
mean_y = sum_y / no_rows

# Deviations from the mean
dev_x = df_train['distance'] - mean_x
dev_y = df_train['fare'] - mean_y

# Calculate the product of deviations and sum of the product
p = dev_x * dev_y
SOP = p.sum()

# Calculate squared deviations of x and the slope (m)
sq_dev_x = dev_x**2
m = SOP / sq_dev_x.sum()

# Calculate the intercept (b)
b = mean_y - (m * mean_x)

# Output the linear regression equation
print(f'y = {m}x + {b}')

# Predictions for test set
df_test['predictions'] = m * df_test['distance'] + b

# Calculate Mean Squared Error (MSE) for test set
mse = ((df_test['fare'] - df_test['predictions'])**2).mean()
print("Mean Squared Error (MSE):", mse)

# Custom "Accuracy"
accuracy = max(0, 100 - mse)
print(f'Custom "Accuracy": {accuracy}%')

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(df_train['distance'], df_train['fare'], color='blue', label='Training data')
plt.scatter(df_test['distance'], df_test['fare'], color='red', label='Testing data')
plt.plot(df_test['distance'], df_test['predictions'], color='green', label='Predictions')  # Corrected this line
plt.xlabel('Distance')
plt.ylabel('Fare')
plt.title('Linear Regression')
plt.legend()
plt.grid(True)
plt.show()






## Binary Logistic

