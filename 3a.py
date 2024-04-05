import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# initial_df = pd.read_csv('Exp4.csv')
data = {
    'Exam Score': [85, 70, 95, 60, 80, 65, 90, 75, 82, 78, 88, 72, 68, 92, 77],
    'Hours Studied': [6, 4, 7, 3, 5, 3.5, 6.5, 4.5, 5.5, 4.7, 6.2, 4.2, 3.8, 7.2, 5.2],
    'Passed': [1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1]
}

initial_df=pd.DataFrame(data)

# Convert the list of lists into a numpy array.


df , test_df = train_test_split(initial_df , test_size=0.3 , random_state=42)

def linear_reg(X , Y):
    mean_EC = df[X].mean()
    mean_PR = df[Y].mean()

    df["dev_y"] = df[Y] - mean_PR
    df["dev_x"] = df[X] - mean_EC

    df["Prod_of_Dev"] = df["dev_x"] * df["dev_y"]
    Sum_of_square = (df["dev_x"] * df["dev_x"]).sum()
    Sum_of_POD = df["Prod_of_Dev"].sum()
    m = Sum_of_POD / Sum_of_square
    b = mean_PR - m * mean_EC
    return m, b

m, b = linear_reg("Exam Score", "Passed")

def Log(m , b , df):
    df["log"] = 1 / (1 + np.exp(-(m * df["Exam Score"] + b)))

def func(thresh , df):
    df['Pred']  = df['log'].apply(lambda x: 1 if x >= thresh else 0)

Log(m , b , df)
func(0.59 , df)

Log(m , b , test_df)
func(0.59 , test_df)

def calculate_accuracy(test_df):
    correct_predictions = (test_df['Pred'] == test_df['Passed']).sum()
    total_instances = len(test_df)
    accuracy = correct_predictions / total_instances * 100
    return accuracy

accuracy = calculate_accuracy(test_df)
print("Accuracy:", accuracy)

def Log(x, m, b):
    return 1 / (1 + np.exp(-(m * x + b)))

P = [Log(x, m, b) for x in np.linspace(0, 300)]
plt.plot(np.linspace(0, 300), P, label="Logistic Curve")
plt.scatter(df["Exam Score"], df["Pred"], color="green", label="Predicted Points")
plt.scatter(df["Exam Score"], df["log"], color="black", label="Predicted Log Points")
plt.legend()
plt.xlabel("Exam Score")
plt.ylabel("Log")
plt.show()

