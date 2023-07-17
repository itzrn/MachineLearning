# Importing the necessary libraries
import pandas as pd
from sklearn.datasets import load_iris

# Load iris dataset
dataset = load_iris()
print(dataset)

# Identifying the features (independent variables) and the dependent variable

# Create dataframe
df1 = pd.DataFrame(data=dataset.data, columns=dataset.feature_names)
# print(type(df1))

# Create matrix of features (X) and dependent variable vector (y)
X = df1.values  # this creates the dataframe into matrix(ndarray)
y = dataset.target  # this is already in matrix(ndarray) form
# print(type(X))

# Print your feature matrix (X) and dependent variable vector (y)
print("Matrix of features X: \n", X)
print("\nDependent variable vector y: \n", y)
