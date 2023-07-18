"""
import the necessary libraries for data preprocessing, including the StandardScaler class.
Load the "Wine Quality Red" dataset into a pandas DataFrame. You can use the pd.read_csv function for this.
Make sure you set the correct delimiter for the file.
Split your dataset intoan 80-20 training-test set. Set random_state to ensure reproducible results.
Create an instance of the StandardScaler class.
Fit the StandardScaler on features from the training set, excluding the target variable 'Quality'.
Use the "fit_transform" method of the StandardScaler object on the training dataset.
Apply the "transform" method of the StandardScaler object on the test dataset.
Print your scaled training and test datasets to verify the feature scaling process.
"""

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the Wine Quality Red dataset
dataset = pd.read_csv("Datasets/wine_quality_red.csv", delimiter=';')
print("Dataset ->\n", dataset)
print()

# Separate features and target
x_feature = dataset.iloc[:, :-1]
y_outcome = dataset.iloc[:, [-1]]

# Split the dataset into an 80-20 training-test set
X_train, X_test, y_train, y_test = train_test_split(x_feature, y_outcome, test_size=0.2, random_state=1)
print("X_train ->\n", X_train)  # ndarray
print("X_test ->\n", X_test)  # ndarray
print("y_train ->\n", y_train)  # ndarray
print("y_test ->\n", y_test)  # ndarray

print()
# Create an instance of the StandardScaler class
sc = StandardScaler()

# Fit the StandardScaler on the features from the training set and transform it
sc.fit(X_train)
X_train = sc.transform(X_train)

# Apply the transform to the test set
X_test = sc.transform(X_test)

# Print the scaled training and test datasets
print("X_train ->\n",X_train)
print("X_test ->\n", X_test)

