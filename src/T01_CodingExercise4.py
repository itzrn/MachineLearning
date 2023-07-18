# Import necessary libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(data= iris.data, columns= iris.feature_names)

# Separate features and target
X = df
y = iris.target

# Split the dataset into an 80-20 training-test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Print the dataset split
print("Training set features:\n", X_train)
print("Test set features:\n", X_test)
print("Training set labels:\n", y_train)
print("Test set labels:\n", y_test)

# Apply feature scaling on the training and test sets
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Print the scaled training and test sets
print("Scaled training set:\n", X_train)
print("Scaled test set:\n", X_test)