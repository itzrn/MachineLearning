# import pandas as pd
# dataset = pd.read_csv("Datasets/titanic.csv")
# print("Data Type of each Column ->\n", dataset.dtypes)  # this is to know the datatype of each column
#
# print()
#
# print("Column Names ->\n", dataset.columns)
# """
# EDA -> Exploratory Data Analysis -> it is highly iterative(means to know the data well, u need to see the
# data again and again)
#
# PassengerId -> This is just a generated ID (Categorical)Pclass -> which class did the passenger ride First,
# second or third Name -> Passenger Name (Categorical)Sex -> Male or Female Age -> (Categorical)Sibsp -> where the
# passenger's spouse or siblings with them on the ship (Categorical)Parch -> where the passenger's parent or children
# with them on the ship Ticket -> Ticket number fare -> ticket price Cabin -> (Categorical)Embarked -> the place the
# ship was going, it was going to three places S->116 people,C -> 65 people,Q -> 2 people (Categorical)Survived ->
# Did the passenger survive sinking of the titanic """
#
# print()
#
# # print("Number of places where the ship has to go ->\n", dataset['Embarked'].value_counts())  To know the number of
# # places ship has to go
#
# print("To see the non-null info ->\n", dataset.info())
#
# print()
#
# print("To see the number of null values in each particular column ->\n", dataset.isnull().sum())
# """
# Missing values is Age, Cabin and Embarked
# More than 70% values are missing in cabin column, we will drop this column
# few columns have inappropriate datatype
# """
# print()
#
# # dropping Cabin column
# dataset.drop(columns=['Cabin'], inplace=True)  # writing inplace=True drop the column from the original dataframe
#
# # replacing all nan value of ge with the mean
# dataset['Age'].fillna(dataset['Age'].mean(), inplace=True)
#
# # In Embarked column we will replace the nan value with most repeated value
# # In Embarked column S is the most repeated value
# dataset['Embarked'].fillna('S', inplace=True)

# Importing the necessary libraries
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Load the dataset
df = pd.read_csv("titanic.csv")

# Identify the categorical data
categorical_feature = ['Sex', 'Embarked', 'Pclass']
# Implement an instance of the ColumnTransformer class
ct = ColumnTransformer(transformers=[
    ('encoder', OneHotEncoder(), categorical_feature)
    ], remainder='passthrough')

# Apply the fit_transform method on the instance of ColumnTransformer
X = ct.fit_transform(df)

# Convert the output into a NumPy array
X = np.array(X)

# Use LabelEncoder to encode binary categorical data
le = LabelEncoder()
y = le.fit_transform(df['Survived'])

# Print the updated matrix of features and the dependent variable vector
print("Updated matrix of features: \n", X)
print("Updated dependent variable vector: \n", y)


