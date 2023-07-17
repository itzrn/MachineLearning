import numpy as np
import pandas as pd

dataset = pd.read_csv("Datasets/covid_toyColumnTransformation.csv")  # type -> DataFrame
print("AAM ZINDAGI")
print()

x_feature = dataset.iloc[:, 0: -1].values  # type -> Numpy.ndarray
y_outcome = dataset.iloc[:, [-1]].values  # type -> Numpy.ndarray

"""
Count the number of similar rows of dataset/dataframe : dataset.value_counts()
Count the number of categories a column of dataset have : df['your column name'].value_counts()
Check for NaN under a single DataFrame column: df['your column name'].isnull().values.any()
Count the NaN under a single DataFrame column: df['your column name'].isnull().sum()
Check for NaN under an entire DataFrame: df.isnull().values.any()
Count the NaN under an entire DataFrame: df.isnull().sum().sum()
Count the NaN under an entire DataFrame for each column: df.isnull().sum()
"""
print("To get the number of NaN in entire dataset in every column ->\n", dataset.isnull().sum())
print()

# filling missing values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# imputer.fit(x_feature[:, [2]])
x_feature[:, [2]] = imputer.fit_transform(x_feature[:, [2]])  # u can fit and the transform separately or
# u can use fit_transform() to fit and transform at once
# print(x_feature)

# doing ordinal encoding
from sklearn.preprocessing import OrdinalEncoder
oe = OrdinalEncoder(categories=[['Mild', 'Strong']], dtype=np.int32)
oe.fit(x_feature[:, [3]])
x_feature[:, [3]] = oe.transform(x_feature[:, [3]])
print("x_feature_column_fever ->\n", x_feature[:, [3]])

print()

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(drop='first', dtype=np.int32, sparse=False)  # sparse=False make the data in numpy.ndarray
new_x_feature = ohe.fit_transform(x_feature[:, [1, 4]])
x_feature = np.hstack((x_feature[:, [0, 2, 3]], new_x_feature))
print(x_feature.shape)
print(x_feature)

# now splitting in training and testing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_feature, y_outcome)
