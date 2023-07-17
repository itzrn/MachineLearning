"""
This is called nominal coding

let we have a column contain three class of color -> yellow, blue, red
so instead of having one column we will make three column
like -> yellow - [1, 0, 0]
        blue - [0, 1, 0]
        red - [0, 0, 1]
like if we have 50 different categories then e have to make 50 different column
these generated extra columns called dummy variable

Dummy Variable Trap ->
       a concept comes -> multi-collinearity, means in data there should not be any mathematical relation.
       like if we see the above data yellow, blue, red
       these three have a mathematical relation, i.e. sum of the row will be 1
                            Sum
       yellow - [1, 0, 0] - 1
       blue - [0, 1, 0 ]  - 1
       red - [0, 0, 1]    - 1
       this make some obstacles like in linear regression, logistic regression kind of model
       so to remove multi-collinearity we remove one column among those(generally we remove first one, but we can remove any)
       now the representation changes
                         Sum
       [0, 0] - yellow - 0
       [1, 0] - blue   - 1
       [0, 1] - red    - 1
       therefore, on having n categories we make n-1 categorical column and making every column independent

OHE -> for most frequent variable
let we have a column which contain 40 different brands
the we are going to have 39 more different column, which increases the dimension of the data and processing gets slow
So, in this scenario we keep the frequent category and other category we put in the completely new category
this approach only be used when some category is very frequent and other category is very less frequent
"""

import pandas as pd
import numpy as np

dataset = pd.read_csv("Datasets/carsOneHotEncoding.csv")
print("value count of brand ->\n", dataset['brand'].value_counts())
# here we have 32 brands, by using len(dataset['brand'].value_counts()) or
# dataset['brand].nunique() we can get there are 32 brands

print()

# print("To find the number of categories in brand column ->\n", dataset['brand'].nunique())
# OHE using Pandas
# we doing OHE on just two columns here -> fuel and owner
df = pd.get_dummies(dataset, columns=['fuel', 'owner'])  # here 12 columns(12 bcz we didn't use the multi-collinearity problem)
print("OHE", df)

print()

# doing k-1 encoding(solving multi collinaerity)
df1 = pd.get_dummies(dataset, columns=['fuel', 'owner'], drop_first=True)  # here 10 columns
print(df1)

"""
when we use Pandas for OHE, there is a problem
Pandas won't learn at what position which column is there, on running second time result can be different
"""

print()

# OHE using sklearn -> this we use in machine learning
x_feature = dataset.iloc[:, :-1].values  # Numpy.ndarray
y_outcome = dataset.iloc[:, -1].values  # Numpy.ndarray
# print(y_outcome)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_feature, y_outcome, test_size=0.2, random_state=2)  # Numpy.ndarray
# print(x_train)
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(drop='first', sparse=False)  # we are dropping first column of created new column of nominal encoding
# one parameter we have in OneHotEncoder is sparse=True -> True value is default,
# if we do False then it will automatically convert it to array

# one parameter we have dtype = np.int32 -> it will convert all the float value to integer value

x_train_new = ohe.fit_transform(x_train[:, [2, 3]])  # Numpy.ndarray
x_test_new = ohe.transform(x_test[:, [2, 3]])  # Numpy.ndarray
print("x_train_new ->\n", x_train_new)
print()
print("x_test_new ->\n", x_test_new)

print()

new_x_train = np.hstack((x_train[:, [0, 1]], x_train_new))  # merging two arrays of training data
new_x_test = np.hstack((x_test_new[:, [0, 1]], x_test_new))  # merging two arrays of testing data
print("New X Training Value ->\n", new_x_train.shape)

print()

# One Hot Encoding with Top Categories
