import numpy as np
# import matplotlib.pyplot as plt # this allows you to plot graph
import pandas as pd

dataset = pd.read_csv("Datasets/T01_Data.csv")
"""
any dataset on which you have to train any ml model, 
u have the same entities which are features and dependent variable vector

in the above dataset, country salary age are the features
and purchased is dependent variable vector

iloc stands for locate indexes
"""
x = dataset.iloc[:, :-1].values  # [rows, column] --> included : Excluded
# using .values here, means it will give the Numpy array of data's
y = dataset.iloc[:, [-1]].values

# print(x)
# print(y)


# taking care of missing data --->
# this missing salary we will replace with the average of the others salary
# not necessary to replace the missing salary with average value, we also replace with mode,
# median, but the most classic one is average
# other thing what we can do is to remove all the rows which have any of the column miss, applied when missing data is 1%
from sklearn.impute import SimpleImputer
# bellow line is making a tool to make all nan value change
# this can only be applied on numerical value
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')  # making the object of a class
# np.nan means, empty value
# strategy here means, which method you want to use, mean median or mode
# we have to now fit the data instead of nan
# we will fit in both age and salary data
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])  # this will exactly do the missing data in the dataset
# transform feature will actually return the new update version of the dataset

# print(x)


"""
Encoding the categorical data
here we have two columns as categorical data -> country and Purchased

To make machine learning model understand String is difficult, so we need to make the these string in numeric
here we have categorical data
one idea to represent the country column as 0->France, 1->Spain, 2->Germany
however if we do this our future machine learning model can understand that bcz France is 0 spain is 1 germany is 2
there is a numerical order between these three country, mostly it could interpret that this order matters but of course
that's absolutely not the case, so we want to avoid such a interpretation bcz that may cause some misinterpreted
co-relation between the features and the outcome want to predict, therefore we can actually do much better than encode
these three countries as above, the thing which we can do encoding is one hot encoding.
And one hot encoding consists of turing this country column into three column.
why three columns?
because there are actually three different classes in this country column.
oneHot encoding(OHE) consists of creating binary vectors of each of these countries

In our dataset, in country column, we have three categorical values
so changing those categories into number
France - [1,0,0]
Spain - [0,1,0]
Germany - [0,0,1]
"""
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# https://www.youtube.com/watch?v=5TVj6iEBR4I

# in transformer we have one tuple bcz we have to encode only one column of independent variable
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
# here we specify what kind of transformation we want and which index have to transform
# remainder --> which will specify that we have to do of the remaining column, so we have to pass through here
# if at the place of passthrough we have wrote drop then the other column got remove
x = np.array(ct.fit_transform(x))  # this will return update version of dataset
# it made the particular column for France, Spain and Germany

# print(x)


# Encoding the dependent Variable
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
# y = np.array(y)
# y = y.reshape(2, 5)
# y = y.ravel()
# print(y)
# print(y.shape)
y = le.fit_transform(y)

# splitting the dataset into training and test set
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
print(x_train)
print(x_test)
print(y_train)
print(y_test)


# Feature Scaling
"""
Do we have to apply feature scaling before splitting or after?
after splitting
bcz while calculating the mean and deviation will get change bcz of test set,we only want
the mean and deviation of only training set and then we will validate the training model on with test set

so, we are prevent information leakage on the test set which supposed not to have until training is done


Standardisation ---> [-3,+3]
This works well all the time, therefore more recommended

Normalisation ---> [0:1]
This is recommended when you have normal distribution in most of your feature


so here we will go for standardisation
"""
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
# standardisation is not require for that column whose value is already between [-3:3](dummy variable)
# fit is used to calculate the necessary parameter of the standard deviation, like here -> average and variance
# transform is use to put that calculated parameter in standard deviation formula and calculate the standard
# deviation for each data
x_train[:, 3:5] = sc.fit_transform(x_train[:, 3:5])
x_test[:, 3:5] = sc.transform(x_test[:, 3:5])

print(x_train)
print(x_test)


"""
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
dataset = pd.read_csv("Datasets/T01_Data.csv")

# print(dataset.isnull().sum())  # to check the number of nan value in each column

# applying SimpleImputer, OrdinalEncoding using ColumnTransformation
cT = ColumnTransformer(transformers=[
    ('simpleImputer', SimpleImputer(missing_values=np.nan, strategy='mean'), ['Age', 'Salary']),
    ('oneHotEncoding', OneHotEncoder(dtype=np.int32, sparse=False, drop='first'), ['Country']),
], remainder='passthrough')

dataset = cT.fit_transform(dataset)  # type -> numpy.ndarray

x_feature = dataset[:, :-1]  # type -> numpy.ndarray
y_outcome = dataset[:, [-1]]  # numpy.ndarray

# doing LabelEncoding
lE = LabelEncoder()
y_outcome = lE.fit_transform(y_outcome)
print(y_outcome)
"""
