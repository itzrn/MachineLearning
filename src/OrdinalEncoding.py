"""
data is majorly divided into two types -> numerical and categorical
categorical is divided into two types -> nominal and ordinal
ordinal
   data having order between categories or relationship
   eg -> high school marks, review(excellent, good, bad(relation is like excellent is greater than good, bad))
   these type of data have have relation between
   here we will use ordinal encoding

nominal
   data having no relation between or any order
   eg -> state, branches of engineering
   here we use oneHot Encoding

when or output feature is categorical data then we wont use ordinal or oneHot encoding
instead we use Label encoder(this is actually work like ordinal encoder but label encoder ia specially designed for
output labels) which is present in sklearn
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder

dataset = pd.read_csv("Datasets/customerOrdinalEncoding.csv")
print("Some samples ->\n", dataset.sample(5))  # this wil bring any 5 random sample
# On review, education we will apply ordinal encoding
# On purchased we will apply label encoding

print()

df = dataset.iloc[:, 2:]
print("We will be seeing only for ordinal encoding and label encoding, so 2 feature and 1 outcome ->\n", df)

print()

# doing train and test split
x_train, x_test, y_train, y_test = train_test_split(df.iloc[:, 0:2], df.iloc[:, -1], test_size=0.2)
print("x_train ->\n", x_train)
print()
print("x_test ->\n", x_test)
print()
print("y_train ->\n", y_train)
print()
print("y_test ->\n", y_test)
print()

# doing ordinal encoding
oe = OrdinalEncoder(categories=[["Poor", "Average", "Good"], ["School", "UG", "PG"]])  # right side is increasing order
"""
here we have two ordinal data one is review and other is education

while making the OrdinalEncode instance we pass a list which contain as many number of list as many number of
ordinal columns are there, and then we pass all the classes of those ordinal column
and we place the value in increasing order
"""
oe.fit(x_train)  # we have to only apply fit on x_train, automatically x_test get fit after transform
print("this gives how many number of categories we have defined ->\n", oe.categories_)
print()
x_train = oe.transform(x_train)
x_test = oe.transform(x_test)
print("x_train ->\n", x_train)
print()
print("x_test ->\n", x_test)
print()

le = LabelEncoder()  # this have no parameters
le.fit(y_train)
y_train = le.transform(y_train)
y_test = le.transform(y_test)
print("y_train->\n", y_train)
print()
print("y_test ->\n", y_test)


