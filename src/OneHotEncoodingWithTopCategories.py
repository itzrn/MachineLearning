import pandas as pd

dataset = pd.read_csv("Datasets/carsOneHotEncoding.csv")
count = dataset['brand'].value_counts()  # to access index we use the syntax -> count[0]
print("Count ->\n", count)

print()

# we will take threshold here, bellow that all classes will be transfer in to a new matrix
threshold = 100
lessThan100 = count[count <= threshold].index
print("Less Than 100 ->\n", lessThan100)

print()

graterThan100 = pd.get_dummies(dataset['brand'].replace(lessThan100, 'uncommon'))
print("Greater Than 100 ->\n", graterThan100)
