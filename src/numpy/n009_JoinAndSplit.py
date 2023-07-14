import numpy as np

# Join
var = np.array([1,2,3,4])
var1 = np.array([5,6,7,8])
ar = np.concatenate((var1,var))
# similarly with multi dimension, but dimension should be same
# and you have to give the axis along which you have to concatenate(axis = 1) 1 means roe, 0 means column
print(ar)

print()

# split array
var2 = np.array([1,2,3,4,5,6])
print(var2)

ar = np.array_split(var2, 3)
print(ar)
print(ar[0])