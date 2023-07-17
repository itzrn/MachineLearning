import numpy as np

var = np.array([[1, 2], [1, 2]])
print(var)
print(var.shape)  # this function fives the shape of the array

print()

var1 = np.array([1, 2, 3, 4, 5], ndmin=4)  # this will make array of 4 dimension
print(var1)
print(var1.shape)

print()

# reshape
var2 = np.array([1, 2, 3, 4, 5, 6])  # array can only be reshape if and only if you create the same element storing.
# like 6 element's are there then (3,2) | (2,3) is possible
print(var2)
x = var2.reshape(3, 2)  # (number of rows, number of column)
print(x)

print()

# converting in multi dimension
var3 = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18])
x2 = var3.reshape(3,3,2)
print(x2)
print("shape ->",x2.shape)

print()

x3 = x2.reshape(-1) # this will directly convert it to single dimension array
print(x3)