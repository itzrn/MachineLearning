import numpy as np

# Array filled with 0's
zeroArr = np.zeros(4) # creation of 1D
print(zeroArr)

print()

zeroArr = np.zeros((3,4))
print(zeroArr)

print()

# Array filled with 1's
oneArr = np.ones((4))
print(oneArr)

print()

oneArr = np.ones((3,4))
print(oneArr)

print()

# Create an empty array
emptyArr = np.empty(4) # this taking previously created array data
print(emptyArr)

print()

# if it won't get the previously created array with same dimension it will generate random number like bellow
emptyArr = np.empty((4,5))
print(emptyArr)

print()

# An Array with a range of elements
rangeArr = np.arange(4)
print(rangeArr)

print()

# Array diagonal element filled with 1's
digArr = np.eye(3) # it will make 2D array diagonal element as 1
print(digArr)

print()

digArr = np.eye(3,4)
print(digArr)

print()

# Create an array with values that are spaced linearly in a specified interval
lineSpaceArr = np.linspace(0,20 , num=5) # means 10 element with spacing 5, starting with 1 and ending with 10
print(lineSpaceArr)
