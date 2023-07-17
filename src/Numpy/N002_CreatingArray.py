import numpy as np

# creating array using List(List which have same type of data type elements)
a = [1, 2, 3, 4, 5]
arr = np.array(a)
print(arr)

print()

# direct creation using list
x = np.array([1, 2, 3, 4, 5])
print(x)

print()

# taking list of integer input and type casting it into array
l = []
for i in range(0,5):
    l.append(int(input("Enter number -> ")))
x = np.array(l)
print(x)
print("To know the dimension of array -->",x.ndim)

print()

# we can manually also make the any Dimension array

arr = np.array([1,2,3,4,5,6], ndmin=10)
print(arr)
print(arr.ndim)

