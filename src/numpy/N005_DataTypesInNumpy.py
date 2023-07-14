import numpy as np
"""
i - integer
b - boolean
u - unsigned integer
f - float
c - complex float
m - timedelta
M - datetime
O - object
S - string
U - Unicode string
V - the fixed chunk of memory for other types(void)
"""
arr = np.array([1,2,3,4]) # int32 -> this is integer data type
print("Data Type Of Array ->",arr.dtype)

print()

arr = np.array([1.3,2.2,3.5,4.6]) # float64 -> this is float data type
print("Data Type Of Array ->",arr.dtype)

print()

arr = np.array(["1","2","3","4"]) # <U1 -> this string data type
print("Data Type Of Array ->",arr.dtype)

print()

arr = np.array([1,2,3,4 ,"A", "B", "C", "D"]) # U11 -> when an array contain both integer and string
print("Data Type Of Array ->",arr.dtype)

print()

# conversion from data type to another
x = np.array([1,2,3,4])
print(x.dtype)
# converting to int8
x.dtype = np.int8  # or -> x = np.array([1,2,3,4],dtype = int8)
print(x.dtype)

print()

# using above notation for the conversion
y = np.array([1,2,3,4],dtype="f")
print(y.dtype)
print(y)

print()

# using like a function
arr = np.array([1,2,3,4,5])
print(arr.dtype)
arr = np.float32(arr)
print(arr.dtype)

print()

# using asType function

ass = np.array([1,2,3,4,5])
ass = ass.astype(float)
print(ass)
print(ass.dtype)