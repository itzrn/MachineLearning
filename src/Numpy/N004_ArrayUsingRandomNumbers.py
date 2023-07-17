import numpy as np

# rand()
arr = np.random.rand(4)
print(arr)

print()

arr = np.random.rand(2, 5)  # for multi Dimensional
print(arr)

print()

# randn()
randnArr = np.random.randn(5)
print(randnArr)

print()

randnArr = np.random.randn(3, 4)  # for multi dimensional
print(randnArr)

print()

# ranf()
var = np.random.ranf(4)
print(var)

print()

var = np.random.ranf((4, 5))  # for multi dimensional
print(var)

print()

# randint()
randintArr = np.random.randint(5, 20, 5)  # (min,max(excluded),total_values)
print(randintArr)

print()

randintArr = np.random.randint(1, 20, (2, 4))  # for multi dimensional, for 3D ->(2,4,4)
print(randintArr)
