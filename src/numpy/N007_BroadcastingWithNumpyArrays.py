import numpy as np

var1 = np.array([[1], [2], [3]])
var2 = np.array([1, 2, 3])
"""
a = var1 + var2
when you will perform the above operation, we will get teh error as mentioned bellow
ValueError: operands could not be broadcast together with shapes (4,) (3,)

for broadcasting we have to see two condition
1-should have same dimension
2-if dimension is not same then check any one(row or column) should be 1 in both of the array
  and the final dimension of array will be the max(row, column) of two array
"""

print(var1 + var2)
# need to explore more



