import numpy as np
var = np.array([1,2,3,4,5])
x = np.where(var == 5)
print(x)

y = np.where((var%2) == 0)
print(y)