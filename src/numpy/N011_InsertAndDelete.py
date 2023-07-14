import numpy as np
var = np.array([1,2,3,4,5])
var = np.insert(var,2,40)
print(var)
var = np.insert(var,(0,5),40)
print(var)

var1 = np.array([[1,2,3,4],[5,6,7,8]])
var1 = np.insert(var1,2,6,axis=0)
print(var1)

var = np.delete(var,2)
print(var)