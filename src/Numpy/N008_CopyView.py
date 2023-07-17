import numpy as np

x = np.array([3, 5, 1, 2])
y = x.copy()
x.sort()
print("X ->", x)
print("Y ->", y)

print()

w = np.array([6, 8, 2, 4, 5])
z = w.view()
w.sort()
print("W ->",w)
print("Z ->",z)
