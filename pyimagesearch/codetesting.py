import numpy as np

a = np.arange(9).reshape((3, 3))
print(a)
# [[0 1 2]
#  [3 4 5]
#  [6 7 8]]

x=np.where(a < 4)
print(x)