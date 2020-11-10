import numpy as np
x = np.array([range(1, 101), range(311, 411), range(100)]).transpose()
y = np.array([range(101, 201), range(711, 811), range(100)])

print(x.shape)  # (3, 100)
print(y.shape)  # (100, 3)
