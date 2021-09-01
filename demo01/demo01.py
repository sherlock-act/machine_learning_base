import numpy as np
from matplotlib import pyplot as plt
from scipy import sparse


eye = np.eye(4)
print("NumPy array:\n{}".format(eye))

sparse_matrix = sparse.csr_matrix(eye)
print("\nSciPy sparse CSR matrix:\n{}".format(sparse_matrix))

x = np.linspace(-10, 10, 100)
y = np.sin(x)
plt.plot(x, y)
plt.show()