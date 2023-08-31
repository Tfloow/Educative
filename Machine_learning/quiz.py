import numpy as np

arr = np.arange(4).reshape((2,2))
print(arr)
somme = np.sum(arr, axis=1)
print(somme)