import matplotlib.pyplot as plt
import numpy as np
from random import gauss


val = [1000,]
N = 1000
for i in range(N):
    val.append(gauss(val[-1], np.sqrt(val[-1])))

plt.plot(val)
plt.show()