# plot_test.py

import numpy as np
import matplotlib.pyplot as plt
data1=np.random.random(10)*10
print(data1)

data2=list(range(10))
print(data2)

plt.plot(data1, label='numpy')
plt.plot(data2, ':', label='list')
plt.legend()
plt.show()