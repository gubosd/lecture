import numpy as np
import matplotlib.pyplot as plt

plt.title('TEST')
plt.plot(range(-10,10),[np.sin(i) for i in range(-10,10)],label='sin(x)')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.xticks(range(-10,10,2))
plt.axis('equal')
plt.grid()
plt.legend()
plt.show()