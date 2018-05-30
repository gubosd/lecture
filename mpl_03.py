import numpy as np
import matplotlib.pyplot as plt

data=[ [np.sin(i**2+j**2) for j in np.arange(-2,2,0.05)]
	for i in np.arange(-2,2,0.05) ]
	
plt.imshow(data)
plt.colorbar()
plt.show()