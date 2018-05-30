import numpy as np
import matplotlib.pyplot as plt

data1=np.random.normal(1,1,size=100)
data2=np.random.normal(3,1,size=100)

plt.plot(data1,'ro-')
plt.plot(data2,'bs--')
plt.title('TEST')
plt.legend(['data1','data2'])
plt.xlabel('time')
plt.ylabel('value')
plt.show()