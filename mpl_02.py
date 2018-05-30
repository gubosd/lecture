import numpy as np
import matplotlib.pyplot as plt

data1=np.random.normal(1,1,size=1000)
data2=np.random.normal(3,1,size=1000)

plt.scatter(data1,data2)
plt.title('TEST2')
plt.xlabel('data1')
plt.ylabel('data2')
plt.show()