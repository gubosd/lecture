import numpy as np
import matplotlib.pyplot as plt

# load iris data

columns=["sepal.length","sepal.width","petal.length","petal.width"]
name={'Setosa':0, 'Versicolor':1, 'Virginica':2}
X=np.loadtxt('iris.csv', delimiter=',', skiprows=1, converters={4: lambda x: name[x.decode().strip('"')]})
X1=X[X[:,-1]==0]
X2=X[X[:,-1]==1]
X3=X[X[:,-1]==2]

# plot

plt.title('IRIS - sepal.length')
plt.plot(X1[:,0],'ro-')
plt.plot(X2[:,0],'gs:')
plt.plot(X3[:,0],'b^--')
plt.legend(name.keys())
plt.xlabel('samples')
plt.ylabel('values')
plt.show()