import numpy as np
import matplotlib.pyplot as plt

# load iris data

columns=["sepal.length","sepal.width","petal.length","petal.width"]
name={'Setosa':0, 'Versicolor':1, 'Virginica':2}

X=np.loadtxt('iris.csv', delimiter=',', skiprows=1, converters={4: lambda x: name[x.strip('"')]}, encoding='utf-8')
X1=X[X[:,-1]==0]
X2=X[X[:,-1]==1]
X3=X[X[:,-1]==2]

# histogram

plt.title('IRIS - petal.length')
plt.hist(X1[:,2],bins=20,alpha=0.5,label='Setosa')
plt.hist(X2[:,2],bins=20,alpha=0.5,label='Versicolor')
plt.hist(X3[:,2],bins=20,alpha=0.5,label='Virginica')
plt.xlabel('petal.length')
plt.ylabel('count')
plt.legend()
plt.show()