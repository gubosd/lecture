import numpy as np
import matplotlib.pyplot as plt

# load iris data

columns=["sepal.length","sepal.width","petal.length","petal.width"]
name={'Setosa':0, 'Versicolor':1, 'Virginica':2}

X=np.loadtxt('iris.csv', delimiter=',', skiprows=1, converters={4: lambda x: name[x.decode().strip('"')]})
X1=X[X[:,-1]==0]
X2=X[X[:,-1]==1]
X3=X[X[:,-1]==2]

# scatter plot

plt.scatter(X[:,0], X[:,1], c=X[:,-1], s=X[:,2]*100, alpha=0.2)
plt.show() # c: color, s: size