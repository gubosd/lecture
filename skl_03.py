import numpy as np
import matplotlib.pyplot as plt

# load iris data

columns=["sepal.length","sepal.width","petal.length","petal.width"]
name={'Setosa':0, 'Versicolor':1, 'Virginica':2}

data=np.loadtxt('iris.csv', delimiter=',', skiprows=1, converters={4: lambda x: name[x.decode().strip('"')]})

# data for machine-learning

X=data[:,:-1]  # (150,4)
y=data[:,-1]   # (150,)

X1=X[y==0]     # (50,4) Setosa
X2=X[y==1]     # (50,4) Versicolor
X3=X[y==2]     # (50,4) Virginica

# scatter plot

fig=plt.figure()
fig.suptitle('Setosa - Versicolor')
plt.scatter(X1[:,0],X1[:,1],marker='o',color='r')
plt.scatter(X2[:,0],X2[:,1],marker='s',color='b')
plt.xlabel(columns[0])
plt.ylabel(columns[1])
plt.legend(['Setosa','Versicolor'])
plt.show()