import numpy as np
import matplotlib.pyplot as plt

# load iris data

columns=["sepal.length","sepal.width","petal.length","petal.width"]
name={'Setosa':0, 'Versicolor':1, 'Virginica':2}

X=np.loadtxt('iris.csv', delimiter=',', skiprows=1, converters={4: lambda x: name[x.decode().strip('"')]})
X1=X[X[:,-1]==0]
X2=X[X[:,-1]==1]
X3=X[X[:,-1]==2]

# subplot2

fig,axes = plt.subplots(2,2,sharex=True,sharey=True)
fig.suptitle('IRIS')
for col in range(4):
	ax=axes[col//2,col%2]
	ax.hist(X1[:,col],bins=20,alpha=0.5,label='Setosa')
	ax.hist(X2[:,col],bins=20,alpha=0.5,label='Versicolor')
	ax.hist(X3[:,col],bins=20,alpha=0.5,label='Virginica')
	ax.set_xlabel(columns[col])
	ax.set_ylabel('count')
	ax.legend()
plt.show()