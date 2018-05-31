import numpy as np
import matplotlib.pyplot as plt

# load iris data

columns=["sepal.length","sepal.width","petal.length","petal.width"]
name={'Setosa':0, 'Versicolor':1, 'Virginica':2}

X=np.loadtxt('iris.csv', delimiter=',', skiprows=1, converters={4: lambda x: name[x.decode().strip('"')]})
X1=X[X[:,-1]==0]
X2=X[X[:,-1]==1]
X3=X[X[:,-1]==2]

# text, annotate

plt.scatter(X[:,0],X[:,1],c=X[:,-1],s=X[:,2]*100, alpha=0.2)
plt.text(6,3.2,'sepal width\n/\nsepal length',fontsize=30,alpha=0.5,ha='center',va='center')
plt.annotate('Setosa',xy=(6,4),xytext=(6.5,4.2),arrowprops=dict(facecolor='black'))
plt.show()