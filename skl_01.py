import numpy as np
import matplotlib.pyplot as plt

# load iris data

columns=["sepal.length","sepal.width","petal.length","petal.width"]
name={'Setosa':0, 'Versicolor':1, 'Virginica':2}

data=np.loadtxt('iris.csv', delimiter=',', skiprows=1, converters={4: lambda x: name[x.strip('"')]}, encoding='utf-8')

# data for machine-learning

X=data[:,:-1]  # (150,4)
y=data[:,-1]   # (150,)

X1=X[y==0]     # (50,4) Setosa
X2=X[y==1]     # (50,4) Versicolor
X3=X[y==2]     # (50,4) Virginica

# feature relations

fig=plt.figure()
fig.suptitle('Setosa')
count=0
for i in range(4):
	for j in range(i+1,4):
		count+=1
		plt.subplot(2,3,count)
		plt.scatter(X1[:,i],X1[:,j])
		plt.xlabel(columns[i])
		plt.ylabel(columns[j])
plt.show()