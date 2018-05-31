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

# linear regression

from sklearn.linear_model import LinearRegression

model=LinearRegression()
model=model.fit(X1[:,0].reshape(-1,1), X1[:,1])
w=model.coef_
b=model.intercept_

plt.title('Setosa: sepal_length/sepal_width')
plt.scatter(X1[:,0],X1[:,1])
plt.xlabel(columns[0])
plt.ylabel(columns[1])
plt.plot([4,6],[4*w+b,6*w+b],'g:')
plt.text(4,4.3,'coef: %f\nintercept: %f' % (w,b),va='top',fontsize=15,color='b')
plt.show()

pred_y = model.predict([[0],[1],[2],[3]])
print(pred_y)