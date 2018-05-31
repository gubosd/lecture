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

# train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,stratify=y,random_state=1)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
print(X_train[y_train==0].shape, X_train[y_train==1].shape, X_train[y_train==2].shape)
print(X_test[y_test==0].shape, X_test[y_test==1].shape, X_test[y_test==2].shape)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train,y_train) # for training : X_train, y_train

pred_y = model.predict(X_test) # for evaluation : X_test, y_test
acc=accuracy_score(y_test,pred_y)
print('accuracy :',acc)
