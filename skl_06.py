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

# Support Vector Machine (SVC)

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,stratify=y,random_state=1)
model=SVC()  #  kernel='rbf', C=1.0, gamma='auto'
model.fit(X_train, y_train)

pred_y = model.predict(X_test)
acc=accuracy_score(y_test,pred_y)
print('accuracy :',acc)