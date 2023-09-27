# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 16:54:22 2023

@author: Admin
"""

import pandas as pd
df = pd.read_csv("C:/Users/Admin/Downloads/Zoo.csv")
df

df.head()
df.tail()
df.describe()
#Missing Values
df.isnull().sum()
#Duplicated Values
df.duplicated().sum()
#columns
df.columns

Y=df['type']

X=df.iloc[:,1:17]
X

#data transformation for X
from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
SS_X = SS.fit_transform(X)
SS_X

# data Transformation for Y
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df["type"] = LE.fit_transform(df["type"])
Y = df["type"]
Y

#---------------------------------------------------
# Data partition
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(SS_X,Y)

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=0)
print("Shape of X_train: ",X_train.shape)
print("Shape of X_test: ", X_test.shape)
print("Shape of Y_train: ",Y_train.shape)
print("Shape of Y_test",Y_test.shape)

#=======================================================
# step3: model fitting
from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors=3)
KNN.fit(X_train,Y_train)

#Using gridsearchCV for finding best n_neighbors
n_neighbors=list(range(1,50))
parameters={'n_neighbors':n_neighbors}

from sklearn.model_selection import GridSearchCV
grid=GridSearchCV(estimator=KNN, param_grid=parameters)
grid.fit(X_train,Y_train)
print(grid.best_score_)
print(grid.best_params_)


#=======================================================
# step4: model predictions
Y_pred_train = KNN.predict(X_train)
Y_pred_test = KNN.predict(X_test)

#=======================================================
# step5: metrics
from sklearn.metrics import accuracy_score
acc1 = accuracy_score(Y_train,Y_pred_train)
print("Training Accuracy: ",acc1.round(2))
acc2 = accuracy_score(Y_test,Y_pred_test)
print("Test Accuracy: ",acc2.round(2))

