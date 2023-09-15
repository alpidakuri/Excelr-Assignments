# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:37:35 2023

@author: Admin
"""

import warnings
warnings.filterwarnings('ignore')


import pandas as pd
import numpy as np
df=pd.read_csv('C:/Users/Admin/Downloads/bank-full (1).csv',sep=';')
df

#Data Analysis
df.head()
df.tail()
#Data Exploration
#Descriptive Statistics
df.describe()
df.info()
#Missing Values
df.isnull().sum()
#Exploratory Data Analysis

table=pd.crosstab(df.job,df.y).plot(kind='bar')

#Apply Standardscalar for X and Labelencoder for y
from sklearn.preprocessing import StandardScaler
SS = StandardScaler()

for column in df.columns:
    if df[column].dtype=='object':
        continue
    df[[column]]=SS.fit_transform(df[[column]])
df.info()    


from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
for column in df.columns:
    if df[column].dtype == np.number:
        continue
    df[column]=LE.fit_transform(df[column])
df.info()
df

#split the variables X and Y
X=df.iloc[:,1:16]
Y=df['y']
Y

from sklearn.linear_model import LogisticRegression
LR=LogisticRegression()
LR.fit(X,Y)
LR.intercept_ #array([-2.80986694])
LR.coef_
'''
array([[ 0.00885352,  0.1543967 ,  0.18787237, -0.35698156,  0.06143338,
        -1.07350824, -0.71226156, -0.63496741, -0.04329876,  0.03890511,
         1.01529071, -0.40553976,  0.35963805,  0.20683525,  0.22503899]])
'''

#model predictions
Y_pred=LR.predict(X)
Y_pred

#Metrics
from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(Y,Y_pred)
cm
ac1=accuracy_score(Y,Y_pred)
print("Accuracy score:",(ac1*100).round(3))
'''
cm=array([[39137,   785],
       [ 4146,  1143]], dtype=int64)
Accuracy score: 89.093
'''

from sklearn.metrics import recall_score,precision_score,f1_score
print("Sensitivity score:",(recall_score(Y,Y_pred)*100).round(3))
print("precision score:",(precision_score(Y,Y_pred)*100).round(3))
print("f1 score:",(f1_score(Y,Y_pred)*100).round(3))
'''
Sensitivity score: 21.611
precision score: 59.284
f1 score: 31.675
'''

#specificity
TN=cm[0,0]
FP=cm[1,0]
TNR=TN/TN+FP
print("Specificity:",(TNR)*100,round(3)) #Specificity: 414700.0 3

#Probabilities
LR.predict_proba(X)
LR.predict_proba(X)[:,0] #1-proba
LR.predict_proba(X)[:,1] #exact proba

#ROC curve
from sklearn.metrics import roc_curve,roc_auc_score
FPR,TNR,NULL=roc_curve(Y,LR.predict_proba(X)[:,1])

#plot curve
import matplotlib.pyplot as plt
plt.scatter(FPR,TNR)
plt.plot(FPR,TNR,color="Red")
plt.Xlable("False positive rate")
plt.Ylabel("True positive rate")
plt.show()
auc=roc_auc_score(Y,LR.predict_proba(X)[:.1])
print("Area under curve score:",(auc*100).round(3))

#Data Partition
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y)
X_train.shape
X_test.shape
Y_train.shape
Y_test.shape

#model fitting
from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(X_train,Y_train)

#model predictions
Y_pred_train=logreg.predict(X_train)
Y_pred_test=logreg.predict(X_test)

#metrics
from sklearn.metrics import accuracy_score
ac1=accuracy_score(Y_train,Y_pred_train)
print("Training accuracy score:",(ac1*100).round(3))
ac2=accuracy_score(Y_test,Y_pred_test)
print("Testing accuracy score:",(ac2*100).round(3))

#cross validation
#K-Fold validation
from sklearn.model_selection import KFold
Kf=KFold(5)
Training_auc=[]
Test_auc=[]
for train_index,test_index in Kf.split(X):
    X_train,X_test=X.loc[train_index],X.iloc[test_index]
    Y_train,Y_test=Y.iloc[train_index],Y.iloc[test_index]
    LR.fit(X_train,Y_train)
    Y_train_pred=LR.predict(X_train)
    Y_test_pred=LR.predict(X_test)

Training_auc.append(accuracy_score(Y_train,Y_train_pred))
Test_auc.append(accuracy_score(Y_test,Y_test_pred))
print('training mean squared error:',np.mean(Training_auc).round(3))    
print('test mean squared error:',np.mean(Test_auc).round(3)) 

#Shrinking Methods
#Lasso Regression
from sklearn.linear_model import Lasso
LS=Lasso(alpha=1)
LS.fit(X,Y)
d1=pd.DataFrame(list(X))
d2=pd.DataFrame(LS.coef_)
df1=pd.concat([d1,d2],axis=1)
df1.columns=['names','alpha1']
df1
''' By using Lasso Regression all the coefficents becomes 0,
    so it is not possible to drop all variables,Hence verifiy with Ridge Regression'''
    
    
    
#Ridge regression    
from sklearn.linear_model import Ridge
RR=Ridge(alpha=1)
RR.fit(X,Y)
d1=pd.DataFrame(list(X))
d2=pd.DataFrame(RR.coef_)
df1=pd.concat([d1,d2],axis=1)
df1.columns=['names','alpha1']
df1
'''By using Ridge Regression all the coefficents near to 0,
    so it is not possible to drop all variables,Hence Finilizing
    this model With all the variables'''
 