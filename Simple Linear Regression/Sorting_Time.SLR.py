# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 15:07:11 2023

@author: Admin
"""
'''

1) Delivery_time -> Predict delivery time using sorting time
'''

import pandas as pd
import numpy as np
df=pd.read_csv("C:/Users/Admin/Downloads/delivery_time.csv")
df

df.describe()
'''
      Delivery Time  Sorting Time
count      21.000000     21.000000
mean       16.790952      6.190476
std         5.074901      2.542028
min         8.000000      2.000000
25%        13.500000      4.000000
50%        17.830000      6.000000
75%        19.750000      8.000000
max        29.000000     10.000000
'''
#To find outliers we have to use box plot
df.boxplot()
#Here the delivery time has outliers


#splitting of X and Y
X=df[['Sorting Time']]
Y=df[['Delivery Time']]


#scatter plot(Data Visuvalization)
import matplotlib.pyplot as plt
plt.scatter(X,Y)
plt.xlabel('Sorting Time')
plt.ylabel('Delivery Time')
plt.show()

#correlation
df.corr()
'''
               Delivery Time  Sorting Time
Delivery Time       1.000000      0.825997
Sorting Time        0.825997      1.000000
'''
#scikitlearn(model fitting)

from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(X,Y)

LR.intercept_ #array([6.58273397])
 
LR.coef_  #array([[1.6490199]])

#model predicted values
Y
Y_pred=LR.predict(X)

#constructing regression line between model predicted values and original values.
import matplotlib.pyplot as plt
plt.scatter(X,Y,color='red')
plt.scatter(X,Y_pred,color='green')
plt.plot(X,Y_pred,color='black')
plt.xlabel('Sorting Time')
plt.ylabel('Delivery Time')
plt.show()

#Finding errors by using matrics
from sklearn.metrics import mean_squared_error,r2_score
mse=mean_squared_error(Y, Y_pred)
R2= r2_score(Y,Y_pred)
print("Mean Squarred Error:",mse.round(4))
print("Root Mean Squarred Error:",np.sqrt(mse).round(4))
print('R square:', R2.round(3))

'''
Mean Squarred Error: 7.7933
Root Mean Squarred Error: 2.7917
R square: 0.682
'''
#================================================================

#Using log tranformation
X_log=np.log(df[['Sorting Time']])

#Data visualization
import matplotlib.pyplot as plt
plt.scatter(X_log,Y,color='red')
plt.xlabel('Sorting Time')
plt.ylabel('Delivery Time')
plt.show()

#model fitting
from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(X_log,Y)

LR.intercept_ #array([1.15968351])
LR.coef_  #array([[9.04341346]])


#prediction
df_pred=LR.predict(X_log)
df_pred

#Constructing regression line between model predicted values and original values
import matplotlib.pyplot as plt
plt.scatter(X_log,Y,color='red')
plt.scatter(X_log,df_pred,color="green")
plt.plot(X_log,df_pred,color="black")
plt.xlabel('Sorting Time')
plt.ylabel('Delivery Time')
plt.show()


#Finding errors by using metrics
from sklearn.metrics import mean_squared_error,r2_score
mse=mean_squared_error(df_pred, Y)
R2= r2_score(df_pred,Y)
print("Mean Squarred Error:",mse.round(4))
print("Root Mean Squarred Error:",np.sqrt(mse).round(4))
print('R square:', R2.round(3))

'''
Mean Squarred Error: 7.4702
Root Mean Squarred Error: 2.7332
R square: 0.562
'''

#==========================================

#Applying log transformation to Y

Y_log=np.log(df[['Delivery Time']])

#Data Visualization
import matplotlib.pyplot as plt
plt.scatter(X,Y_log,color='red')
plt.xlabel('Sorting Time')
plt.ylabel('Delivery Time')
plt.show()

#model fittiing
from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(X,Y_log)

LR.intercept_ #array([2.12137185])
LR.coef_  #array([[0.1055516]])

#prediction
df_y_pred=LR.predict(X)
df_y_pred

#constructing regression line between model predicted values and original values
import matplotlib.pyplot as plt
plt.scatter(X,Y_log,color='red')
plt.scatter(X,df_y_pred,color="green")
plt.plot(X,df_y_pred,color="black")
plt.xlabel('Sorting Time')
plt.ylabel('Delivery Time')
plt.show()

#Finding errors by using metrics
from sklearn.metrics import mean_squared_error,r2_score
mse=mean_squared_error(Y_log,df_y_pred)
R2= r2_score(Y_log,df_y_pred)
print("Mean Squarred Error:",mse.round(4))
print("Root Mean Squarred Error:",np.sqrt(mse).round(4))
print('R square:', R2.round(3))

'''
Mean Squarred Error: 0.0279
Root Mean Squarred Error: 0.167
R square: 0.711
'''
#=================================================


X_log=np.log(df[['Sorting Time']])
Y_log=np.log(df[['Delivery Time']])

#Data Visualisation
import matplotlib.pyplot as plt
plt.scatter(X_log,Y_log,color='red')
plt.xlabel('Sorting Time')
plt.ylabel('Delivery Time')
plt.show()

#model fitting
from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(X_log,Y_log)

LR.intercept_ #array([1.74198709])
LR.coef_  # array([[0.59752233]])

#prediction
df_y_pred=LR.predict(X_log)
df_y_pred

#Constructing regression line between model predicted values and original values
import matplotlib.pyplot as plt
plt.scatter(X_log,Y_log,color='red')
plt.scatter(X_log,df_y_pred,color="green")
plt.plot(X_log,df_y_pred,color="black")
plt.xlabel('Sorting Time')
plt.ylabel('Delivery Time')
plt.show()


#Finding errors by using metrics
from sklearn.metrics import mean_squared_error,r2_score
mse=mean_squared_error(Y_log,df_y_pred)
R2= r2_score(Y_log,df_y_pred)
print("Mean Squarred Error:",mse.round(4))
print("Root Mean Squarred Error:",np.sqrt(mse).round(4))
print('R square:', R2.round(3))

'''
Mean Squarred Error: 0.022
Root Mean Squarred Error: 0.1482
R square: 0.772
'''
#==============================================

#Applying square root function for x

X_sq = np.sqrt(df[["Sorting Time"]])


#Data visulation
import matplotlib.pyplot as plt
plt.scatter(X_sq,Y)
plt.xlabel("Sorting Time")
plt.ylabel("Delivery Time")
plt.show()

#Model fitting
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X_sq,Y)
LR.intercept_   #array([-2.51883662])
LR.coef_        #array([[7.93659075]])

#Predectoin
df[["Sorting Time"]]
deli_pred = LR.predict(X_sq)


#constructing regrassion line between model predicted values and original values
import matplotlib.pyplot as plt
plt.scatter(X_sq,Y,color='red')
plt.scatter(X_sq,deli_pred,color='blue')
plt.plot(X_sq,deli_pred,color='black')
plt.xlabel('Sorting Time')
plt.ylabel('Delivery Time')
plt.show()


#Finding Errors by using Metrics
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(Y,deli_pred)
R2= r2_score(Y, deli_pred)
print("Mean_squared_error:", mse.round(3))
print("Root Mean Sqaure error:",np.sqrt(mse).round(3))
print("R square:", R2.round(3))
"""Mean_squared_error: 7.461
Root Mean Sqaure error: 2.732
R square: 0.696"""


#==================================================================================
#Applying Sq Root Transformation of X x and y
x_sq = np.sqrt(df[["Sorting Time"]])
y_sq = np.sqrt(df[["Delivery Time"]])

#Data visulation
import matplotlib.pyplot as plt
plt.scatter(x_sq,y_sq)
plt.xlabel("Sorting Time")
plt.ylabel("Delivery Time")
plt.show()

#Model fitting
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(x_sq,y_sq)
LR.intercept_   #array([1.61347867])
LR.coef_        #array([[1.00221688]])

#Predectoin
df[["Sorting Time"]]
deli_pred = LR.predict(x_sq)
y_sq

#constructing regrassion line between model predicted values and original values
import matplotlib.pyplot as plt
plt.scatter(x_sq,y_sq,color='red')
plt.scatter(x_sq,deli_pred,color='blue')
plt.plot(x_sq,deli_pred,color='black')
plt.xlabel('Sorting Time')
plt.ylabel('Delivery Time')
plt.show()


#Finding Errors by using Metrics
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_sq,deli_pred)
R2= r2_score(y_sq, deli_pred)
print("Mean_squared_error:", mse.round(3))
print("Root Mean Sqaure error:",np.sqrt(mse).round(3))
print("R square:", R2.round(3))


"""Mean_squared_error: 0.101
Root Mean Sqaure error: 0.318
R square: 0.729"""

