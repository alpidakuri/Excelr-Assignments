# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 15:03:03 2023

@author: Admin
"""

'''
2) Salary_hike -> Build a prediction model for Salary_hike
'''
import numpy as np
import pandas as pd
df=pd.read_csv('C:/Users/Admin/Downloads/Salary_Data.csv')
df

x=df[['YearsExperience']]
y=df[['Salary']]


df.info()

df.boxplot()
df.describe()
'''
       YearsExperience         Salary
count        30.000000      30.000000
mean          5.313333   76003.000000
std           2.837888   27414.429785
min           1.100000   37731.000000
25%           3.200000   56720.750000
50%           4.700000   65237.000000
75%           7.700000  100544.750000
max          10.500000  122391.000000

'''
#scatter plot
import matplotlib.pyplot as plt
plt.scatter(x,y,color='pink')
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.show()

#Scikitlearn(model fitting)
from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(x,y)

LR.intercept_ #array([25792.20019867])
LR.coef_  #array([[9449.96232146]])

#model predicted values
y
y_pred=LR.predict(x)

#Regression line between model predicted values and original values                  
import matplotlib.pyplot as plt
plt.scatter(x,y,color='pink')
plt.scatter(x,y_pred,color='green')
plt.plot(x,y_pred,color='black')
plt.show()

#finding errors by using matrics
from sklearn.metrics import mean_squared_error,r2_score
mse=mean_squared_error(y,y_pred)
r2=r2_score(y,y_pred)
print('mean_squarred error:',mse.round(3))
print('root mean squarred error:',np.sqrt(mse).round(3))
print('R square:',r2.round(4))

'''
mean_squarred error: 31270951.722
root mean squarred error: 5592.044
R square: 0.957
'''

#Applying log tranformation to x

x_sq = np.sqrt(df[["YearsExperience"]])


#Data visulation
import matplotlib.pyplot as plt
plt.scatter(x_sq,y)
plt.xlabel("YearsExperience")
plt.ylabel("Salary")
plt.show()

#Model fitting
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(x_sq,y)
LR.intercept_   #array([-16055.76911696])
LR.coef_        #array([[41500.68058303]])

#Predectoin
df[["Salary"]]
salary_pred = LR.predict(x_sq)
y

#constructing regrassion line between model predicted values and original values
import matplotlib.pyplot as plt
plt.scatter(x_sq,y,color='red')
plt.scatter(x_sq,salary_pred,color='blue')
plt.plot(x_sq,salary_pred,color='black')
plt.xlabel('YearsExperience')
plt.ylabel('salary_hike')
plt.show()


#Finding Errors by using Metrics
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y,salary_pred)
R2= r2_score(y, salary_pred)
print("Mean_squared_error:", mse.round(3))
print("Root Mean Sqaure error:",np.sqrt(mse).round(3))
print("R square:", R2.round(3))

'''
Mean_squared_error: 50127755.617
Root Mean Sqaure error: 7080.096
R square: 0.931
'''
#=========================================

#Applying log transfromation to y

y_log = np.log(df[["Salary"]])

#Data visulation
import matplotlib.pyplot as plt
plt.scatter(x,y_log)
plt.xlabel("YearsExperience")
plt.ylabel("Salary")
plt.show()

#Model fitting
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(x,y_log)
LR.intercept_   #array([10.5074019])
LR.coef_        #array([[0.12545289]])

#Predectoin
df[["YearsExperience"]]
deli_pred = LR.predict(x)
y_log

#constructing regrassion line between model predicted values and original values
import matplotlib.pyplot as plt
plt.scatter(x,y_log,color='red')
plt.scatter(x,deli_pred,color='blue')
plt.plot(x,deli_pred,color='black')
plt.xlabel('YearsExperience')
plt.ylabel('salary_hike')
plt.show()


#Finding Errors by using Metrics
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_log,salary_pred)
R2= r2_score(y_log,salary_pred)
print("Mean_squared_error:", mse.round(3))
print("Root Mean Sqaure error:",np.sqrt(mse).round(3))
print("R square:", R2.round(3))
'''
Mean_squared_error: 6451110810.96
Root Mean Sqaure error: 80318.807
R square: -49068915764.984
'''
#==================================================================================
#Applying log transfromation to x and y
x_log = np.log(df[["YearsExperience"]])
y_log = np.log(df[["Salary"]])

#Data visulation
import matplotlib.pyplot as plt
plt.scatter(x_log,y_log)
plt.xlabel("YearsExperience")
plt.ylabel("salary_hike")
plt.show()

#Model fitting
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(x_log,y_log)
LR.intercept_   #array([10.32804318])
LR.coef_        #array([[0.56208883]])

#Predectoin
df[["YearsExperience"]]
salary_pred = LR.predict(x_log)
y_log

#constructing regrassion line between model predicted values and original values
import matplotlib.pyplot as plt
plt.scatter(x_log,y_log,color='red')
plt.scatter(x_log,salary_pred,color='blue')
plt.plot(x_log,salary_pred,color='black')
plt.xlabel('YearsExperience')
plt.ylabel('salary_hike')
plt.show()


#Finding Errors by using Metrics
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_log,salary_pred)
R2= r2_score(y_log, salary_pred)
print("Mean_squared_error:", mse.round(3))
print("Root Mean Sqaure error:",np.sqrt(mse).round(3))
print("R square:", R2.round(3))
"""Mean_squared_error: 0.012
Root Mean Sqaure error: 0.112
R square: 0.905"""

#==================================================================================
#Applying Sq Root Transformation of X
x_sq = np.sqrt(df[["YearsExperience"]])


#Data visulation
import matplotlib.pyplot as plt
plt.scatter(x_sq,y)
plt.xlabel("YearsExperience")
plt.ylabel("salary_hike")
plt.show()

#Model fitting
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(x_sq,y)
LR.intercept_   #array([-16055.76911696])
LR.coef_        #array([[41500.68058303]])

#Predectoin
df[["Salary"]]
salary_pred = LR.predict(x_sq)
y

#constructing regrassion line between model predicted values and original values
import matplotlib.pyplot as plt
plt.scatter(x_sq,y,color='red')
plt.scatter(x_sq,salary_pred,color='blue')
plt.plot(x_sq,salary_pred,color='black')
plt.xlabel('YearsExperience')
plt.ylabel('salary_hike')
plt.show()


#Finding Errors by using Metrics
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y,salary_pred)
R2= r2_score(y, salary_pred)
print("Mean_squared_error:", mse.round(3))
print("Root Mean Sqaure error:",np.sqrt(mse).round(3))
print("R square:", R2.round(3))
"""Mean_squared_error: 50127755.617
Root Mean Sqaure error: 7080.096
R square: 0.931"""

#==================================================================================
#Applying Sq Root Transformation of X x and y
x_sq = np.sqrt(df[["YearsExperience"]])
y_sq = np.sqrt(df[["Salary"]])

#Data visulation
import matplotlib.pyplot as plt
plt.scatter(x_sq,y_sq)
plt.xlabel("YearsExperience")
plt.ylabel("Salary")
plt.show()

#Model fitting
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(x_sq,y_sq)
LR.intercept_   #array([103.56803065])
LR.coef_        #array([[75.6269319]])

#Predectoin
df[["YearsExperience"]]
salry_pred = LR.predict(x_sq)
y_sq

#constructing regrassion line between model predicted values and original values
import matplotlib.pyplot as plt
plt.scatter(x_sq,y_sq,color='red')
plt.scatter(x_sq,salary_pred,color='blue')
plt.plot(x_sq,salary_pred,color='black')
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.show()



#Finding Errors by using Metrics
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_sq,salary_pred)
R2= r2_score(y_sq, salary_pred)
print("Mean_squared_error:", mse.round(3))
print("Root Mean Sqaure error:",np.sqrt(mse).round(3))
print("R square:", R2.round(3))
"""Mean_squared_error: 6409195034.906
Root Mean Sqaure error: 80057.448
R square: -2687836.779"""



