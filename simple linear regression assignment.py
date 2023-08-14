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

#==============================================================
'''
2) Salary_hike -> Build a prediction model for Salary_hike
'''

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



