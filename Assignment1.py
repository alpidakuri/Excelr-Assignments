# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 19:09:59 2023

@author: Admin
"""
#Q7) Calculate Mean, Median, Mode, Variance, Standard Deviation, Range &     comment about the values / draw inferences, for the given dataset
#	For Points,Score,Weigh>
#Find Mean, Median, Mode, Variance, Standard Deviation, and Range and also Comment about the values/ Draw some inferences.
import pandas as pd
df=pd.read_csv("C:/Users/Admin/Downloads/Cars.csv")
df
df.shape

df.mean()
df.median()
df.std()
df.var()
df.mode()
df.head()

#histograph
df.dtypes
df["HP"].hist()
df["MPG"].hist()
df["VOL"].hist()
df["SP"].hist()
df["WT"].hist()

#scatter plot
import matplotlib.pyplot as plt
plt.scatter()
#==============================================================

#Q3) Three Coins are tossed, find the probability that two heads and one tail are obtained?
from scipy.stats import binom
bi=binom(n=8,p=0.3)
bi.pmf(3)
#==================================================================

#Q4)  Two Dice are rolled, find the probability that sum is
#a)	Equal to 1
#b)	Less than or equal to 4
#c)	Sum is divisible by 2 and  3
#a)
from scipy.stats import binom
bi=binom(n=36,p=1)
bi.pmf(1)

#b)	Less than or equal to 4
from scipy.stats import binom
bi=binom(n=36,p=0.6)
bi.pmf(6)

#c)	Sum is divisible by 2 and  3
from scipy.stats import binom
bi=binom(n=36,p=1.1)
bi.pmf(1.1)

#Q5)A bag contains 2 red, 3 green and 2 blue balls. Two balls are drawn at random.What is the probability that none of the balls drawn is blue?
from scipy.stats import binom
bi=binom(n=21,p=1)
bi.pmf(0.3)





