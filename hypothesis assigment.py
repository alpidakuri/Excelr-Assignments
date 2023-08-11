# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 15:31:44 2023

@author: Admin
"""

import pandas as pd
df=pd.read_csv("C:/Users/Admin/Downloads/Cutlets.csv")
df
df['Unit A'].mean()
df['Unit B'].mean()

from scipy import stats
zcalc,pval=stats.ttest_ind(df['Unit A'],df['Unit B'])
print('Z calculated value is',zcalc.round(4))
print('p-vale is',pval,round(4))

alpha=0.05
if(pval<alpha):
    print("ho is rejected and h1 is accepted")
else:
    print("h1 is rejected and h0 is accepted")

'''
hence we are proved that there is no difference between the cutlets.
hence we proved that h1 is rejected and h0 is accepted 
'''

#=============================================================

import pandas as pd
import numpy as np
df=pd.read_csv("C:/Users/Admin/Downloads/LabTAT.csv")
df

df.shape
df.head()
df.tail()

'''
By using anova one way_test 
h0:There is no difference in average TAT among the different laboratories.
h1:There is difference in average TAT amoung the different laboratories.
'''
'By using Anova F_oneway test'
lab_1=df.iloc[:,0]
lab_2=df.iloc[:,1]
lab_3=df.iloc[:,2]
lab_4=df.iloc[:,3]


from scipy import stats
from scipy.stats import f_oneway
statics,p_value=stats.f_oneway(lab_1,lab_2,lab_3,lab_4)

if p_value<0.05:
    print("h0 is rejected and h1 is acceped")
else:
    print("h1 is rejected and h1 is accepted ")
    
'''
p_value= 2.1156708949992414e-57
here we got ho is rejected and h1 is accepted.
The p value is also less than 0.05.By this we concluded that there is 
difference in average TAT amoung the different laboratories.
'''
#======================================================


import numpy as np
import pandas as pd
df=pd.read_csv("C:/Users/Admin/Downloads/BuyerRatio (1).csv")
df

df.head()
df.tail()
df.shape

df_updated=df.iloc[:,1:]
df_updated

#creating a array with the update data set
np.array(df_updated)


import scipy.stats as stats

#here using chi-contingency 
stats.chi2_contingency(df_updated)

#chisqure_value:1.595945538661058,
pvalue=0.6603094907091882
#DregeeofFreedom:3

if pvalue < 0.05:
    print("ho is rejected and h1 is accepted")
else:
    print("h1 is rejected and ho is accepeted ")
'''
Here pvalue is less then alpha. so, h1 is rejected and ho is accepeted 
'''
##=======================================================================


 
import pandas as pd
import numpy as np
df = pd.read_csv("C:/Users/Admin/Downloads/Costomer+OrderForm (1).csv")
df


df.shape
df.head()
df.tail()

df.describe()
'''
       Phillippines   Indonesia       Malta       India
count           300         300         300         300
unique            2           2           2           2
top      Error Free  Error Free  Error Free  Error Free
freq            271         267         269         280
'''
df.info()

df_updated = np.array([[271,267,269,280],[29,33,31,20]])

import scipy.stats as stats

#here using chi-contingency 
stats.chi2_contingency(df_updated)

#chisqure_value:3.858960685820355,
pvalue=0.2771020991233135
#DregeeofFreedom:3

if pvalue < 0.05:
    print("ho is rejected and h1 is accepted")
else:
    print("h1 is rejected and ho is accepeted ")
#p-value > 0.05,h1 is rejected and ho is accepeted 
# Hence the Customer order form defective % doesn't varies across the given countries

#============================================================

