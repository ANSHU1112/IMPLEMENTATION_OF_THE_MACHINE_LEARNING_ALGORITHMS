
# coding: utf-8

# In[1]:

#ASSIGNMENT - 1 LINEAR REGRESSION


# In[2]:

import pandas as pd
import matplotlib.pyplot as plt
colnames=['LENGTH', 'MILEAGE']
data = pd.read_csv('C:/Users/PRIYANSHU SHARMA/Desktop/PRIYANSHU/6 STUDY/6 SEMSTER/MACHINE LEARNING/LAB/mileagecar.csv', names=colnames, header=None)
data.head()


# In[3]:

from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression


# In[4]:

X = data[['LENGTH']].values
Y = data[['MILEAGE']].values


# In[5]:

regr = linear_model.LinearRegression()
regr.fit(X, Y)


# In[6]:

plt.scatter(X, Y,  color='black')
plt.plot(X, regr.predict(X), color='blue', linewidth=3)
plt.xticks(())
plt.yticks(())
plt.show()


# In[7]:

regr.coef_ #slope


# In[8]:

regr.intercept_ #intercept


# In[9]:

print('Slope: %.3f' % regr.coef_[0])
print('Intercept: %.3f' % regr.intercept_[0])


# In[10]:

def lin_regplot(X, y, model):
    plt.scatter(X, y, c='blue')
    plt.plot(X, model.predict(X), color='red')
    return None


# In[11]:

lin_regplot(X, Y, regr)
plt.xlabel('LENGTH')
plt.ylabel('MILEAGE')
plt.show()


# In[13]:

mileage_std = regr.predict(200)    #when MILEAGE = 200
print("MILEAGE: %.3f" %mileage_std)

