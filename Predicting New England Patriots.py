#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn import pairplot
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from statsmodels.graphics.correlation import plot_corr


# In[2]:


NE = pd.read_csv(r'C:\Users\mckinleyharlett\Desktop - Local\NewEngland.csv')
NE.head()


# In[3]:


NE['home_or_awayNE'] = NE['home_or_awayNE'].astype('float64')
NE['score'] = NE['score'].astype('float64')
NE.info()


# ### EDA

# In[4]:


pairplot(NE)


# In[5]:


corr = NE.corr()
corr


# In[6]:


NE = NE.fillna(NE.mean())


# In[7]:


# Creating Traning and Testing sets
X = pd.DataFrame(NE, columns = ['weather_temperature', 'home_or_awayNE'])
y = pd.DataFrame(NE, columns = ['score'])

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=1)


# In[8]:


# Building our model

#linear regression 
regression_model = LinearRegression()

regression_model.fit(X_train,y_train)

pred = regression_model.predict(X_test)


# In[9]:


print(regression_model.intercept_)


# In[10]:


print(regression_model.coef_)


# In[11]:


# R-Squared
test_set_r2 = r2_score(y_test, pred)
print(test_set_r2)


# In[12]:


test_set_rmse = np.sqrt(mean_squared_error(y_test,pred))

print(test_set_rmse)


# In[13]:


NE_results = y_test
NE_results['Predicted'] = pred.ravel()
NE_results['Residuals'] = abs(NE_results['score']) - abs(NE_results['Predicted'])


# In[14]:


print(NE_results)


# ##### Assumption 1: Linearity 

# In[15]:


fig = plt.figure(figsize=(10,7))
sns.residplot(x = "Predicted", y = "score", data = NE_results, color = 'red')
plt.title('Residuals for New England')
plt.xlabel('Predicted')
plt.ylabel('Residual')


# ##### Assumption 2: Homoscedasticity

# We can see from the above residual plot that there is a relative constant variance

# ###### Assumption 3 - Normally Distributed Residuals

# In[16]:


plt.subplots(figsize=(12,6))
plt.title('Distribution of Residuals for New England')
sns.distplot(NE_results['Residuals'])
plt.show()


# ###### Assumption 4 - Little to no multicollinearity 

# In[17]:


fig=plot_corr(corr,xnames=corr.columns)


# In[ ]:




