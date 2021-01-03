#!/usr/bin/env python
# coding: utf-8

# In[63]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


# In[64]:


pwd


# In[65]:


cd PycharmProjects


# In[66]:


cd pythonProject


# In[67]:


cd salary prediction


# In[68]:


cd salary-prediction-master


# In[69]:


pwd


# In[48]:


dataset = pd.read_csv('salary.csv')


# In[49]:


dataset


# In[50]:


x=dataset.iloc[:,:1].values


# In[51]:


y=dataset.iloc[:,1:].values


# In[52]:


fig=plt.figure()
ax=fig.add_axes([0,0,1,1])
ax.scatter(x,y,color='r')


# In[53]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[54]:


from sklearn.linear_model import LinearRegression


# In[61]:


regressor=LinearRegression()


# In[62]:


regressor.fit(x_train,y_train)


# In[70]:


y_pred=regressor.predict(x_test)


# In[71]:


y_pred


# In[72]:


y_test


# In[73]:


plt.scatter(x,y,color='r')
plt.plot(x,regressor,predict(x),color='blue')


# In[74]:


from sklearn.preprocessing import polynomialFeatures


# In[75]:


poly=polynomialFeatures(degree=2)
x_poly=poly.fit_transform(x)


# In[76]:


regressor.fit(x_poly,y)


# In[77]:


plt.scatter(x,y,color='r')
plt.plot(x,regressor,predict(poly.fit_transform(x),color='blue')


# In[ ]:




