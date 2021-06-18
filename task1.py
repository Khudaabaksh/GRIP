#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np                 
import pandas as pd                
import matplotlib.pyplot as plt    
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import metrics
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression


# In[2]:


url = "http://bit.ly/w-data"
student_data = pd.read_csv(url)
print("Successfully imported data into console")


# In[3]:


student_data.head(25)


# In[4]:


student_data.plot(x = 'Hours',y = 'Scores', style='o')
plt.title('Hours vs Percentage')
plt.xlabel('The Hours Studied')
plt.ylabel('The Percentage Score')
plt.show()


# In[5]:


student_data.hist()


# In[6]:


X = student_data.iloc[:, :-1].values    
y = student_data.iloc[:, 1].values 
print(X,y)


# In[8]:


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=0)


# In[9]:


regressor = LinearRegression()    
regressor.fit(X_train, y_train)   
print("Training Completed !")


# In[11]:


line = regressor.coef_*X+regressor.intercept_  
plt.scatter(X, y)  
plt.plot(X, line);  
plt.show()


# In[12]:


print(X_test)   
y_pred = regressor.predict(X_test)


# In[16]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})    
df


# In[17]:


hours = [[9.25]]  
own_pred = regressor.predict(hours)  
print("Number of hours = {}".format(hours))  
print("Prediction Score = {}".format(own_pred[0]))


# In[18]:


print("Mean Absolute Error :",metrics.mean_absolute_error(y_test, y_pred))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




