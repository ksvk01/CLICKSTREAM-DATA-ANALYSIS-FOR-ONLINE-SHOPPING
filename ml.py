#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv("C:/Users/Vaishnavi/Downloads/archive (1)/e-shop clothing 2008.csv", delimiter=";")


# In[3]:


df


# In[4]:


from sklearn import preprocessing
  
# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()
  
# Encode labels in column 'species'.
df['page 2 (clothing model)']= label_encoder.fit_transform(df['page 2 (clothing model)'])
  
df['page 2 (clothing model)'].unique()


# In[5]:


x = df[['page 1 (main category)', 'country','page 2 (clothing model)','month','price 2']]
y = df['price']


# In[6]:


# importing train_test_split from sklearn

# splitting the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)


# In[7]:


# importing module
from sklearn.linear_model import LinearRegression
# creating an object of LinearRegression class
LR = LinearRegression()
# fitting the training data
LR.fit(x_train,y_train)


# In[8]:


y_prediction =  LR.predict(x_test)
y_prediction


# In[9]:


# importing r2_score module
from sklearn.metrics import r2_score
# predicting the accuracy score
score=r2_score(y_test,y_prediction)
print('r2 score is',score)


# In[ ]:


sns.pairplot(df)
plt.show()


# In[10]:


df.dropna()


# In[12]:


df.shape


# In[13]:


df=df.dropna()


# In[14]:


df.shape


# In[15]:


sns.pairplot(df)
plt.show()


# In[19]:


cols=df.columns
corr_mat=np.corrcoef(df[cols].values.T)


# In[20]:


hm=sns.heatmap(corr_mat, annot=True)


# In[17]:


df.columns


# In[ ]:




