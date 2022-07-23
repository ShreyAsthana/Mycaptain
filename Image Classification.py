#!/usr/bin/env python
# coding: utf-8

# In[26]:


#importing depedencies
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')


# In[28]:


# using pandas to read the database stored in the same folder
data = pd.read_csv("/Users/shrey/Downloads/archive (6)/mnist_train.csv")


# In[29]:


data.head()


# In[30]:


a = data.iloc[9,1:].values


# In[31]:


a = a.reshape(28,28).astype('uint8') #astype takes the array and casts it to the given datatype (here datatype is uint) 
plt.imshow(a)


# In[32]:


df_x = data.iloc[:,1:]
df_y= data.iloc[:,0]


# In[33]:


x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size = 0.2, random_state=4)


# In[34]:


x_train.head()


# In[35]:


y_train.head()


# In[36]:


rf= RandomForestClassifier(n_estimators=100)


# In[37]:


rf.fit(x_train, y_train)


# In[38]:


pred = rf.predict(x_test)


# In[39]:


pred


# In[40]:


# check prediction accuracy
s = y_test.values

# calculate number of correctly predicted values
count = 0
for i in range(len(pred)):
    if pred[i] == s[i]:
        count = count +1


# In[41]:


count


# In[42]:


# total values that the prediction code was run on
len(pred)


# In[43]:


# accuracy value
11626/12000


# In[ ]:




