#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import datasets 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
iris=datasets.load_iris()
type(iris)


# In[2]:


iris.keys()


# In[3]:


iris.data.shape


# In[4]:


iris.target_names


# In[5]:


iris.feature_names


# In[6]:


x=iris.data


# In[7]:


x


# In[8]:


y=iris.target


# In[9]:


y


# In[10]:


df=pd.DataFrame(x,columns=iris.feature_names)


# In[11]:


df


# In[12]:


pd.plotting.scatter_matrix(df,c=y,figsize=[8,8],s=150,marker='CD')


# In[13]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=6)
knn.fit(iris['data'],iris['target'])


# In[14]:


x_new=np.random.rand(3,4)
x_new


# In[15]:


prediction=knn.predict(x_new)


# In[16]:


prediction


# In[17]:


x_new=np.array([[5.6,2.8,3.9,1.1],
               [5.7,2.6,3.8,1.3],
               [4.7,3.2,1.3,0.2]])


# In[18]:


prediction=knn.predict(x_new)


# In[19]:


prediction


# In[20]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=27,stratify=y)
knn=KNeighborsClassifier(n_neighbors=8)
knn.fit(x_train,y_train)
y_pred=knn.predict(x_test)


# In[21]:


x_train


# In[22]:


knn


# In[23]:


y_pred


# In[24]:


x_test


# In[25]:


knn.score(x_test,y_test)


# In[32]:


data=pd.read_csv('dataa.csv')


# In[33]:


data


# In[35]:


x=data.drop('PE',axis=1).values


# In[36]:


x


# In[37]:


y=data['PE'].values


# In[38]:


y


# In[39]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[40]:


x_test


# In[23]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train,y_train)
y_pred=knn.predict(x_test)


# In[24]:


y_pred


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




