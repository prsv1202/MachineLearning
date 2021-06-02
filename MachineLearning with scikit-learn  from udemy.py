#!/usr/bin/env python
# coding: utf-8

# # Data_preprocessing

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


data=pd.read_csv('Pearsons Height Data.csv')


# In[3]:


x=data['Father'].values.reshape(-1,1)


# In[4]:


y=data['Son'].values


# # LInearRegression

# In[5]:


from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(x,y)


# In[6]:


plt.scatter(x,y,color='red')
plt.show()


# In[7]:


plt.scatter(x,y,color='red')
plt.plot(x,lin_reg.predict(x),color='blue')
plt.show()


# In[8]:


lin_reg.predict([[70.7]])


# # PolynomialFetures

# In[9]:


from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(degree=4)
x_poly=poly.fit_transform(x)
poly.fit(x_poly,y)


# In[10]:


lin_reg_poly=LinearRegression()
lin_reg_poly.fit(x_poly,y)


# In[11]:


plt.scatter(x,y,color='red')
plt.plot(x,lin_reg_poly.predict(poly.fit_transform(x)),color='blue',linewidth=4)
plt.show()


# In[12]:


lin_reg_poly.predict(poly.fit_transform([[70.7]]))


# # PolynomialFeature

# In[13]:


import pandas as pd


# In[14]:


dat=pd.read_csv('Admission_Predict.csv')


# In[15]:


dat.head()


# In[16]:


x=dat.iloc[:,0:-1].values


# In[17]:


x


# In[18]:


y=dat.iloc[:,-1].values


# In[59]:


y


# # Training_the_Data

# In[20]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)


# In[23]:


from sklearn.preprocessing import PolynomialFeatures
multi_poly=PolynomialFeatures(degree=2)
x_poly=multi_poly.fit_transform(x_train)
multi_poly.fit(x_poly, y_train)


# In[24]:


from sklearn.linear_model import LinearRegression
lin_reg_multi=LinearRegression()
lin_reg_multi.fit(x_poly,y_train)


# In[25]:


y_pred=lin_reg_multi.predict(multi_poly.fit_transform(x_test))


# In[26]:


from sklearn import metrics
print(metrics.mean_squared_error(y_pred,y_test))


# # HealthIsuue

# In[1]:


import pandas as pd


# In[2]:


daa=pd.read_csv('Data.csv')


# In[3]:


daa.tail()


# In[4]:


daa.shape


# In[5]:


x=daa.iloc[:,2:29].values


# In[6]:


x


# In[7]:


y=daa.iloc[:,1].values


# In[8]:


y


# # Training_the_Data

# In[9]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)


# In[10]:


x_train


# # StandardScaler

# In[20]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.fit_transform(x_test)


# # LogisticRegression

# In[21]:


from sklearn.linear_model import LogisticRegression
logistic_classifier=LogisticRegression()+
logistic_classifier.fit(x_train,y_train)


# # using_confusionmatrix

# In[31]:


y_preds=logistic_classifier.predict(x_test)
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,y_preds))


# # SVC

# In[32]:


from sklearn.svm import SVC
svm=SVC(kernel='rbf')
svm.fit(x_train,y_train)


# In[33]:


y_preds=svm.predict(x_test)
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,y_preds))


# # DecisionTree

# In[38]:


from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion='entropy')
dt.fit(x_train,y_train)


# In[39]:


y_preds=dt.predict(x_test)
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,y_preds))


# # RandomForestClassifier

# In[45]:


from sklearn.ensemble import RandomForestClassifier
forest=RandomForestClassifier(n_estimators=100,criterion='entropy')
forest.fit(x_train,y_train)


# In[46]:


y_preds=forest.predict(x_test)
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,y_preds))


# # Boosting  and Optimization

# In[12]:


import pandas as pd


# In[13]:


da=pd.read_csv('data.csv')


# In[14]:


da.head()


# In[15]:


x=da.iloc[:,2:29].values


# In[16]:


y=da.iloc[:,1].values


# In[17]:


y


# In[34]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[35]:


x_train


# In[36]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.fit_transform(x_test)


# In[37]:


x_train


# # Principal Component Analysis

# In[38]:


#together all dependent(x) into a (single feature) particular number
from sklearn.decomposition import PCA
pca=PCA(n_components=1)
x_train_scaled=pca.fit_transform(x_train)
print(x_train_scaled[:10])


# In[39]:


import matplotlib.pyplot as plt
plt.scatter(x_train_scaled,y_train)
plt.show()


# # gradient_Boosting

# In[40]:


from sklearn.ensemble import GradientBoostingClassifier
gradientboost=GradientBoostingClassifier()
gradientboost.fit(x_train,y_train)


# In[41]:


y_preds=gradientboost.predict(x_test)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,y_preds))


# # xgboost

# In[55]:


from xgboost import XGBClassifier
xgboost=XGBClassifier()
xgboost.fit(x_train,y_train)


# In[56]:


y_preds=xgboost.predict(x_test)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,y_preds))


# In[ ]:




