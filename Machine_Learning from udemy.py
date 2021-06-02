#!/usr/bin/env python
# coding: utf-8

# # Data_preprocessing
# 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


dataset=pd.read_csv('Salary_Data.csv')


# In[3]:


dataset


# In[10]:


x=dataset['YearsExperience'].values


# In[11]:


x


# In[12]:


y=dataset['Salary'].values


# In[13]:


y


# #  Missing_values

# In[16]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy='median')
imputer.fit(x[:,:-1])
x[:,:-1]=imputer.transform(x[:,:-1])


# In[ ]:


x


# # ColumnTransfer

# 
# 

# In[68]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[3])],remainder='passthrough')
x=np.array(ct.fit_transform(x))


# In[ ]:





# 
# 

# In[17]:


x


# # unlabelled data formating
# 

# In[ ]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y=le.fit_transform(y)


# In[ ]:


y


# # Training_the_Data

# In[22]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)


# In[23]:


print(x_train)


# In[24]:


print(x_test)


# In[25]:


print(y_test)


# In[26]:


print(y_train)


# # Scaling
# 

# In[ ]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train[:,3:]=sc.fit_transform(x_train[:,3:])
x_test[:,3:]=sc.transform(x_test[:,3:])


# In[ ]:


print(x_train)


# In[ ]:


print(x_test)


# In[ ]:


print(y_train)


# In[ ]:


print(y_test)


# # SimpleLinear_Regression

# In[27]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 


# In[28]:


data = pd.read_csv('Salary_Data.csv')


# In[29]:


data


# In[30]:


x=data.iloc[:,:-1].values


# In[31]:


x


# In[32]:


y=data.iloc[:,-1].values


# In[33]:


y


# In[34]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)


# In[35]:


x_train


# In[36]:


x_test


# In[37]:


y_train


# In[38]:


y_test


# In[39]:


from sklearn.linear_model import LinearRegression
Regressor=LinearRegression()
Regressor.fit(x_train,y_train)


# In[50]:


y_pred = Regressor.predict(x_train)


# In[51]:


y_pred


# In[49]:


print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# In[ ]:


print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# In[53]:


plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, Regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# In[55]:


plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, Regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# In[57]:


Regressor.predict([[10.5]]) 


# In[ ]:





# # MUltiple_Linear_Regression

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


dataa=pd.read_csv('50_Startups.csv')


# In[3]:


dataa


# In[4]:


x=dataa.iloc[:,:-1].values


# In[5]:


x


# In[6]:


y=dataa.iloc[:,-1].values


# In[7]:


y


# In[ ]:





# In[8]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy='median')
imputer.fit(x[:,0:3])
x[:,0:3] = imputer.transform(x[:,0:3])


# In[9]:


x


# In[10]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[3])],remainder='passthrough')
x=np.array(ct.fit_transform(x))


# In[ ]:





# In[ ]:





# In[11]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)


# In[12]:


x_train


# In[14]:


y_train


# In[13]:


x_test


# In[15]:


y_test


# In[ ]:





# In[16]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)


# In[17]:


y_pred = regressor.predict(x_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# In[18]:


y_pred = regressor.predict(x_test)


# In[19]:


y_pred.reshape(-1,1)


# In[20]:


y_test.reshape(-1,1)


# In[ ]:





# # Polynomial Regression

# In[13]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[14]:


datas=pd.read_csv('Position_Salaries.csv')


# In[15]:


datas


# In[16]:


x=datas.iloc[:,1:-1].values


# In[17]:


x


# In[18]:


y=datas.iloc[:,-1].values


# In[19]:


y


# In[ ]:





# In[31]:


from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(x,y)


# In[32]:


from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
x_poly=poly_reg.fit_transform(x)
lin_reg2=LinearRegression()
lin_reg2.fit(x_poly,y)


# In[ ]:





# In[33]:


plt.scatter(x, y,color='red')
plt.plot(x, lin_reg2.predict(poly_reg.fit_transform(x)),color='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# In[34]:


x_grid = np.arange(min(x_train), max(x_train), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_grid, lin_reg2.predict(poly_reg.fit_transform(x_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# In[38]:


lin_reg.predict([[5.3]])


# In[45]:



lin_reg2.predict(poly_reg.fit_transform([[5.3]]))               


# In[39]:


66029.0


# # lin_reg2.coef_

# In[ ]:


lin_reg2.predict(poly_reg.fit_transform([[5.3]]))


# In[ ]:





# In[47]:


lin_reg2.coef_


# # Support_vector_Regression

# In[49]:


import pandas as pd
import numpy as  np
import matplotlib.pyplot as plt


# In[50]:


daa=pd.read_csv('Position_Salaries.csv')


# In[51]:


daa


# In[52]:


x=daa.iloc[:,1:-1].values


# In[53]:


x


# In[54]:


y=daa.iloc[:,-1].values


# In[55]:


y=y.reshape(len(y),1)


# In[56]:


y


# In[57]:


from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
sc_y=StandardScaler()
x=sc_x.fit_transform(x)
y=sc_y.fit_transform(y)


# In[58]:


x


# In[59]:


y


# In[64]:


from sklearn.svm import SVR
regressor=SVR(kernel='linear')
regressor.fit(x,y)


# In[65]:


sc_y.inverse_transform(regressor.predict(sc_x.transform([[6]])))


# # Regression model_selection in python

# In[35]:


#multiple Linear_Regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[36]:


dataset=pd.read_csv('Dataa.csv')


# In[37]:


dataset.head()


# In[38]:


x=dataset.iloc[:,:-1]


# In[39]:


x


# In[40]:


y=dataset['PE'].values


# In[41]:


y


# In[73]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[42]:


x_train


# In[43]:


y_test


# In[44]:


x_test


# In[45]:





from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(x_train,y_train)


# In[46]:


y_pred = lin_reg.predict(x_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# In[ ]:





# In[47]:


from sklearn.metrics import r2_score
r2_score(y_test, y_pred)


# In[ ]:





# In[54]:


#PolynomialRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(x_train)
regressor = LinearRegression()
regressor.fit(x_poly, y_train)


# In[55]:


y_pred = regressor.predict(poly_reg.transform(x_test))
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# In[56]:


from sklearn.metrics import r2_score
r2_score(y_test, y_pred)


# In[57]:


#DecisionTreeRegression
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(x_train, y_train)


# In[58]:


y_pred = regressor.predict(x_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# In[59]:


from sklearn.metrics import r2_score
r2_score(y_test, y_pred)


# In[60]:


#RandomForestRegression
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(x_train, y_train)


# In[61]:


y_pred = regressor.predict(x_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# In[62]:


from sklearn.metrics import r2_score
r2_score(y_test, y_pred)


# In[ ]:





# In[63]:


#Randforest_Regression
y = dataset.iloc[:, -1].values


# In[64]:


y = y.reshape(len(y),1)


# In[65]:


y


# In[66]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


# In[67]:


y_test


# In[68]:


from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x_train = sc_x.fit_transform(x_train)
y_train = sc_y.fit_transform(y_train)


# In[69]:


x_train


# In[70]:


from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(x_train, y_train)


# In[36]:


y_pred = sc_y.inverse_transform(regressor.predict(sc_x.transform(x_test)))
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# In[37]:


from sklearn.metrics import r2_score
r2_score(y_test, y_pred)


# # Classification
# 

# In[24]:


#LogisticRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[25]:


dat=pd.read_csv('Social_Network_Ads.csv')


# In[26]:


dat


# In[27]:


x=dat.drop(['Purchased'],axis=1).values


# In[28]:


x


# In[29]:


y=dat['Purchased'].values


# In[30]:


y


# In[31]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)


# In[32]:


x_train


# In[33]:


y_test


# In[34]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[35]:


x_train


# In[36]:


x_test


# In[37]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(x_train, y_train)


# In[40]:


print(classifier.predict(sc.transform([[19,54631]])))


# In[17]:


y_pred = classifier.predict(x_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# In[18]:


from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_test,y_pred)
print(cm)
accuracy_score(y_test,y_pred)


# In[20]:


from matplotlib.colors import ListedColormap
x_set, y_set = sc.inverse_transform(x_train), y_train
x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 10, stop = x_set[:, 0].max() + 10, step = 0.25),
                     np.arange(start = x_set[:, 1].min() - 1000, stop = x_set[:, 1].max() + 1000, step = 0.25))
plt.contourf(x1, x2, classifier.predict(sc.transform(np.array([x1.ravel(), x2.ravel()]).T)).reshape(x1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Random Forest Classification (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


# In[ ]:


from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(X_test), y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


# # Classification model selection

# In[74]:


#LogisticRegression
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[75]:


dataset = pd.read_csv('NData.csv')


# In[76]:


dataset


# In[77]:


x=dataset.drop(['Class'],axis=1).values


# In[78]:


x


# In[79]:


y=dataset['Class'].values


# In[80]:


y


# In[81]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)


# In[82]:


x_test


# In[83]:


y_test


# In[84]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)


# In[85]:


x_test


# In[86]:


y_test


# In[87]:


from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(x_train,y_train)


# In[ ]:





# In[89]:


from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)


# In[91]:


#k_nearest_neighbor 
from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
classifier.fit(x_train,y_train)


# In[92]:


from sklearn.metrics import confusion_matrix,accuracy_score
y_pred=classifier.predict(x_test)
cm=confusion_matrix(y_test,y_pred)
print(cm)
accuracy_score(y_test,y_pred)


# In[110]:


from sklearn.svm import SVC 
classifier=SVC(kernel='linear',random_state=0)
classifier.fit(x_train,y_train)


# In[111]:


from sklearn.metrics import confusion_matrix,accuracy_score
y_pred=classifier.predict(x_test)
cm=confusion_matrix(y_test,y_pred)
print(cm)
accuracy_score(y_test,y_pred)


# In[117]:


from sklearn.svm import SVC 
classifier=SVC(kernel='rbf',random_state=0)
classifier.fit(x_train,y_train)


# In[118]:


from sklearn.metrics import confusion_matrix,accuracy_score
y_pred=classifier.predict(x_test)
cm=confusion_matrix(y_test,y_pred)
print(cm)
accuracy_score(y_test,y_pred)


# In[97]:


#naive bayes
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(x_train,y_train)


# In[98]:


from sklearn.metrics import confusion_matrix,accuracy_score
y_pred=classifier.predict(x_test)
cm=confusion_matrix(y_test,y_pred)
print(cm)
accuracy_score(y_test,y_pred)


# In[124]:


from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion='entropy',random_state=0)
classifier.fit(x_train,y_train)


# In[125]:


from sklearn.metrics import confusion_matrix,accuracy_score
y_pred=classifier.predict(x_test)
cm=confusion_matrix(y_test,y_pred)
print(cm)
accuracy_score(y_test,y_pred)


# In[128]:


from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
classifier.fit(x_train,y_train)


# In[129]:


from sklearn.metrics import confusion_matrix,accuracy_score
y_pred=classifier.predict(x_test)
cm=confusion_matrix(y_test,y_pred)
print(cm)
accuracy_score(y_test,y_pred)


# # Clustering 
# 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


dataset = pd.read_csv('Mall_Customers.csv')


# In[13]:


dataset.info


# In[4]:


x = dataset.iloc[:,3:].values


# In[5]:


x


# In[6]:


from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[37]:


wcss


# In[7]:


kmeans


# In[8]:


i


# In[ ]:





# In[9]:


kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(x)


# In[10]:


kmeans


# In[12]:


y_kmeans.size


# In[15]:


plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(x[y_kmeans == 3, 0], x[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(x[y_kmeans == 4, 0], x[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()


# In[16]:


kmeans.cluster_centers_[:, 0]


# In[17]:


kmeans.cluster_centers_[:, 1]


# In[1]:


def fun1():
    print("quiz")
def fun2(var):
    for i in range(var):
        fun1()
fun2(4)


# In[ ]:




