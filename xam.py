#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data=pd.read_csv(r"C:\Users\admin\Downloads\diabetes.csv")


# In[3]:


data.head()


# In[4]:


data.isna().sum()


# In[5]:


y=data['Outcome']


# In[6]:


data.replace(0,np.nan,inplace=True)


# In[7]:


data.isna().sum()


# In[8]:


plt.hist(data['Pregnancies'])
plt.show()


# In[9]:


plt.hist(data['Glucose'])
plt.show()


# In[10]:


plt.hist(data['BloodPressure'])
plt.show()


# In[11]:


plt.hist(data['SkinThickness'])
plt.show()


# In[12]:


plt.hist(data['Insulin'])
plt.show()


# In[13]:


plt.hist(data['BMI'])
plt.show()


# In[14]:


plt.hist(data['Outcome'])
plt.show()


# In[15]:


data['Pregnancies']=data['Pregnancies'].fillna(data['Pregnancies'].median())
data['Glucose']=data['Glucose'].fillna(data['Glucose'].mean())
data['BloodPressure']=data['BloodPressure'].fillna(data['BloodPressure'].mean())
data['SkinThickness']=data['SkinThickness'].fillna(data['SkinThickness'].median())
data['Insulin']=data['Insulin'].fillna(data['Insulin'].median())
data['BMI']=data['BMI'].fillna(data['BMI'].median())


# In[16]:


data.isna().sum()


# In[ ]:





# In[17]:


data.columns


# In[18]:


plt.boxplot(data['Pregnancies'])


# In[19]:


plt.boxplot(data['Glucose'])
plt.show()


# In[20]:


plt.boxplot(data['BloodPressure'])


# In[21]:


plt.boxplot(data['SkinThickness'])


# In[22]:


plt.boxplot(data['Insulin'])


# In[23]:


plt.boxplot(data['BMI'])


# In[24]:


plt.boxplot(data['DiabetesPedigreeFunction'])


# In[25]:


plt.boxplot(data['Age'])


# In[26]:


data=data.drop(['Outcome'],axis=1)


# In[29]:


x=data[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age']]


# In[30]:


x.head()


# In[31]:


y.head()


# In[32]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# # logisticRegrssion

# In[33]:


from sklearn.linear_model import LogisticRegression


# In[34]:


lr= LogisticRegression()


# In[35]:


lr.fit(x_train,y_train)


# In[36]:


from sklearn.metrics import accuracy_score


# In[39]:


log_ac=accuracy_score(y_test,y_pred)


# In[40]:


log_ac


# # confusiom matrix

# In[41]:


from sklearn.metrics import confusion_matrix


# In[42]:


confusion_matrix(y_test,y_pred)


# # knn

# In[43]:


from sklearn.neighbors import KNeighborsClassifier


# In[44]:


m=[]
n=np.arange(3,15)
for K in n:
    clf=KNeighborsClassifier(n_neighbors=K)
    clf.fit(x_train,y_train)
    pred=clf.predict(x_test)
    acc=accuracy_score(y_test,pred)
    m.append(acc)


# In[45]:


plt.plot(n,m,'o-')
plt.xlabel('K values')
plt.ylabel('accuracy')
plt.grid()


# In[47]:


clf=KNeighborsClassifier(n_neighbors=9)
clf.fit(x_train,y_train)


# In[48]:


y_pred_knn=clf.predict(x_test)
acc_knn=accuracy_score(y_test,y_pred_knn)
print(acc_knn)


# # svc

# In[53]:


from sklearn.svm import SVC
sv=SVC()


# In[54]:


sv.fit(x_train,y_train)


# In[55]:


y_pred_sv=sv.predict(x_test)


# In[56]:


accuracy_score(y_test,y_pred_sv)


# # decision tree

# In[57]:


from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()


# In[58]:


dt.fit(x_train,y_train)


# In[59]:


y_pred_dt=dt.predict(x_test)


# In[61]:


dt_ac=accuracy_score(y_test,y_pred_dt)


# In[62]:


dt_ac


# # random forest

# In[63]:


from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()


# In[64]:


rf.fit(x_train,y_train)


# In[65]:


y_pred_rf=rf.predict(x_test)


# In[66]:


accuracy_score(y_test,y_pred_rf)


# # precision score

# In[71]:


from sklearn.metrics import precision_score


# In[72]:


pr=precision_score(y_test,y_pred)


# In[75]:


pr


# In[ ]:





# 

# In[ ]:





# In[ ]:




