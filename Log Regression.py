#!/usr/bin/env python
# coding: utf-8

# In[134]:


import pandas as pd
from sklearn.linear_model import LogisticRegression


# In[135]:


bank=pd.read_csv('C:/Users/prate/Downloads/Assignment/logR/bank-full.csv',delimiter=';')
bank.head()


# In[136]:


bank.isnull().sum()


# In[6]:


bank.shape


# In[137]:


bank.drop(["day","contact","default","campaign","pdays","previous","poutcome","month"],axis=1,inplace=True)


# In[138]:


bank.columns


# In[124]:


bank.tail()


# In[139]:


bank1=pd.get_dummies(bank,columns=['job','marital','education','housing','loan','y'])


# In[140]:


bank1.head()


# In[127]:


bank1.describe


# In[145]:


X = df = bank1.loc[ : , bank1.columns != 'y_yes']
  
Y = bank1.iloc[:,27]


# In[117]:


Y.head()


# In[152]:


X.drop(["y_no"],inplace=True,axis=1)


# In[154]:


#Logistic regression and fit the model
classifier = LogisticRegression()
classifier.fit(X,Y)


# In[155]:


#Predict for X dataset
y_pred = classifier.predict(X)


# In[156]:


y_pred_df= pd.DataFrame({'actual': Y,
                         'predicted_prob': classifier.predict(X)})


# In[157]:


y_pred_df


# In[158]:


# Confusion Matrix for the model accuracy
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Y,y_pred)
print (confusion_matrix)


# In[69]:


(39238+964)/(39238+684+4325+964)


# In[160]:


#Classification report
from sklearn.metrics import classification_report
print(classification_report(Y,y_pred))


# In[161]:


#ROC
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

fpr, tpr, thresholds = roc_curve(Y, classifier.predict_proba (X)[:,1])

auc = roc_auc_score(Y, y_pred)

import matplotlib.pyplot as plt
plt.plot(fpr, tpr, color='red', label='logit model ( area  = %0.2f)'%auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
plt.ylabel('True Positive Rate')


# In[162]:


auc


# In[ ]:




