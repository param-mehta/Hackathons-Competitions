#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations 
import numpy as np
from catboost import CatBoostRegressor
from sklearn.preprocessing import LabelEncoder


# In[2]:


train = pd.read_csv('C:/Users/DELL/Desktop/HackerEarth_Shipping_Cost/train.csv')
test = pd.read_csv('C:/Users/DELL/Desktop/HackerEarth_Shipping_Cost/test.csv')

t = test['Customer Id']

train['Cost'] = train['Cost'].abs()
test['Cost'] = -1

data1 = pd.concat([train,test],axis=0)
data1 = data1.drop(['Customer Id'],axis=1)


# In[3]:


d = combinations(['International','Express Shipment'],2)

for i, j in d:
     data1[i + '_' + j] = data1[i].astype(str) + '_' + data1[j].astype(str)


# In[4]:


d = combinations(['International','Express Shipment','Transport','Installation Included','Fragile'],2)

for i, j in d:
     data1[i + '_' + j] = data1[i].astype(str) + '_' + data1[j].astype(str)
        
d = combinations(['International','Express Shipment','Transport','Installation Included','Fragile'],3)
for i, j,k in d:
     data1[i + '_' + j + '_' + k ] = data1[i].astype(str) + '_' + data1[j].astype(str)+ '_'+ data1[k].astype(str)


# In[5]:


data1.fillna({'Artist Reputation':np.mean(data1['Artist Reputation'])              ,'Height':np.mean(data1['Height']),'Width':np.mean(data1['Width'])              ,'Weight':np.mean(data1['Weight']),'Material':'None'}, inplace=True)


dic = {"No_No":'Roadways',"No_Yes":'Airways',"Yes_No":'Waterways',"Yes_Yes":'Airways'}

data1.Transport = data1.Transport.fillna(data1['International_Express Shipment'].map(dic))

a = data1['Remote Location'].value_counts(normalize=True)

data1['Remote Location'] = data1['Remote Location'].fillna(pd.Series(np.random.choice(a.index, 
                                                       p=a.values, size=len(data1))))


# In[8]:


data1['Area'] = data1['Height'] * data1['Width']


data1['Area'] = data1['Area'].astype(int)

data1['Priceperrep'] =  data1['Price Of Sculpture'] / data1['Base Shipping Price'] 

data1['Priceperrep'] = data1['Priceperrep'].round(3)

data1['repperprice'] =  data1['Artist Reputation'] / data1['Price Of Sculpture'] 

data1['repperprice'] = data1['repperprice'].round(3)

data1['PriceperW'] =  data1['Base Shipping Price'] / data1['Weight'] 

data1['PriceperW'] = data1['PriceperW'].round(3)

data1['Price Of Sculpture'] = data1['Price Of Sculpture'] * data1['Price Of Sculpture']


data1 = data1.drop(['Artist Name','International','Express Shipment','Transport','Installation Included','Fragile'],axis=1)


# In[9]:


data1['Scheduled Date'] = pd.to_datetime(data1['Scheduled Date'],format = '%m/%d/%y')

data1['Delivery Date'] = pd.to_datetime(data1['Delivery Date'],format='%m/%d/%y')

data1['time_diff_days'] = (data1['Scheduled Date'] - data1['Delivery Date']).dt.days

data1['time_diff_months'] = (data1['Scheduled Date'].dt.month - data1['Delivery Date'].dt.month)


data1['time_diff_month'] = data1['Scheduled Date'].dt.month 

data1['time_diff_months'] =  data1['Delivery Date'].dt.month

data1.drop(['Scheduled Date','Delivery Date'],axis=1,inplace=True)


# In[10]:


data1['Ratio'] =  data1['PriceperW'] / data1['Priceperrep']

data1['Ratio'] = data1['Ratio'].round(2)

for i in ['Priceperrep','PriceperW','Ratio']:
    data1[i] = data1[i].astype(int)
    


# In[11]:


num = list(data1.select_dtypes(['int64','float64']).columns)
num.remove('Cost')


# In[15]:


# Specify which column to normalize
col_to_normalize = ['Weight','Price Of Sculpture','Height','Width','Artist Reputation']

# Log normalization
for i in col_to_normalize:
    # Add log normalized column
    data1[i] = np.log(data1[i])
    
    data1[i] = data1[i].round(1)
    # Drop the original column


# In[16]:


def state(cell):
    try :
        cell = cell.split(sep=', ')[1].split()[0]
    
    except :
        cell = cell.split(sep=' ')[1]
    
    return cell
    


# In[17]:


data1['Customer Location']=data1['Customer Location'].apply(state)


# In[18]:


cat = list(data1.select_dtypes('object').columns)


# In[19]:


le = LabelEncoder()

for i in cat:
    data1[i] = le.fit_transform(data1[[i]])


# In[20]:


test = data1[data1['Cost']== -1]

train = data1[data1['Cost']!= -1]


# In[21]:


y = train['Cost']
x = train.drop(['Cost'],axis=1)


# In[23]:


test = data1[data1['Cost']== -1]
train = data1[data1['Cost']!= -1]
test_l = test.drop(['Cost'],axis=1)


# In[24]:


# Specify which column to normalize
col_to_normalize = ['Cost']

# Log normalization
for i in col_to_normalize:
    # Add log normalized column
    train[i] = np.log(train[i])
    train[i] = train[i].round(1)
    # Drop the original column


# In[26]:


y = train['Cost']
x = train.drop(['Cost'],axis=1)


# In[36]:


params = {'n_estimators': 800, 'max_depth': 5, 'learning_rate': 0.16}
regressor = CatBoostRegressor(**params) 
regressor.fit(x, y)
y_pred_log = regressor.predict(test_l)

y_pred = [(np.exp(x)) for x in [i for i in y_pred_log]]

test['Customer Id']=t 
test['Cost'] = y_pred

test[['Customer Id','Cost']].to_csv('retry2.csv',index=False)

