#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from category_encoders import MEstimateEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer
import datetime as dt
from sklearn import metrics
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import VotingClassifier
from scipy.cluster.vq import kmeans, vq


train = pd.read_csv('C:/Users/DELL/Desktop/HackerEarth_Predicting_Churn_Rate/train.csv')
test = pd.read_csv('C:/Users/DELL/Desktop/HackerEarth_Predicting_Churn_Rate/test.csv')

t = test['customer_id']
train['churn_risk_score'] = train['churn_risk_score'].abs()
test['churn_risk_score'] = -1
data = pd.concat([train,test],axis = 0)


# Identifying hidden NAN values
data['joined_through_referral'] = data['joined_through_referral'].replace('?',np.nan)
data['medium_of_operation'] = data['medium_of_operation'].replace('?',np.nan)
data['avg_frequency_login_days'] = data['avg_frequency_login_days'].replace('Error',np.nan)


# Applying mod to negative values
data['days_since_last_login'] = data['days_since_last_login'].abs()
data['avg_time_spent'] = data['avg_time_spent'].abs()
data['points_in_wallet'] = data['points_in_wallet'].abs()

data['avg_frequency_login_days'] = data['avg_frequency_login_days'].astype(float)
data['avg_frequency_login_days'] = data['avg_frequency_login_days'].abs()


# Extracting dates
data['joining_date'] = pd.to_datetime(data['joining_date'],infer_datetime_format = True)
data['last_visit_time'] = pd.to_datetime(data['last_visit_time'],format = '%H:%M:%S').dt.hour
a = data['joining_date'].max()
data['days_since_joined'] = a - data['joining_date']
data['days_since_joined'] = data['days_since_joined'].dt.days 
data = data.drop(['Name','customer_id','security_no','referral_id','joining_date'],axis=1)


# Removing missing values
data.fillna({'region_category':data['region_category'].mode()[0],'joined_through_referral':data['joined_through_referral'].mode()[0],'preferred_offer_types':data['preferred_offer_types'].mode()[0],'medium_of_operation':data['medium_of_operation'].mode()[0]}, inplace=True)
data.fillna({'points_in_wallet':data['points_in_wallet'].mean(),'avg_frequency_login_days':data['avg_frequency_login_days'].mean()},inplace = True)

categorical = data.select_dtypes('object').columns
for col in categorical:
    data[col] = data[col].astype(str)


#Feature Engineering
data['days_since_last_login'] = data['days_since_last_login'].replace(999,30)
data['avg_login_days'] = data['days_since_joined'] / data['avg_frequency_login_days']

data['x'] = data['avg_transaction_value'] / data['points_in_wallet']
data['y'] = data['x'] * data['points_in_wallet']**0.5 

data['days_since_last_login'] = data['days_since_last_login'].astype(float)
data['age'] = data['age'].astype(float)
data['days_since_joined'] = data['days_since_joined'].astype(float)
data['avg_frequency_login_days'] = data['avg_frequency_login_days'].astype(float)
data['last_visit_time'] = data['last_visit_time'].astype(float)


cluster_centers,_ = kmeans(data['days_since_last_login'],5)
data['dsll'], _ = vq(data['days_since_last_login'],cluster_centers)
data['dsll'] = data['dsll'].astype(str)

cluster_centers,_ = kmeans(data['age'],4)
data['age1'], _ = vq(data['age'],cluster_centers)
data['ag1'] = data['age1'].astype(str)

cluster_centers,_ = kmeans(data['days_since_joined'],4)
data['dsj'], _ = vq(data['days_since_joined'],cluster_centers)
data['dsj'] = data['dsj'].astype(str)

cluster_centers,_ = kmeans(data['avg_frequency_login_days'],3)
data['ald'], _ = vq(data['avg_frequency_login_days'],cluster_centers)
data['ald'] = data['ald'].astype(str)

cluster_centers,_ = kmeans(data['last_visit_time'],4)
data['lvt'], _ = vq(data['last_visit_time'],cluster_centers)
data['lvt'] = data['lvt'].astype(str)



# Normalizing data
col_to_normalize = ['avg_time_spent']
for i in col_to_normalize:
    data[i] = np.log(data[i])
    data[i] = data[i].round(1)


# Encoding
categorical = list(data.select_dtypes('object').columns)
le = LabelEncoder()
for col in categorical:
    data[col] = le.fit_transform(data[[col]])


# Splitting into train and test
test = data[data['churn_risk_score']== -1]
train = data[data['churn_risk_score']!= -1]
test = test.drop(['churn_risk_score'],axis=1)

y = train['churn_risk_score']
x = train.drop(['churn_risk_score'],axis=1)

# Target Encoding
encoder = MEstimateEncoder(return_df = True,m = 3,random_state = 42)
encoder.fit(x,y)
x = encoder.transform(x)
test = encoder.transform(test)

def score(y_true, y_pred):
    score = 100 * metrics.f1_score(y_true, y_pred, average="macro")
    return score

xgbc = XGBClassifier(verbose = 0,n_jobs = -1,max_depth = 5,n_estimators = 100, learning_rate = 0.09,random_state = 42)
cbc = CatBoostClassifier(verbose=0,max_depth = 5,n_estimators = 500, learning_rate = 0.04,random_state = 42)


#Final Submission
est = [('cbc',cbc),('xgbc',xgbc)]
model = VotingClassifier(estimators = est,voting = 'soft')
model.fit(x,y)
pred = model.predict(test)

df = pd.DataFrame()
df['customer_id']=t 
df['churn_risk_score'] = pred

df[['customer_id','churn_risk_score']].to_csv('submission40.csv',index=False)
