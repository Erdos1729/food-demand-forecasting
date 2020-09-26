#!/usr/bin/env python
# coding: utf-8

# In[6]:


#importing libraries
import pandas as pd
import numpy as np
# !pip install seaborn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# In[7]:


test_file = pd.read_csv('datasets/test_file.csv')
train_file = pd.read_csv('datasets/train_file.csv')
meal_file = pd.read_csv('datasets/meal_info.csv')
fullfilment_file = pd.read_csv('datasets/fulfilment_center_info.csv')


# In[ ]:


df=test_file.copy()
print(df.head())


# In[ ]:


df.rename(columns={"id,week,center_id,meal_id,checkout_price,base_price,emailer_for_promotion,homepage_featured":"new"},inplace=True)


# In[ ]:


# df['id'] = df.new.str.split(',').str[0]
# df['week'] = df.new.str.split(',').str[1]
# df['center_id'] = df.new.str.split(',').str[2]
# df['meal_id'] = df.new.str.split(',').str[3]
# df['checkout_price'] = df.new.str.split(',').str[4]
# df['base_price'] = df.new.str.split(',').str[5]
# df['emailer_for_promotion'] = df.new.str.split(',').str[6]
# df['homepage_featured'] = df.new.str.split(',').str[7]
# df = df.iloc[:,1:]


# In[ ]:


df['id']=df['id'].astype('int')
df['week']=df['week'].astype('int')
df['center_id']=df['center_id'].astype('int')
df['meal_id']=df['meal_id'].astype('int')
df['checkout_price']=df['checkout_price'].astype('float')
df['base_price']=df['base_price'].astype('float')

df['emailer_for_promotion']=df['emailer_for_promotion'].astype('int')
df['homepage_featured']=df['homepage_featured'].astype('int')
df.head()


# In[ ]:


train_df = pd.merge(train_file,fullfilment_file, on='center_id')
test_df= pd.merge(df,fullfilment_file, on='center_id')


# In[ ]:


train_df = pd.merge(train_df,meal_file, on='meal_id')
test_df = pd.merge(test_df,meal_file, on='meal_id')


# In[ ]:


train_df.tail()
test_df.head()


# In[ ]:


encoder=LabelEncoder()
encoder2=LabelEncoder()
encoder3=LabelEncoder()


# In[ ]:


main_data=train_df.copy()
main_data.head()


# In[ ]:


main_data['category']=encoder.fit_transform(main_data['category'])
main_data['center_type']=encoder2.fit_transform(main_data['center_type'])
main_data['cuisine']=encoder3.fit_transform(main_data['cuisine'])
main_data.head()


# In[ ]:


main_data1= main_data.drop(['id'], axis=1)
correlation = main_data1.corr(method='pearson')
columns = correlation.nlargest(8, 'num_orders').index
columns


# In[ ]:


sns.heatmap(correlation,annot=True)
plt.show()


# In[ ]:


features = columns.drop(['num_orders'])
main_data2 = main_data[features]
X = main_data2.values
y = main_data['num_orders'].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=0)


# In[ ]:


x_train.shape


# In[ ]:


test_data=test_df.copy()
test_data.head()


# In[ ]:


test_data['category']=encoder.fit_transform(test_data['category'])
test_data['center_type']=encoder2.fit_transform(test_data['center_type'])
test_data['cuisine']=encoder3.fit_transform(test_data['cuisine'])
test_data.head()


# In[ ]:


test_data1=test_data.copy()
test_data1.head()


# In[ ]:


test_data_final=test_data1.drop(columns=['id','center_id','meal_id','week','checkout_price','base_price','center_type'],axis=1)
test_data_final.head()


# In[ ]:


#Building Models


# In[ ]:


DTR = DecisionTreeRegressor()
DTR.fit(x_train, y_train)
y_pred = DTR.predict(x_test)


# In[ ]:


RMSEL=mean_squared_log_error(y_test, y_pred)
RMSEL=np.sqrt(RMSEL)
RMSEL


# In[ ]:


from xgboost import XGBRegressor
XG = XGBRegressor()
XG.fit(x_train, y_train)
y_pred = XG.predict(x_test)
y_pred[y_pred<0] = 0
RMSEL=mean_squared_log_error(y_test, y_pred)
RMSEL=np.sqrt(RMSEL)
RMSEL


# In[ ]:





# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor
GB = GradientBoostingRegressor()
GB.fit(x_train, y_train)
y_pred = GB.predict(x_test)
y_pred[y_pred<0] = 0
RMSEL=mean_squared_log_error(y_test, y_pred)
RMSEL=np.sqrt(RMSEL)
RMSEL


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
RF = RandomForestRegressor()
RF.fit(x_train, y_train)
y_pred = RF.predict(x_test)
y_pred[y_pred<0] = 0
RMSEL=mean_squared_log_error(y_test, y_pred)
RMSEL=np.sqrt(RMSEL)
RMSEL


# In[ ]:


y=train_file['num_orders']


# In[ ]:


from lightgbm import LGBMRegressor,plot_importance
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_log_error,mean_squared_error
from sklearn.model_selection import GridSearchCV


# In[ ]:


lgb_model=LGBMRegressor(importance_type='gain')
lgbm_params = {
    "n_estimators":[230,260],
    "num_leaves":[41,51],
    'min_child_samples':[40,45,50],
    'random_state':[2019]
  
}
lgb_model.set_params(**lgbm_params) #base model


# In[ ]:


lgb_grid=GridSearchCV(lgb_model,lgbm_params,cv=5,scoring='neg_mean_squared_error',n_jobs=8)


# In[ ]:


model=lgb_grid.fit(x_train,y_train)


# In[ ]:


lgb_estimate=model.best_estimator_
lgb_estimate


# In[ ]:


y_pred=model.predict(x_test)
y_pred


# In[ ]:





# In[ ]:


y_pred[y_pred<0] = 0
RMSEL=mean_squared_log_error(y_test, y_pred)
RMSEL=np.sqrt(RMSEL)
RMSEL


# In[ ]:


#by comparision the error value of differents models for training data, the lowest error value got from Random Forest Regressio Model
#prediction on test data by Random Forest Regressor


# In[ ]:


test_data_final.head()


# In[ ]:


main_data2.head()


# In[ ]:


pred_test_data= RF.predict(test_data_final)
pred_test_data[pred_test_data<0] = 0
submit = pd.DataFrame({
    'id' :test_data['id'],
    'num_orders' : pred_test_data
})


# In[ ]:


submit.head()


# In[ ]:


submit.to_csv("submission_RFR.csv", index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




