#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# In[10]:


df = pd.read_csv('~/Capstone/data/airbnb-listings_cleaned.csv', low_memory=False)


# In[11]:


df.head(5)


# In[12]:


# Encode categorical features with mean price
categorical_cols = df.select_dtypes(include=['object']).columns
df_le = df.copy()
for col in categorical_cols:
    le = LabelEncoder()
    df_le[col] = df_le.groupby(col)['Price'].transform('mean')

# Convert datetime columns to integer format (seconds since epoch)
datetime_column = df.select_dtypes(include=['datetime64']).columns
for col in datetime_column:
    df_le[col] = df_le[col].astype('int64') // 10**9

# Compute the correlation matrix
corr_matrix = df_le.corr()

# Select the correlations of other variables with the target variable "Price"
price_corr = corr_matrix['Price'].abs().sort_values(ascending=False)

# Select top 30 correlations excluding the target variable itself
top_30_corr = price_corr[1:31]

# Plot the correlation bar chart
#plt.figure(figsize=(12, 8))
#top_30_corr.plot(kind='barh', color='lightcoral')
#plt.title('Top 30 Correlations with Price', fontsize=16)
#plt.xlabel('Correlation', fontsize=12)
#plt.ylabel('Variables', fontsize=12)
#plt.show()


# #### Feature Selection

# In[13]:


sf = ['Amenities', 'Street', 'Neighbourhood Cleansed', 'Host Name', 'Bedrooms', 'Accommodates', 'Room Type', 'Bathrooms']
X = df_le[sf]
y = df_le['Price']


# #### First Split

# In[14]:


X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.10, random_state=42)


# #### Second Split

# In[15]:


X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1111, random_state=42)


# #### Linear Regression Model Application and Evaluation

# In[16]:


lm_model = LinearRegression()


# #### K-Fold Cross Validation

# In[17]:


kf = KFold(n_splits=5, shuffle=True, random_state=42)


# In[21]:


print("Model 1 - Linear Regression")
cv_r = []
for fold, (train_i,val_i) in enumerate(kf.split(X_train),1):
    X_train_f, X_val_f = X_train.iloc[train_i], X_train.iloc[val_i]
    y_train_f, y_val_f = y_train.iloc[train_i], y_train.iloc[val_i]

    s = StandardScaler()
    X_train_s = s.fit_transform(X_train_f)
    X_val_s = s.transform(X_val_f)

    lm_model.fit(X_train_s,y_train_f)
    y_pred = lm_model.predict(X_val_s)

    mse = mean_squared_error(y_val_f, y_pred)
    mae = mean_absolute_error(y_val_f, y_pred)
    r2 = r2_score(y_val_f, y_pred)

    cv_r.append({'Fold': fold, 'MSE': mse, 'MAE': mae, 'R2': r2})
    
    print(f"Fold {fold} - MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")


# In[22]:


avg_mse = np.mean([fold['MSE'] for fold in cv_r])
avg_mae = np.mean([fold['MAE'] for fold in cv_r])
avg_r2 = np.mean([fold['R2'] for fold in cv_r])
print(f"\nAverage CV - MSE: {avg_mse:.4f}, MAE: {avg_mae:.4f}, R2: {avg_r2:.4f}")


# #### Train model on whole train set and evaluate on validation set

# In[23]:


s_2 = StandardScaler()
X_train_s = s_2.fit_transform(X_train)
X_val_s = s_2.transform(X_val)


# In[24]:


lm_model.fit(X_train_s,y_train)
y_pred_val = lm_model.predict(X_val_s)


# In[25]:


mse = mean_squared_error(y_val, y_pred_val)
mae = mean_absolute_error(y_val, y_pred_val)
r2 = r2_score(y_val, y_pred_val)
print(f"\nValidation Results - MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")


# #### Last Evaluation on test set

# In[26]:


X_test_s = s_2.transform(X_test)
y_pred_test = lm_model.predict(X_test_s)


# In[27]:


mse_t = mean_squared_error(y_test, y_pred_test)
mae_t = mean_absolute_error(y_test, y_pred_test)
r2_t = r2_score(y_test, y_pred_test)
print(f"\nTest Results - MSE: {mse_t:.4f}, MAE: {mae_t:.4f}, R2: {r2_t:.4f}")


# #### Random Forest Regression Model Application and Evaluation

# In[29]:


rf_model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)


# #### K-Fold Cross Validation

# In[30]:


kf = KFold(n_splits=5, shuffle=True, random_state=42)


# In[31]:


print("Model 2 - Random Forest Regression")
cv_r = []
for fold, (train_i,val_i) in enumerate(kf.split(X_train),1):
    X_train_f, X_val_f = X_train.iloc[train_i], X_train.iloc[val_i]
    y_train_f, y_val_f = y_train.iloc[train_i], y_train.iloc[val_i]

    s = StandardScaler()
    X_train_s = s.fit_transform(X_train_f)
    X_val_s = s.transform(X_val_f)

    rf_model.fit(X_train_s,y_train_f)
    y_pred = rf_model.predict(X_val_s)

    mse = mean_squared_error(y_val_f, y_pred)
    mae = mean_absolute_error(y_val_f, y_pred)
    r2 = r2_score(y_val_f, y_pred)

    cv_r.append({'Fold': fold, 'MSE': mse, 'MAE': mae, 'R2': r2})
    
    print(f"Fold {fold} - MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")


# In[32]:


avg_mse = np.mean([fold['MSE'] for fold in cv_r])
avg_mae = np.mean([fold['MAE'] for fold in cv_r])
avg_r2 = np.mean([fold['R2'] for fold in cv_r])
print(f"\nAverage CV - MSE: {avg_mse:.4f}, MAE: {avg_mae:.4f}, R2: {avg_r2:.4f}")


# #### Train model on whole train set and evaluate on validation set

# In[33]:


s_2 = StandardScaler()
X_train_s = s_2.fit_transform(X_train)
X_val_s = s_2.transform(X_val)


# In[34]:


rf_model.fit(X_train_s,y_train)
y_pred_val = rf_model.predict(X_val_s)


# In[35]:


mse = mean_squared_error(y_val, y_pred_val)
mae = mean_absolute_error(y_val, y_pred_val)
r2 = r2_score(y_val, y_pred_val)
print(f"\nValidation Results - MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")


# #### Last Evaluation on test set

# In[36]:


X_test_s = s_2.transform(X_test)
y_pred_test = rf_model.predict(X_test_s)


# In[37]:


mse_t = mean_squared_error(y_test, y_pred_test)
mae_t = mean_absolute_error(y_test, y_pred_test)
r2_t = r2_score(y_test, y_pred_test)
print(f"\nTest Results - MSE: {mse_t:.4f}, MAE: {mae_t:.4f}, R2: {r2_t:.4f}")


# #### Gradient Boosting Regression Model Application and Evaluation

# In[39]:


gb_model = GradientBoostingRegressor()


# #### K-Fold Cross Validation

# In[40]:


kf = KFold(n_splits=5, shuffle=True, random_state=42)


# In[41]:


print("Model 3 - Gradient Boosting Regression")
cv_r = []
for fold, (train_i,val_i) in enumerate(kf.split(X_train),1):
    X_train_f, X_val_f = X_train.iloc[train_i], X_train.iloc[val_i]
    y_train_f, y_val_f = y_train.iloc[train_i], y_train.iloc[val_i]

    s = StandardScaler()
    X_train_s = s.fit_transform(X_train_f)
    X_val_s = s.transform(X_val_f)

    gb_model.fit(X_train_s,y_train_f)
    y_pred = gb_model.predict(X_val_s)

    mse = mean_squared_error(y_val_f, y_pred)
    mae = mean_absolute_error(y_val_f, y_pred)
    r2 = r2_score(y_val_f, y_pred)

    cv_r.append({'Fold': fold, 'MSE': mse, 'MAE': mae, 'R2': r2})
    
    print(f"Fold {fold} - MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")


# In[43]:


avg_mse = np.mean([fold['MSE'] for fold in cv_r])
avg_mae = np.mean([fold['MAE'] for fold in cv_r])
avg_r2 = np.mean([fold['R2'] for fold in cv_r])
print(f"\nAverage CV - MSE: {avg_mse:.4f}, MAE: {avg_mae:.4f}, R2: {avg_r2:.4f}")


# #### Train model on whole train set and evaluate on validation set

# In[44]:


s_2 = StandardScaler()
X_train_s = s_2.fit_transform(X_train)
X_val_s = s_2.transform(X_val)


# In[45]:


gb_model.fit(X_train_s,y_train)
y_pred_val = gb_model.predict(X_val_s)


# In[46]:


mse = mean_squared_error(y_val, y_pred_val)
mae = mean_absolute_error(y_val, y_pred_val)
r2 = r2_score(y_val, y_pred_val)
print(f"\nValidation Results - MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")


# #### Last Evaluation on test set

# In[47]:


X_test_s = s_2.transform(X_test)
y_pred_test = gb_model.predict(X_test_s)


# In[48]:


mse_t = mean_squared_error(y_test, y_pred_test)
mae_t = mean_absolute_error(y_test, y_pred_test)
r2_t = r2_score(y_test, y_pred_test)
print(f"\nTest Results - MSE: {mse_t:.4f}, MAE: {mae_t:.4f}, R2: {r2_t:.4f}")


# #### K Nearest Neighbors Regression Model Application and Evaluation

# In[49]:


knn_model = KNeighborsRegressor()


# #### K-Fold Cross Validation

# In[50]:


kf = KFold(n_splits=5, shuffle=True, random_state=42)


# In[51]:


print("Model 4 - K-Nearest Neighbors Regression")
cv_r = []
for fold, (train_i,val_i) in enumerate(kf.split(X_train),1):
    X_train_f, X_val_f = X_train.iloc[train_i], X_train.iloc[val_i]
    y_train_f, y_val_f = y_train.iloc[train_i], y_train.iloc[val_i]

    s = StandardScaler()
    X_train_s = s.fit_transform(X_train_f)
    X_val_s = s.transform(X_val_f)

    knn_model.fit(X_train_s,y_train_f)
    y_pred = knn_model.predict(X_val_s)

    mse = mean_squared_error(y_val_f, y_pred)
    mae = mean_absolute_error(y_val_f, y_pred)
    r2 = r2_score(y_val_f, y_pred)

    cv_r.append({'Fold': fold, 'MSE': mse, 'MAE': mae, 'R2': r2})
    
    print(f"Fold {fold} - MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")


# In[52]:


avg_mse = np.mean([fold['MSE'] for fold in cv_r])
avg_mae = np.mean([fold['MAE'] for fold in cv_r])
avg_r2 = np.mean([fold['R2'] for fold in cv_r])
print(f"\nAverage CV - MSE: {avg_mse:.4f}, MAE: {avg_mae:.4f}, R2: {avg_r2:.4f}")


# #### Train model on whole train set and evaluate on validation set

# In[53]:


s_2 = StandardScaler()
X_train_s = s_2.fit_transform(X_train)
X_val_s = s_2.transform(X_val)


# In[54]:


knn_model.fit(X_train_s,y_train)
y_pred_val = knn_model.predict(X_val_s)


# In[55]:


mse = mean_squared_error(y_val, y_pred_val)
mae = mean_absolute_error(y_val, y_pred_val)
r2 = r2_score(y_val, y_pred_val)
print(f"\nValidation Results - MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")


# #### Last Evaluation on test set

# In[56]:


X_test_s = s_2.transform(X_test)
y_pred_test = knn_model.predict(X_test_s)


# In[57]:


mse_t = mean_squared_error(y_test, y_pred_test)
mae_t = mean_absolute_error(y_test, y_pred_test)
r2_t = r2_score(y_test, y_pred_test)
print(f"\nTest Results - MSE: {mse_t:.4f}, MAE: {mae_t:.4f}, R2: {r2_t:.4f}")


# In[ ]:




