#!/usr/bin/env python
# coding: utf-8

import psycopg2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Connect to database 
conn = psycopg2.connect("dbname=credit_fraud user=postgres password=Zrnkbx8O")

# Get data from table
# Data obtained from https://www.kaggle.com/mishra5001/credit-card 
all_data = pd.read_sql_query('''select * from application''', conn)

all_data.head()
all_data.info()


# ### Visualisation: Python Pandas vs. SQL

# **Only 8% of the applications were fraudulent, hence the dataset is very unbalanced**

# Python
all_data.describe()

# SQL

def query(text, conn):
    return pd.read_sql_query(text, conn)

text1 = '''select count(target) from application where target=0;'''
print('Number of non-frauds: \n', query(text1, conn), '\n')
text2 = '''select count(target) from application where target=1;'''
print('Number of frauds: \n', query(text2, conn), '\n')
text3 = '''select count(target) from application'''
print('Percentage of frauds: \n', 100 * query(text2, conn)/query(text3, conn))
del text1, text2, text3


# **Deal with NULL values**

# drop columns with more than 40% NULL values

# Python
perc_null = all_data.isnull().sum()*100/len(all_data)
perc_null

null_cols = perc_null[perc_null>40]
print(len(null_cols), ' columns to be dropped')
null_cols

all_data = all_data.drop(null_cols.index,axis=1)

all_data.shape

# SQL

text1 = 'with new as (select '
for col in all_data.columns:
    text1 = text1 + '100 * (count(*) - count(' + col + ')) / count(*) as ' + col + ', '
text1 = text1[:-2] + ' from application)'

null_cols = []
for col in all_data.columns:
    text2 = text1 + ' select count(*)'
    text2 = text2 + ' from new where ' + col + ' > 40;'
    if query(text2, conn)['count'][0]>0:
        null_cols.append(col)

print(len(null_cols), ' columns to be dropped')


# Python<br>
# drop rows with more than 50% missing data

null_rows = all_data.isnull().sum(axis=1)/all_data.shape[1]
null_rows[null_rows>0.5]

# there are no such rows, so moving on
# check columns with few missing data; perform data imputation

low_missing = perc_null[(perc_null>0) & (perc_null<15)]
low_missing

# These cols are divided between those with 0-1% missing data and those with ~ 13% missing data<br>
# Those with 0-1% can be imputted with the column mode. Below we investigate those with ~ 13%

low_missing_cols = low_missing.index.tolist()
all_data[low_missing_cols].info()

# check columns with ~ 13% missing data

cols_13 = low_missing[low_missing>13].index.tolist()
all_data[cols_13].info()

print('\nUnique values of columns: \n')
for col in cols_13:
    print(col, ' ', all_data[col].unique())

# check the most common values for each column

for col in cols_13:
    print(all_data[col].value_counts(),'\n')

# For all columns, the most common value is by far 0 (column mode). We hence update the missing data with this value.

for col in cols_13:
    all_data.loc[all_data[col].isnull(),col] = 0

# check value_counts() for data with 0-1% missing values

cols_0 = low_missing[low_missing<1].index.tolist()
for col in cols_0:
    print(all_data[col].value_counts(),'\n')

for col in cols_0:
    all_data[col].fillna(all_data[col].mode()[0],inplace=True)

# check remaining columns with null values

perc_null = 100*all_data.isnull().sum()/all_data.shape[0]
perc_null[perc_null>0]

print(all_data['ext_source_3'].value_counts())
all_data['occupation_type'].value_counts()

# For missing data imputation, get the columns most correlated to the given column

# numerical column filled with median of column
all_data['ext_source_3'].fillna(all_data['ext_source_3'].median(),inplace=True)

# categorical column filled with mode of column
all_data['occupation_type'].fillna(all_data['occupation_type'].mode()[0],inplace=True)

# The dataset contains no more missing values

nvl = all_data.isnull().sum()
nvl[nvl>0]

# **Investigate imbalance of column data**

# within many columns, there is the same value for almost all rows, so these columns are not very useful

maybe_discarded = []
for col in all_data.columns:
    if all_data[col].value_counts().sort_values(ascending=False).iloc[0]/all_data.shape[0] > 0.95:
        maybe_discarded.append(col)

discarded = []
for col in maybe_discarded:
    if col.startswith('flag'):
        discarded.append(col)
        
all_data.drop(columns=discarded, inplace=True)

discarded

all_data.shape


# **Set column types**

all_data.info()

# convert 'object' type columns to 'category' type

for col in all_data.columns:
    if (all_data[col].dtype != 'int64') and (all_data[col].dtype != 'float64'):
        all_data[col] = all_data[col].astype('category')

# convert some numerical columns to 'category' type, since they have a limited no. of possible values

categorical_cols = ['target', 'name_contract_type','code_gender','flag_own_car','flag_own_realty',
                   'reg_city_not_work_city','live_city_not_work_city','reg_city_not_live_city','amt_req_credit_bureau_hour',
                   'name_education_type','name_housing_type','name_family_status','weekday_appr_process_start',
                   'name_type_suite','name_income_type','amt_req_credit_bureau_day','amt_req_credit_bureau_week',
                   'def_60_cnt_social_circle','def_30_cnt_social_circle','amt_req_credit_bureau_qrt','cnt_children',
                   'cnt_fam_members','occupation_type','hour_appr_process_start','amt_req_credit_bureau_mon',
                    'amt_req_credit_bureau_year','obs_60_cnt_social_circle','obs_30_cnt_social_circle','organization_type',
                   'flag_document_3','flag_document_6','flag_document_8','reg_region_not_live_region',
                    'reg_region_not_work_region','live_region_not_work_region','flag_email','flag_phone','flag_work_phone',
                   'flag_emp_phone']

for col in categorical_cols:
    all_data[col] = all_data[col].astype('category')

all_data.info()

# **Resolve negative values**

# The following 4 columns have negative values, which should be positive

day_cols = ['days_birth','days_employed','days_registration','days_id_publish']
all_data[day_cols].describe()

# Transform no. of days to no. of years and rename columns

all_data[day_cols] = all_data[day_cols].abs()
all_data[day_cols] = all_data[day_cols]/365
all_data[day_cols].describe()

rename_dict = {}
for col in day_cols:
    rename_dict[col] = 'years' + col[4:]

all_data.rename(columns = rename_dict,inplace=True)

'''
# **Correlation between variables**

# Python

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
df_numeric = all_data.select_dtypes(include = numerics)
df_numeric.head()


# In[56]:


corrs = df_numeric.corr() # correlation of all-to-all
plt.matshow(corrs)
plt.show()
corrs[np.abs(corrs>0.2) & (corrs<1)].stack().index # extract the column tuples with corr>0.2


# SQL

# correlation all-to-all would be too expensive

# **Investigate which variables are most correlated to the target and find their correlation**

# Python

# In[58]:


plt.hist(corrs['target'].drop(['target'],axis=0), bins=20)
plt.title('Histogram of target-to-all correlation')
plt.xlabel('Correlation')
plt.ylabel('No. of variables')

pos = corrs[corrs['target'] > 0.05].index.tolist()
neg = corrs[corrs['target'] < -0.1].index.tolist()
print('Variables with high positive correlation: \n', corrs[corrs['target'] > 0.05].index.tolist(), '\n')
print('Variables with high negative correlation: \n', corrs[corrs['target'] < -0.1].index.tolist(), '\n')


# SQL

# In[59]:


text = 'SELECT '
for var in pos:
    text = text + 'corr(target,' + var + ') ' + var + ','
text = text[:-1] + ' FROM application;'
df = query(text, conn)
df.index = ['target']
df


# **Visualise variables highly correlated with the target**

# Python

# In[60]:


for var in pos[1:]:
    bar0 = all_data[all_data['target']==0][var].mean()
    bar1 = all_data[all_data['target']==1][var].mean()
    plt.bar(['0', '1'], [bar0,bar1])
    plt.title('Sum of variable ' + var)
    plt.xlabel('Target')
    plt.show()


# SQL

# In[23]:


for var in pos[1:]:
    text = 'SELECT AVG(' + var + ')' + ' FROM application WHERE target=0;' 
    bar0 = query(text, conn)
    text = 'SELECT AVG(' + var + ')' + ' FROM application WHERE target=1;' 
    bar1 = query(text, conn)
    
    plt.bar(['0', '1'], [bar0.values[0][0],bar1.values[0][0]])
    plt.title('Sum of variable ' + var)
    plt.xlabel('Target')
    plt.show()


# In[24]:


df_objects = all_data.select_dtypes(include='object')


# In[25]:


df_objects


# In[ ]:
'''




