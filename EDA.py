import psycopg2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Connect to database 
conn = psycopg2.connect("dbname=credit_fraud user=usr password=pwd")

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

# **Outliers**
all_data.describe()

# Investigate suspicious variables individually
# amt_income_total
get_ipython().run_line_magic('matplotlib', 'inline')
sns.boxplot(all_data['amt_income_total'])
plt.show()

all_data[all_data['amt_income_total'] > 100000000]

# remove rows outside of mean +/- 3*|z|
all_data = all_data[np.abs(stats.zscore(all_data['amt_income_total'])) < 3]

# amt_credit
sns.boxplot(all_data['amt_credit'])
plt.show()

# remove rows outside of mean +/- 3*|z|
all_data = all_data[np.abs(stats.zscore(all_data['amt_credit'])) < 3]

# amt_annuity
get_ipython().run_line_magic('matplotlib', 'inline')
sns.boxplot(all_data['amt_annuity'])
plt.show()

# remove rows outside of mean +/- 3*|z|
all_data = all_data[np.abs(stats.zscore(all_data['amt_annuity'])) < 3]

# amt_goods_price
get_ipython().run_line_magic('matplotlib', 'inline')
sns.boxplot(all_data['amt_goods_price'])
plt.show()

# remove rows outside of mean +/- 3*|z|
all_data = all_data[np.abs(stats.zscore(all_data['amt_goods_price'])) < 3]

# region_population_relative
get_ipython().run_line_magic('matplotlib', 'inline')
sns.boxplot(all_data['region_population_relative'])
plt.show()

# remove rows outside of mean +/- 3*|z|
all_data = all_data[np.abs(stats.zscore(all_data['region_population_relative'])) < 3]

# years_employed
get_ipython().run_line_magic('matplotlib', 'inline')
sns.boxplot(all_data['years_employed'])
plt.show()

# remove rows outside of mean +/- 3*|z|
all_data = all_data[np.abs(stats.zscore(all_data['years_employed'])) < 3]

# years_registration
get_ipython().run_line_magic('matplotlib', 'inline')
sns.boxplot(all_data['years_registration'])
plt.show()

# remove rows outside of mean +/- 3*|z|
all_data = all_data[np.abs(stats.zscore(all_data['years_registration'])) < 3]

# ext_source_3
get_ipython().run_line_magic('matplotlib', 'inline')
sns.boxplot(all_data['ext_source_3'])
plt.show()

# remove rows outside of mean +/- 3*|z|
all_data = all_data[np.abs(stats.zscore(all_data['ext_source_3'])) < 3]

# days_last_phone_change
get_ipython().run_line_magic('matplotlib', 'inline')
sns.boxplot(all_data['years_last_phone_change'])
plt.show()

# remove rows outside of mean +/- 3*|z|
all_data = all_data[np.abs(stats.zscore(all_data['years_last_phone_change'])) < 3]

# **Correlation between variables**
# Python
# all-to-all
# temporarily make target var. int
all_data['target'] = all_data['target'].astype(int)

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
df_numeric = all_data.select_dtypes(include=numerics)
df_numeric.head()

corrs = df_numeric.corr()  # correlation of all-to-all
plt.figure(figsize=(12, 8))
sns.heatmap(corrs, cmap="coolwarm")
plt.show()
# corrs[np.abs(corrs>0.2) & (corrs<1)].stack().index # extract the column tuples with corr>0.2

# target-to-all
plt.figure(figsize=(12, 8))
corrs['target'].drop('target', axis=0).plot.bar(x='target', y='val')
plt.title('Histogram of target-to-all correlation')
plt.xlabel('Correlation')
plt.ylabel('No. of variables')
plt.show()

pos = corrs[corrs['target'] > 0.05].drop(['target'], axis=0).index.tolist()
neg = corrs[corrs['target'] < -0.05].index.tolist()
print('Variables with high positive correlation: \n', pos, '\n')
print('Variables with high negative correlation: \n', neg, '\n')

# SQL
neg_sql = ['days_birth', 'days_id_publish', 'ext_source_2', 'ext_source_3', 'days_last_phone_change']
text = 'SELECT '
for var in (pos + neg_sql):
    text = text + 'corr(target,' + var + ') ' + var + ','
text = text[:-1] + ' FROM application;'
df = query(text, conn)
df.index = ['target']
df

# **Univariate analysis: Visualise variables highly correlated with the target**
# Python
for var in pos:
    bar0 = all_data[all_data['target'] == 0][var].mean()
    bar1 = all_data[all_data['target'] == 1][var].mean()
    plt.bar(['0', '1'], [bar0, bar1])
    plt.title('Mean of variable ' + var)
    plt.xlabel('Target')
    plt.show()

for var in neg:
    bar0 = all_data[all_data['target'] == 0][var].median()
    bar1 = all_data[all_data['target'] == 1][var].median()
    plt.bar(['0', '1'], [bar0, bar1])
    plt.title('Median of variable ' + var)
    plt.xlabel('Target')
    plt.show()

# SQL
for var in pos:
    text = 'SELECT AVG(' + var + ')' + ' FROM application WHERE target=0;'
    bar0 = query(text, conn)
    text = 'SELECT AVG(' + var + ')' + ' FROM application WHERE target=1;'
    bar1 = query(text, conn)

    plt.bar(['0', '1'], [bar0.values[0][0], bar1.values[0][0]])
    plt.title('Mean of variable ' + var)
    plt.xlabel('Target')
    plt.show()

for var in neg_sql:
    text = 'SELECT AVG(' + var + ')' + ' FROM application WHERE target=0;'
    bar0 = query(text, conn)
    text = 'SELECT AVG(' + var + ')' + ' FROM application WHERE target=1;'
    bar1 = query(text, conn)

    plt.bar(['0', '1'], [bar0.values[0][0], bar1.values[0][0]])
    plt.title('Mean of variable ' + var)
    plt.xlabel('Target')
    plt.show()

# Categorical variables
df_objects = all_data.select_dtypes(include='object')
df_objects

# name_contract_type
plt.figure(figsize=(20, 8))

plt.subplot(1, 2, 1)
ax = sns.countplot(all_data[all_data['target'] == 0]['name_contract_type'],
                   order=all_data[all_data['target'] == 0]['name_contract_type'].value_counts().index)
plt.title('Target 0')

plt.subplot(1, 2, 2)
ax = sns.countplot(all_data[all_data['target'] == 1]['name_contract_type'],
                   order=all_data[all_data['target'] == 0]['name_contract_type'].value_counts().index)
plt.title('Target 1')
plt.show()

# code_gender
# there are more females in the dataset, which explains the higher bar on the right plot;<br>
# however, males are more likely than females to have target=1
plt.figure(figsize=(20, 8))

plt.subplot(1, 2, 1)
ax = sns.countplot(all_data[all_data['target'] == 0]['code_gender'],
                   order=all_data[all_data['target'] == 0]['code_gender'].value_counts().index)
plt.title('Target 0')

plt.subplot(1, 2, 2)
ax = sns.countplot(all_data[all_data['target'] == 1]['code_gender'],
                   order=all_data[all_data['target'] == 0]['code_gender'].value_counts().index)
plt.title('Target 1')
plt.show()

print('Male percentage of target=1: ', 100 * all_data[all_data['code_gender'] == 'M']['target'].value_counts()[1] /
      all_data[all_data['code_gender'] == 'M']['target'].value_counts().sum(), ' %')
print('Female percentage of target=1: ', 100 * all_data[all_data['code_gender'] == 'F']['target'].value_counts()[1] /
      all_data[all_data['code_gender'] == 'F']['target'].value_counts().sum(), ' %')

# flag_own_car
plt.figure(figsize=(20, 8))

plt.subplot(1, 2, 1)
ax = sns.countplot(all_data[all_data['target'] == 0]['flag_own_car'],
                   order=all_data[all_data['target'] == 0]['flag_own_car'].value_counts().index)
plt.title('Target 0')

plt.subplot(1, 2, 2)
ax = sns.countplot(all_data[all_data['target'] == 1]['flag_own_car'],
                   order=all_data[all_data['target'] == 0]['flag_own_car'].value_counts().index)
plt.title('Target 1')
plt.show()

# flag_own_realty
plt.figure(figsize=(20, 8))

plt.subplot(1, 2, 1)
ax = sns.countplot(all_data[all_data['target'] == 0]['flag_own_realty'],
                   order=all_data[all_data['target'] == 0]['flag_own_realty'].value_counts().index)
plt.title('Target 0')

plt.subplot(1, 2, 2)
ax = sns.countplot(all_data[all_data['target'] == 1]['flag_own_realty'],
                   order=all_data[all_data['target'] == 0]['flag_own_realty'].value_counts().index)
plt.title('Target 1')
plt.show()

# name_type_suite
plt.figure(figsize=(20, 8))

plt.subplot(1, 2, 1)
ax = sns.countplot(all_data[all_data['target'] == 0]['name_type_suite'],
                   order=all_data[all_data['target'] == 0]['name_type_suite'].value_counts().index)
plt.title('Target 0')

plt.subplot(1, 2, 2)
ax = sns.countplot(all_data[all_data['target'] == 1]['name_type_suite'],
                   order=all_data[all_data['target'] == 0]['name_type_suite'].value_counts().index)
plt.title('Target 1')
plt.show()

# name_income_type
plt.figure(figsize=(25, 8))

plt.subplot(1, 2, 1)
ax = sns.countplot(all_data[all_data['target'] == 0]['name_income_type'],
                   order=all_data[all_data['target'] == 0]['name_income_type'].value_counts().index)
plt.title('Target 0')

plt.subplot(1, 2, 2)
ax = sns.countplot(all_data[all_data['target'] == 1]['name_income_type'],
                   order=all_data[all_data['target'] == 0]['name_income_type'].value_counts().index)
plt.title('Target 1')
plt.show()

# name_education_type
plt.figure(figsize=(25, 8))

plt.subplot(1, 2, 1)
ax = sns.countplot(all_data[all_data['target'] == 0]['name_education_type'],
                   order=all_data[all_data['target'] == 0]['name_education_type'].value_counts().index)
plt.title('Target 0')

plt.subplot(1, 2, 2)
ax = sns.countplot(all_data[all_data['target'] == 1]['name_education_type'],
                   order=all_data[all_data['target'] == 0]['name_education_type'].value_counts().index)
plt.title('Target 1')
plt.show()

# name_family_status
plt.figure(figsize=(25, 8))

plt.subplot(1, 2, 1)
ax = sns.countplot(all_data[all_data['target'] == 0]['name_family_status'],
                   order=all_data[all_data['target'] == 0]['name_family_status'].value_counts().index)
plt.title('Target 0')

plt.subplot(1, 2, 2)
ax = sns.countplot(all_data[all_data['target'] == 1]['name_family_status'],
                   order=all_data[all_data['target'] == 0]['name_family_status'].value_counts().index)
plt.title('Target 1')
plt.show()

# name_housing_type
plt.figure(figsize=(25, 8))

plt.subplot(1, 2, 1)
ax = sns.countplot(all_data[all_data['target'] == 0]['name_housing_type'],
                   order=all_data[all_data['target'] == 0]['name_housing_type'].value_counts().index)
plt.title('Target 0')

plt.subplot(1, 2, 2)
ax = sns.countplot(all_data[all_data['target'] == 1]['name_housing_type'],
                   order=all_data[all_data['target'] == 0]['name_housing_type'].value_counts().index)
plt.title('Target 1')
plt.show()

# occupation_type
plt.figure(figsize=(25, 8))

plt.subplot(1, 2, 1)
ax = sns.countplot(all_data[all_data['target'] == 0]['occupation_type'],
                   order=all_data[all_data['target'] == 0]['occupation_type'].value_counts().index)
plt.title('Target 0')

plt.subplot(1, 2, 2)
ax = sns.countplot(all_data[all_data['target'] == 1]['occupation_type'],
                   order=all_data[all_data['target'] == 0]['occupation_type'].value_counts().index)
plt.title('Target 1')
plt.show()

# for better understanding of the plot above, here are the occupation types:
all_data[all_data['target'] == 0]['occupation_type'].value_counts().index

# weekday_appr_process_start
plt.figure(figsize=(25, 8))

plt.subplot(1, 2, 1)
ax = sns.countplot(all_data[all_data['target'] == 0]['weekday_appr_process_start'],
                   order=all_data[all_data['target'] == 0]['weekday_appr_process_start'].value_counts().index)
plt.title('Target 0')

plt.subplot(1, 2, 2)
ax = sns.countplot(all_data[all_data['target'] == 1]['weekday_appr_process_start'],
                   order=all_data[all_data['target'] == 0]['weekday_appr_process_start'].value_counts().index)
plt.title('Target 1')
plt.show()

# organization_type
plt.figure(figsize=(25, 8))

plt.subplot(1, 2, 1)
ax = sns.countplot(all_data[all_data['target'] == 0]['organization_type'],
                   order=all_data[all_data['target'] == 0]['organization_type'].value_counts().index)
plt.title('Target 0')

plt.subplot(1, 2, 2)
ax = sns.countplot(all_data[all_data['target'] == 1]['organization_type'],
                   order=all_data[all_data['target'] == 0]['organization_type'].value_counts().index)
plt.title('Target 1')
plt.show()

# for better understanding of the plot above, here are the organization types:
all_data[all_data['target'] == 0]['organization_type'].value_counts().index

# **Set column types**
all_data.info()

for col in all_data.columns:
    if (all_data[col].dtype != 'int64') and (all_data[col].dtype != 'float64'):
        all_data[col] = all_data[col].astype('category')

categorical_cols = ['target', 'name_contract_type', 'code_gender', 'flag_own_car', 'flag_own_realty',
                    'reg_city_not_work_city', 'live_city_not_work_city', 'reg_city_not_live_city',
                    'amt_req_credit_bureau_hour',
                    'name_education_type', 'name_housing_type', 'name_family_status', 'weekday_appr_process_start',
                    'name_type_suite', 'name_income_type', 'amt_req_credit_bureau_day', 'amt_req_credit_bureau_week',
                    'def_60_cnt_social_circle', 'def_30_cnt_social_circle', 'amt_req_credit_bureau_qrt', 'cnt_children',
                    'cnt_fam_members', 'occupation_type', 'hour_appr_process_start', 'amt_req_credit_bureau_mon',
                    'amt_req_credit_bureau_year', 'obs_60_cnt_social_circle', 'obs_30_cnt_social_circle',
                    'organization_type',
                    'flag_document_3', 'flag_document_6', 'flag_document_8', 'reg_region_not_live_region',
                    'reg_region_not_work_region', 'live_region_not_work_region', 'flag_email', 'flag_phone',
                    'flag_work_phone',
                    'flag_emp_phone', 'region_rating_client', 'region_rating_client_w_city']

for col in categorical_cols:
    all_data[col] = all_data[col].astype('category')

# **Divide column numerical range into quartile categories**
all_data['code_gender'].value_counts()
all_data.loc[all_data['code_gender'] == 'XNA', 'code_gender'] = all_data['code_gender'].mode()[0]

qnt10 = np.arange(0, 1.1, 0.2)  # 5 intervals
bins = np.quantile(all_data['amt_income_total'], qnt10)
labels = [0, 1, 2, 3, 4]
all_data['amt_income_total_cat'] = pd.cut(all_data['amt_income_total'], bins, labels=labels)

qnt10 = np.arange(0, 1.1, 0.2)  # 5 intervals
bins = np.quantile(all_data['amt_credit'], qnt10)
labels = [0, 1, 2, 3, 4]
all_data['amt_credit_cat'] = pd.cut(all_data['amt_credit'], bins, labels=labels)

all_data.drop(['amt_credit','amt_income_total'],axis=1,inplace=True)

all_data.info()

# **Data preparation for model**
# Get dummies of categorical variables
cat_cols = all_data.select_dtypes('category').columns.tolist()
cat_cols

for col in cat_cols:
    all_data = pd.concat([all_data,pd.get_dummies(all_data[col],prefix=col)],axis=1)
    all_data.drop(col,axis=1,inplace=True)

# **Save cleaned dataset**
all_data.to_csv('data/cleaned_application_data.csv')