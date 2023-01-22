#!/usr/bin/env python
# coding: utf-8

# ### Dragon real estate price prediction

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


housing = pd.read_csv('data.csv')


# In[3]:


housing.head()


# In[4]:


housing.info()


# In[5]:


housing['CHAS'].value_counts()


# In[6]:


housing.describe()


# In[7]:


housing.hist(bins=50,figsize=(20,15))


# ### now check train test split data

# In[7]:


def train_test_split(data, test_ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    print(shuffled)
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


# In[9]:


train_set, test_set = train_test_split(housing, 0.2)


# In[10]:


print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}\n")


# In[11]:


### above this code we have total data 506 so testing doing 101, train doing 405


# In[10]:


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# In[11]:


strat_test_set['CHAS'].value_counts()


# In[12]:


strat_train_set['CHAS'].value_counts()


# In[15]:


housing = strat_train_set.copy()


# #### Model Selection 

# In[13]:


from sklearn.model_selection import train_test_split 


# In[14]:


train_set,test_set = train_test_split(housing, test_size=0.2, random_state=42)
print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}\n")


# In[18]:


### coreelations 


# In[15]:


corr_matrix = housing.corr()


# In[16]:


corr_matrix['MEDV'].sort_values(ascending=False)


# In[17]:


from pandas.plotting import scatter_matrix
attributes = ["MEDV","RM","ZN","LSTAT"]
scatter_matrix(housing[attributes], figsize =(12,8))


# In[20]:


housing.plot(kind="scatter", x="RM", y="MEDV", alpha=0.5 )


# #### Trying attribute combinations

# In[21]:


housing ["TAXRM"] = housing['TAX']/housing['RM']


# In[22]:


housing.head()


# In[23]:


corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)


# In[24]:


housing.plot(kind="scatter", x="TAXRM", y="MEDV", alpha=0.5 )


# In[25]:


housing = strat_train_set.drop("MEDV", axis=1)
housing_labels = strat_train_set["MEDV"].copy()


# ### Missing attributes

#   To care of missing attributes , we have three options 
#    1. get rid of the missing data points
#    2. get rid of the whole attribute
#    3. set the value to soome value(0, mean or median)

# In[27]:


a = housing.dropna(subset=["RM"])


#  Option 1 : dropna Drop Null Attributes

# In[28]:


a = housing.dropna(subset=["RM"])


# In[29]:


a.shape


#  Option 2  there is no RM column and also note that housing dataframe will remane changed 

# In[31]:


housing.drop("RM", axis=1)


#  Option 3 we used  compute median for option 3

# In[32]:


median = housing["RM"].median()


# In[33]:


housing["RM"].fillna(median)

 # note  there is no RM column and also note that housing dataframe will remane changed 


# In[34]:


housing.shape


# In[35]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = "median")
imputer.fit(housing)


# In[36]:


imputer.statistics_


# In[37]:


x = imputer.transform(housing)


# In[39]:


housing_tr = pd.DataFrame(x,columns=housing.columns)


# In[40]:


housing_tr.describe()


# # Scikit-learn Design

# Primarily, three types of objects
# 
# Estimators - It estimates some parameter based on a dataset. Eg. imputer. It has a fit method and transform method. Fit method - Fits the dataset and calculates internal parameters
# 
# Transformers - transform method takes input and returns output based on the learnings from fit(). It also has a convenience function called fit_transform() which fits and then transforms.
# 
# Predictors - LinearRegression model is an example of predictor. fit() and predict() are two common functions. It also gives score() function which will evaluate the predictions.

# # Feature Scaling

# Primarily, two types of feature scaling methods:
# 1. Min-max scaling (Normalization)
#     (value - min)/(max - min)
#     Sklearn provides a class called MinMaxScaler for this
#     
# 2. Standardization
#     (value - mean)/std
#     Sklearn provides a class called StandardScaler for this
# 
# 

# # Creating a Pipeline

# In[41]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler()),
])


# In[42]:


housing_num_tr = my_pipeline.fit_transform(housing_tr)


# In[43]:


housing_num_tr


# #### Selecting a desired model for Dragon real estate 

# In[44]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

 # model = LinearRegression()
# model = DecisionTreeRegressor()  
model = RandomForestRegressor()
model.fit(housing_num_tr, housing_labels)


# In[45]:


some_data = housing.iloc[:5]


# In[46]:


some_labels = housing_labels.iloc[:5]


# In[47]:


prepared_data = my_pipeline.transform(some_data)


# In[48]:


model.predict(prepared_data)


# In[50]:


list(some_labels)


# # Evaluatiing the model

# In[51]:


from sklearn.metrics import mean_squared_error
housing.predictions = model.predict(housing_num_tr)
mse = mean_squared_error(housing_labels,housing.predictions)
rmse = np.sqrt(mse)


# In[52]:


rmse


# above the meansuarederro(mse) we got 0.0 means our model is ovverfitting so now we use cross validation to solve this problem

# ## Using better evaluaton technique - Cross validation

# In[53]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, housing_num_tr, housing_labels, scoring="neg_mean_squared_error",cv=10)
rmse_scores = np.sqrt(-scores)


# In[55]:


rmse_scores


# In[56]:


def print_scores(score):
    print("Scores:", score)
    print(",Mean:", score.mean())
    print("Standard deviations:", score.std())


# In[57]:


print_scores(rmse_scores)


# ### save the model 

# In[58]:


from joblib import dump, load
dump(model, 'Dragon.joblib')


# #### Testing the model on test data

# In[59]:


x_test = strat_test_set.drop("MEDV", axis=1)
y_test = strat_test_set["MEDV"].copy()
x_test_prepared = my_pipeline.transform(x_test)
final_predictions = model.predict(x_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)


# In[60]:


final_rmse

