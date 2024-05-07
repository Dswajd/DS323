#!/usr/bin/env python
# coding: utf-8

# #### DS 323: Machine Learning
# #### Project 
# ##### 2nd Semester, 2023-2024
# 

# <center>
#   <img src='pnupng.png' alt="Image Description">
# </center>

# <center><h2><strong>Using Machine Learning Models to predict next year's Allegheny County Employee Salaries based on historical data of the last 5 years</strong></h2></center>

# <p style="text-align: center;"><strong>Group #1</strong></p>

# <table style="text-align: left;">
#   <tr>
#     <th>Student Names</th>
#     <th>Student IDs</th>
#   </tr>
#   <tr>
#     <td>Asayil Alangari</td>
#     <td>443007452</td>
#   </tr>
#   <tr>
#     <td>Layan Alosaimi</td>
#     <td>443007398</td>
#   </tr>
#   <tr>
#     <td>Reema Alshehri</td>
#     <td>443007421</td>
#   </tr>
#   <tr>
#     <td>Wajd Basfar</td>
#     <td>443007410</td>
#   </tr>
# </table>

# ## Table of Contents 
# 
# - [Problem Statement](#problem)
# - [Dataset Description](#dataset)
# - [Data Preprocessing](#process)
# - [Core Analysis](#core)
# - [Explanatory Data Analysis](#eda)
# - [Building ML Models](#build)
# - [Evaluaton of ML Models](#eval)
# 

# <a id='problem'></a>
# ## Problem Statement

# >The county government is facing budget constraints due to the rising costs of employee salaries. The aim of this project is to develop a machine learning model that can accurately predict employee salaries for the next year based on historical data. The objectives are to reduce budget overruns by 10% within the next year and improve budget planning by accurately predicting salary expenses. This problem can be solved using a regression model, trained on historical employee salary data from 2016 to 2020

# <a id='dataset'></a>
# ## Dataset Description

# We created a new dataset of multiple existing datasets by merging them all together and drop some unused columns.
# 

# Here are the orginal datasets:https://catalog.data.gov/dataset/allegheny-county-employee-salaries

# > There are 6 different datasets, each of them represents a single year of Allegheny County Employee Salaries from 2016 up to 2021. Each dataset consists of 16 columns which are: FIRST_NAME, LAST_NAME, DEPARTMENT, JOB_TITLE, ELECTED_OFFICIAL, DATE_START, SEX, ETHNICITY, ORIG_START, DATE_TERM, PAY_STATUSTED, ANNUAL_SALARY, REGULAR_PAY, OVERTIME_PAY, INCENTIVE_PAY, GROSS_PAY. Each datasets has more 6000 record. Datasets' resource is data.gov.

# Steps we took to finally get our newly dataset:
# 1) Of each dataset we dropped the following columns: DEPARTMENT, JOB_TITLE, ELECTED_OFFICIAL, DATE_START, SEX, ETHNICITY, ORIG_START, DATE_TERM, PAY_STATUSTED.
# 
# 
# 2) Of each dataset we renamed ANNUAL_SALARY, REGULAR_PAY, OVERTIME_PAY, INCENTIVE_PAY, GROSS_PAY columns by their year.
# 
# 
# 3) We assigned FIRST_NAME, LAST_NAME as the common columns for merging, so each record assigns to its record in the other dataset.
# 
# 
# 4) We start merging each two datasets together up to the final dataset.
# 
# 
# 5) After finally merging all the datasets together. We dropped the missing values, because they belong to the employees that were not been working the whole 6 years. Our target is only on the employees that had been working all the past 6 years, so we can get more accurate results.
# 
# 
# 6) We did preprocess the data such as dropping the missing values, and dropping the replicated records.

# The new dataset description:

# > The dataset consists of 32 columns which are FIRST_NAME, LAST_NAME, ANNUAL_SALARY_16, REGULAR_PAY_16, OVERTIME_PAY_16, INCENTIVE_PAY_16, GROSS_PAY_16, ANNUAL_SALARY_17, REGULAR_PAY_17, OVERTIME_PAY_17, INCENTIVE_PAY_17, GROSS_PAY_17, ANNUAL_SALARY_18, REGULAR_PAY_18, OVERTIME_PAY_18, INCENTIVE_PAY_18, GROSS_PAY_18, ANNUAL_SALARY_19, REGULAR_PAY_19, OVERTIME_PAY_19, INCENTIVE_PAY_19, GROSS_PAY_19, ANNUAL_SALARY_20, REGULAR_PAY_20, OVERTIME_PAY_20, INCENTIVE_PAY_20, GROSS_PAY_20, ANNUAL_SALARY_21, REGULAR_PAY_21, OVERTIME_PAY_21, INCENTIVE_PAY_21, GROSS_PAY_21. Has 860 records, which are the employees that stayed the whole 6 years from 2016 to 2021. 

# In[52]:


#Loading the new dataset
import pandas as pd
import pickle
df = pd.read_csv('NewCollectedDataset (3).csv')


# In[2]:


df


# In[3]:


df.info()


# <a id='process'></a>
# ## Data Preprcessing

# In[4]:


# 1- MISSING VALUES
# Check missing values
df.isnull().sum()


# In[5]:


# 2- DUPLICATED VALUES
# Check for duplicated values
df.duplicated().sum()


# In[6]:


# 3- OUTLIERS
# Check outliers
#Q1 =df.quantile(0.25)
#Q3 =df.quantile(0.75)
#IQR = Q3 - Q1
#IQR


# In[7]:


for column in df.columns:
    df[column] = pd.to_numeric(df[column], errors='coerce')

# Select only numeric columns
numeric_df = df.select_dtypes(include=['number'])

# Calculate quantiles and IQR
Q1 = numeric_df.quantile(0.25)
Q3 = numeric_df.quantile(0.75)
IQR = Q3 - Q1

# Optionally, handle NaN values, e.g., by filling or dropping them
# numeric_df.fillna(numeric_df.mean(), inplace=True)

print("Q1:\n", Q1)
print("Q3:\n", Q3)
print("IQR:\n", IQR)


# In[8]:


((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).sum()


# In[9]:


df_outliers = df[~((df < (Q1 - 1.5 * IQR)) |(df> (Q3 + 1.5 * IQR))).any(axis=1)]


# In[10]:


((df_outliers < (Q1 - 1.5 * IQR)) |(df_outliers > (Q3 + 1.5 * IQR))).sum()


# In[11]:


# Define main TARGET and main PREDICTORS
df_outliers['Total_Pay_21'] = df_outliers['ANNUAL_SALARY_21'] + df_outliers['REGULAR_PAY_21'] + df_outliers['OVERTIME_PAY_21'] + df_outliers['INCENTIVE_PAY_21'] + df_outliers['GROSS_PAY_21']
df_outliers['Total_Pay_20'] = df_outliers['ANNUAL_SALARY_20'] + df_outliers['REGULAR_PAY_20'] + df_outliers['OVERTIME_PAY_20'] + df_outliers['INCENTIVE_PAY_20'] + df_outliers['GROSS_PAY_20']
df_outliers['Total_Pay_19'] = df_outliers['ANNUAL_SALARY_19'] + df_outliers['REGULAR_PAY_19'] + df_outliers['OVERTIME_PAY_19'] + df_outliers['INCENTIVE_PAY_19'] + df_outliers['GROSS_PAY_19']
df_outliers['Total_Pay_18'] = df_outliers['ANNUAL_SALARY_18'] + df_outliers['REGULAR_PAY_18'] + df_outliers['OVERTIME_PAY_18'] + df_outliers['INCENTIVE_PAY_18'] + df_outliers['GROSS_PAY_18']
df_outliers['Total_Pay_17'] = df_outliers['ANNUAL_SALARY_17'] + df_outliers['REGULAR_PAY_17'] + df_outliers['OVERTIME_PAY_17'] + df_outliers['INCENTIVE_PAY_17'] + df_outliers['GROSS_PAY_17']
df_outliers['Total_Pay_16'] = df_outliers['ANNUAL_SALARY_16'] + df_outliers['REGULAR_PAY_16'] + df_outliers['OVERTIME_PAY_16'] + df_outliers['INCENTIVE_PAY_16'] + df_outliers['GROSS_PAY_16']


# <a id='eda'></a>
# ## Explanatory Data Analysis

# <a id='core'></a>
# ## Core Analysis

# In[12]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[13]:


correlation_matrix = df_outliers[['Total_Pay_17','Total_Pay_16','Total_Pay_18','Total_Pay_19','Total_Pay_20', 'Total_Pay_21']].corr()


# In[14]:


# Print the correlation matrix
print(correlation_matrix)

correlation_with_target = correlation_matrix['Total_Pay_21']
print(correlation_with_target)


# In[15]:


# Create a heatmap plot to visualize the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Create scatter plots to visualize the relationships between independent variables and the dependent variable
for col in df_outliers[['Total_Pay_17','Total_Pay_16','Total_Pay_18','Total_Pay_19','Total_Pay_20', 'Total_Pay_21']]:
    plt.scatter(df_outliers[col], df_outliers['Total_Pay_21'])
    plt.xlabel(col)
    plt.ylabel('Total Pay for year 2021')
    plt.title(f'{col} vs. Total Pay for year 2021')
    plt.show()


# <a id='build'></a>
# ## Building ML Models

# In[16]:


#Import all the needed libaries
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor


# In[17]:


# Split predictors and target variables
predictors = df_outliers[['Total_Pay_17','Total_Pay_16','Total_Pay_18','Total_Pay_19','Total_Pay_20']]
targets = df_outliers[['Total_Pay_21']]


# ### First Model: Gradient Boosting 

# In[18]:


# Split the data into training and testing sets
X_train_lnr, X_test_lnr, y_train_lnr, y_test_lnr = train_test_split(predictors, targets, test_size=0.2, random_state=42)


# In[57]:


# Create an LinearRegression
lnr = GradientBoostingRegressor() 

# Fit the LNR model to the training data
lnr.fit(X_train_lnr, y_train_lnr)
pickle.dump(lnr,open('gradient_boosting_model.pkl', 'wb') ) 
# Make predictions on the test set
y_pred_lnr = lnr.predict(X_test_lnr)

# Evaluate the model
trainingAccuracy = lnr.score(X_train_lnr, y_train_lnr) 
print("Training Accuracy:",trainingAccuracy)
testAccuracy = lnr.score(X_test_lnr, y_test_lnr) 
print("Testing Accuracy:",testAccuracy)


# In[20]:


# Tune some parameters by using GridSearchCV

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.5],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create the GridSearchCV object
grid_search = GridSearchCV(lnr, param_grid, cv=5)

# Fit the GridSearchCV object to the training data
grid_search.fit(X_train_lnr, y_train_lnr)


# In[21]:


# Get the best parameters and best score from GridSearchCV
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best Parameters:", best_params)
print("Best Score (negative mean squared error):", best_score)


# In[22]:


# Rebuild a model on the training set using the optimum parameters' values
# evaluate the model on the test set
# Split the data into training and testing sets
X_train1_lnr, X_test1_lnr, y_train1_lnr, y_test1_lnr = train_test_split(predictors, targets, test_size=0.2, random_state=42)
relnr = grid_search.best_estimator_
relnr.fit(X_train1_lnr, y_train1_lnr)
y_pred1_lnr = relnr.predict(X_test1_lnr)
training_score = relnr.score(X_train1_lnr, y_train1_lnr)
test_score = relnr.score(X_test1_lnr, y_test1_lnr)

print("Training set score with best parameters: {:.2f}".format(training_score))
print("Test set score with best parameters: {:.2f}".format(test_score))


# ### Second Model: MLP

# In[23]:


# Split the data into training and testing sets
X_train_mlp, X_test_mlp, y_train_mlp, y_test_mlp = train_test_split(predictors, targets, test_size=0.2, random_state=42)


# In[56]:


# Create MLPRegressor
mlp = MLPRegressor()

# Fit the model to the training data
mlp.fit(X_train_mlp, y_train_mlp)
pickle.dump(mlp,open('mlp_model.pkl', 'wb') ) 
# Make predictions on the test data
y_pred_mlp = mlp.predict(X_test_mlp)

# Evaluate the model
trainingAccuracy = mlp.score(X_train_mlp, y_train_mlp) 
print("Training Accuracy:",trainingAccuracy)
testAccuracy = mlp.score(X_test_mlp, y_test_mlp) 
print("Testing Accuracy:",testAccuracy)


# In[25]:


# Tune some parameters by using GridSearchCV

# Define the parameter grid
param_grid = {
    'hidden_layer_sizes': [(100,), (50, 50), (100, 50)],
    'activation': ['relu', 'tanh', 'identity'],
    'solver': ['adam', 'lbfgs', 'sgd'],
    'learning_rate': ['constant', 'invscaling', 'adaptive'],
    'alpha': [0.0001, 0.001, 0.01]
}

# Create the GridSearchCV object
grid_search = GridSearchCV(mlp, param_grid, scoring='neg_mean_squared_error', cv=5)

# Fit the GridSearchCV object to your training data
grid_search.fit(X_train_mlp, y_train_mlp)


# In[26]:


# Get the best parameters and best score from GridSearchCV
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best Parameters:", best_params)
print("Best Score (negative mean squared error):", best_score)


# In[27]:


# Rebuild a model on the training set using the optimum parameters' values
# evaluate the model on the test set
# Split the data into training and testing sets
X_train1_mlp, X_test1_mlp, y_train1_mlp, y_test1_mlp = train_test_split(predictors, targets, test_size=0.2, random_state=42)
remlp = grid_search.best_estimator_
remlp.fit(X_train1_mlp, y_train1_mlp)
y_pred1_mlp = remlp.predict(X_test1_mlp)
training_score = remlp.score(X_train1_mlp, y_train1_mlp)
test_score = remlp.score(X_test1_mlp, y_test1_mlp)

print("Training set score with best parameters: {:.2f}".format(training_score))
print("Test set score with best parameters: {:.2f}".format(test_score))


# ### Third Model: K- Nearest Neigbor ----- Proposed Model 

# In[28]:


# Split the data into training and testing sets
X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(predictors, targets, test_size=0.2, random_state=42)


# In[55]:


# Create an KNNRegressor
knn = KNeighborsRegressor() 

# Fit the KNN model to the training data
knn.fit(X_train_knn, y_train_knn)
pickle.dump(knn,open('knn_model.pkl', 'wb') ) 
# Make predictions on the test set
y_pred_knn = knn.predict(X_test_knn)

# Evaluate the model
trainingAccuracy = knn.score(X_train_knn, y_train_knn) 
print("Training Accuracy:",trainingAccuracy)
testAccuracy = knn.score(X_test_knn, y_test_knn) 
print("Testing Accuracy:",testAccuracy)


# In[30]:


# Tune some parameters by using GridSearchCV

# Set up the parameter grid for GridSearchCV
param_grid = {
    'n_neighbors': [5, 10, 15],
    'weights': ['uniform', 'distance'],
    'metric': ['minkowski', 'manhattan'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
}

# Create a GridSearchCV object
grid_search = GridSearchCV(knn, param_grid, scoring='neg_mean_squared_error')

# Fit the GridSearchCV object to the training data
grid_search.fit(X_train_knn, y_train_knn)


# In[31]:


# Get the best parameters and best score from GridSearchCV
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best Parameters:", best_params)
print("Best Score (negative mean squared error):", best_score)


# In[32]:


# Rebuild a model on the training set using the optimum parameters' values
# evaluate the model on the test set
# Split the data into training and testing sets
X_train1_knn, X_test1_knn, y_train1_knn, y_test1_knn = train_test_split(predictors, targets, test_size=0.2, random_state=42)
reknn = grid_search.best_estimator_
reknn.fit(X_train1_knn, y_train1_knn)
y_pred1_knn = reknn.predict(X_test1_knn)
training_score = reknn.score(X_train1_knn, y_train1_knn)
test_score = reknn.score(X_test1_knn, y_test1_knn)

print("Training set score with best parameters: {:.2f}".format(training_score))
print("Test set score with best parameters: {:.2f}".format(test_score))


# ### Fourth Model: Random Forest

# In[33]:


# Split the data into training and testing sets
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(predictors, targets, test_size=0.2, random_state=42)


# In[54]:


# Create an RandomForestRegressor
rf = RandomForestRegressor()

# Fit the RF model to the training data
rf.fit(X_train_rf, y_train_rf)

# Make predictions on the test set
y_pred_rf = rf.predict(X_test_rf)
pickle.dump(rf,open('rf_model.pkl', 'wb') ) 

# Evaluate the model
trainingAccuracy = rf.score(X_train_rf, y_train_rf) 
print("Training Accuracy:",trainingAccuracy)
testAccuracy = rf.score(X_test_rf, y_test_rf) 
print("Testing Accuracy:",testAccuracy)


# In[35]:


# Tune some parameters by using GridSearchCV

# Define the parameter grid
param_grid = {
    'n_estimators': [100],
    'max_features': ['sqrt'],
    'max_depth': [10],
    'min_samples_split': [2],
    'min_samples_leaf': [1],
    'bootstrap': [True]
}

# Create the GridSearchCV object
grid_search = GridSearchCV(rf, param_grid, cv=5)

# Fit the GridSearchCV object to the training data
grid_search.fit(X_train_rf, y_train_rf)


# In[36]:


# Get the best parameters and best score from GridSearchCV
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best Parameters:", best_params)
print("Best Score (negative mean squared error):", best_score)


# In[37]:


# Rebuild a model on the training set using the optimum parameters' values
# evaluate the model on the test set
# Split the data into training and testing sets
X_train1_rf, X_test1_rf, y_train1_rf, y_test1_rf = train_test_split(predictors, targets, test_size=0.2, random_state=42)
rerf = grid_search.best_estimator_
rerf.fit(X_train1_rf, y_train1_rf)
y_pred1_rf = rerf.predict(X_test1_rf)
training_score = rerf.score(X_train1_rf, y_train1_rf)
test_score = rerf.score(X_test1_rf, y_test1_rf)

print("Training set score with best parameters: {:.2f}".format(training_score))
print("Test set score with best parameters: {:.2f}".format(test_score))


# <a id='eval'></a>
# ## Evaluaton of ML Models

# In[38]:


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score


# ### First Model: Gradient Boosting 

# In[39]:


# DEFAULT PARAMETERS
# Calculate mean squared error (MSE)
mse = mean_squared_error(y_test_lnr, y_pred_lnr)
print("Mean Squared Error (MSE):", mse)

# Calculate mean absolute error (MAE)
mae = mean_absolute_error(y_test_lnr, y_pred_lnr)
print("Mean Absolute Error (MAE):", mae)

# Calculate R-squared (coefficient of determination)
r2 = r2_score(y_test_lnr, y_pred_lnr)
print("R-squared (R2):", r2)

explained_variance = explained_variance_score(y_test_lnr, y_pred_lnr)
print("Explained Variance Score:", explained_variance)


# In[40]:


#BEST PARAMETERS
# Calculate mean squared error (MSE)
mse = mean_squared_error(y_test1_lnr, y_pred1_lnr)
print("Mean Squared Error (MSE):", mse)

# Calculate mean absolute error (MAE)
mae = mean_absolute_error(y_test1_lnr, y_pred1_lnr)
print("Mean Absolute Error (MAE):", mae)

# Calculate R-squared (coefficient of determination)
r2 = r2_score(y_test1_lnr, y_pred1_lnr)
print("R-squared (R2):", r2)

explained_variance = explained_variance_score(y_test1_lnr, y_pred1_lnr)
print("Explained Variance Score:", explained_variance)


# When comparing these two sets, lower values of MSE and MAE indicate better performance, while higher values of R-squared and explained variance score indicate better fit.
# - Comparing the MSE and MAE:
#    - The set that is after tuning some parameters has a lower MSE (120442335.15150993) compared to the default parameters set (124198647.36456893). This suggests that the predictions after tuning some parameters have, on average, smaller squared differences from the true values, indicating better accuracy.
#    - The set that is after tuning some parameters also has a lower MAE (6722.786741094893) compared to the default parameters set (6715.53166621311). This indicates that the predictions in after tuning some parameters have, on average, smaller absolute differences from the true values, suggesting better accuracy as well.
# - Comparing the R-squared and explained variance score:
#    - The set that is after tuning some parameters has a higher R-squared (0.9436249796250809) compared to the default parameters set (0.9418667757735915). This indicates that the model in after tuning some parameters explains a larger proportion of the variance in the target variable, suggesting a better fit to the data.
#   - Similarly, The set that is after tuning some parameters has a higher explained variance score (0.9436330750259015) compared to he default parameters set (0.9419630911135012), indicating that the model after tuning some parameters captures a larger portion of the underlying patterns in the data.
#   
#   
# Based on these comparisons, after tuning some parameters generally outperforms the model with default parameters across all evaluation metrics. It has lower MSE and MAE values, indicating better accuracy, as well as higher R-squared and explained variance scores, indicating a better fit to the data. Therefore, After tuning some parameters is considered better than using the default parameters in terms of model performance.

# ### Second Model: MLP

# In[41]:


# DEFAULT PARAMETERS
# Calculate mean squared error (MSE)
mse = mean_squared_error(y_test_mlp, y_pred_mlp)
print("Mean Squared Error (MSE):", mse)

# Calculate mean absolute error (MAE)
mae = mean_absolute_error(y_test_mlp, y_pred_mlp)
print("Mean Absolute Error (MAE):", mae)

# Calculate R-squared (coefficient of determination)
r2 = r2_score(y_test_mlp, y_pred_mlp)
print("R-squared (R2):", r2)

explained_variance = explained_variance_score(y_test_mlp, y_pred_mlp)
print("Explained Variance Score:", explained_variance)


# In[42]:


#BEST PARAMETERS
# Calculate mean squared error (MSE)
mse = mean_squared_error(y_test1_mlp, y_pred1_mlp)
print("Mean Squared Error (MSE):", mse)

# Calculate mean absolute error (MAE)
mae = mean_absolute_error(y_test1_mlp, y_pred1_mlp)
print("Mean Absolute Error (MAE):", mae)

# Calculate R-squared (coefficient of determination)
r2 = r2_score(y_test1_mlp, y_pred1_mlp)
print("R-squared (R2):", r2)

explained_variance = explained_variance_score(y_test1_mlp, y_pred1_mlp)
print("Explained Variance Score:", explained_variance)


# When comparing these two sets, lower values of MSE and MAE indicate better performance, while higher values of R-squared and explained variance score indicate better fit.
# 
# - Comparing the MSE and MAE:
# 
#   - The set that is after tuning some parameters has a lower MSE (123305170.5816038) compared to the default parameters set (410021911.8735744). This suggests that the predictions after tuning some parameters have, on average, smaller squared differences from the true values, indicating better accuracy.
#   - The set that is after tuning some parameters also has a lower MAE (5935.114552856925) compared to the default parameters set (14387.928793969657). This indicates that the predictions in after tuning some parameters have, on average, smaller absolute differences from the true values, suggesting better accuracy as well.
# 
# - Comparing the R-squared and explained variance score:
# 
#   - The set that is after tuning some parameters has a higher R-squared (0.9422849823101952) compared to he default parameters set (0.8080824852244965). This indicates that the model in after tuning some parameters explains a larger proportion of the variance in the target variable, suggesting a better fit to the data.
#   - Similarly, The set that is after tuning some parameters has a higher explained variance score (0.9423452728183545) compared to he default parameters set (0.8231292445324268), indicating that the model after tuning some parameters captures a larger portion of the underlying patterns in the data.
#   
# Based on these comparisons, after tuning some parameters generally outperforms the model with default parameters across all evaluation metrics. It has lower MSE and MAE values, indicating better accuracy, as well as higher R-squared and explained variance scores, indicating a better fit to the data. Therefore, After tuning some parameters is considered better than using the default parameters in terms of model performance.

# ### Third Model: K- Nearest Neigbor --------- [Proposed Model]

# In[43]:


# DEFAULT PARAMETERS
# Calculate mean squared error (MSE)
mse = mean_squared_error(y_test_knn, y_pred_knn)
print("Mean Squared Error (MSE):", mse)

# Calculate mean absolute error (MAE)
mae = mean_absolute_error(y_test_knn, y_pred_knn)
print("Mean Absolute Error (MAE):", mae)

# Calculate R-squared (coefficient of determination)
r2 = r2_score(y_test_knn, y_pred_knn)
print("R-squared (R2):", r2)

explained_variance = explained_variance_score(y_test_knn, y_pred_knn)
print("Explained Variance Score:", explained_variance)


# In[44]:


#BEST PARAMETERS
# Calculate mean squared error (MSE)
mse = mean_squared_error(y_test1_knn, y_pred1_knn)
print("Mean Squared Error (MSE):", mse)

# Calculate mean absolute error (MAE)
mae = mean_absolute_error(y_test1_knn, y_pred1_knn)
print("Mean Absolute Error (MAE):", mae)

# Calculate R-squared (coefficient of determination)
r2 = r2_score(y_test1_knn, y_pred1_knn)
print("R-squared (R2):", r2)

explained_variance = explained_variance_score(y_test1_knn, y_pred1_knn)
print("Explained Variance Score:", explained_variance)


# When comparing these two sets, lower values of MSE and MAE indicate better performance, while higher values of R-squared and explained variance score indicate better fit.
# - Comparing the MSE and MAE:
# 
#    - The set that is after tuning some parameters has a lower MSE (105999431.38878469) compared to the default parameters set (114804280.92545553). This suggests that the predictions after tuning some parameters have, on average, smaller squared differences from the true values, indicating better accuracy.
#   - The set that is after tuning some parameters also has a lower MAE (5859.971824706159) compared to the default parameters set (6095.094173913045). This indicates that the predictions in after tuning some parameters have, on average, smaller absolute differences from the true values, suggesting better accuracy as well.
# - Comparing the R-squared and explained variance score:
#    - The set that is after tuning some parameters has a higher R-squared (0.9503852188123433) compared to the default parameters set (0.9462639638449475). This indicates that the model in after tuning some parameters explains a larger proportion of the variance in the target variable, suggesting a better fit to the data.
#   - Similarly, The set that is after tuning some parameters has a higher explained variance score (0.9510323416922383) compared to he default parameters set (0.9470110304243363), indicating that the model after tuning some parameters captures a larger portion of the underlying patterns in the data.
#   
#   
# Based on these comparisons, after tuning some parameters generally outperforms the model with default parameters across all evaluation metrics. It has lower MSE and MAE values, indicating better accuracy, as well as higher R-squared and explained variance scores, indicating a better fit to the data. Therefore, After tuning some parameters is considered better than using the default parameters in terms of model performance.

# ### Third Model: Linear Regression

# ### Fourth Model: Random Forest

# In[45]:


# DEFAULT PARAMETERS
# Calculate mean squared error (MSE)
mse = mean_squared_error(y_test_rf, y_pred_rf)
print("Mean Squared Error (MSE):", mse)

# Calculate mean absolute error (MAE)
mae = mean_absolute_error(y_test_rf, y_pred_rf)
print("Mean Absolute Error (MAE):", mae)

# Calculate R-squared (coefficient of determination)
r2 = r2_score(y_test_rf, y_pred_rf)
print("R-squared (R2):", r2)

explained_variance = explained_variance_score(y_test_rf, y_pred_rf)
print("Explained Variance Score:", explained_variance)


# In[46]:


#BEST PARAMETERS
# Calculate mean squared error (MSE)
mse = mean_squared_error(y_test1_rf, y_pred1_rf)
print("Mean Squared Error (MSE):", mse)

# Calculate mean absolute error (MAE)
mae = mean_absolute_error(y_test1_rf, y_pred1_rf)
print("Mean Absolute Error (MAE):", mae)

# Calculate R-squared (coefficient of determination)
r2 = r2_score(y_test1_rf, y_pred1_rf)
print("R-squared (R2):", r2)

explained_variance = explained_variance_score(y_test1_rf, y_pred1_rf)
print("Explained Variance Score:", explained_variance)


# When comparing these two sets, lower values of MSE and MAE indicate better performance, while higher values of R-squared and explained variance score indicate better fit.
# - Comparing the MSE and MAE:
#   - The set that is after tuning some parameters has a lower MSE (126953064.55964723) compared to the default parameters set (128081894.79487355). This suggests that the predictions after tuning some parameters have, on average, smaller squared differences from the true values, indicating better accuracy.
#   - The set that is after tuning some parameters also has a lower MAE (6613.836068634177) compared to the default parameters set (6470.652577898576). This indicates that the predictions in after tuning some parameters have, on average, smaller absolute differences from the true values, suggesting better accuracy as well.
# - Comparing the R-squared and explained variance score:
#   - The set that is after tuning some parameters has a higher R-squared (0.9405775254008033) compared to the default parameters set (0.9400491578012324). This indicates that the model in after tuning some parameters explains a larger proportion of the variance in the target variable, suggesting a better fit to the data.
#   - Similarly, The set that is after tuning some parameters has a higher explained variance score (0.9406141610992932) compared to he default parameters set (0.9403728223393665), indicating that the model after tuning some parameters captures a larger portion of the underlying patterns in the data.
#   
#   
# Based on these comparisons, after tuning some parameters generally outperforms the model with default parameters across all evaluation metrics. It has lower MSE and MAE values, indicating better accuracy, as well as higher R-squared and explained variance scores, indicating a better fit to the data. Therefore, After tuning some parameters is considered better than using the default parameters in terms of model performance.
