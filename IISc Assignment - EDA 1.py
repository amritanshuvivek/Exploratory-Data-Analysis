#!/usr/bin/env python
# coding: utf-8

# In[13]:


# Import necessary libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[14]:


# Load the dataset

df = pd.read_csv('buildings.csv')


# In[15]:


# Print Descriptive Statistics

print(df.describe())


# In[16]:


df.head()


# In[17]:


df.isnull()


# In[18]:


# Create a bar chart with the number of missing values in each column

missing_values = df.isnull().sum()
plt.figure(figsize=(10, 5))
missing_values.plot(kind='bar')
plt.title('Number of Missing Values in Each Column')
plt.xlabel('Columns')
plt.ylabel('Missing Values')
plt.show()


# In[19]:


# Drop columns containing more than 50% missing values

df = df.dropna(thresh=(0.5) * len(df), axis=1)


# In[20]:


df


# In[23]:


# Plot the distribution of categorical variables

categorical_columns = df.select_dtypes(include='object').columns

for column in categorical_columns:
    value_counts = df[column].value_counts()
    value_counts.plot(kind='bar', figsize=(10, 5), color='salmon')
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.show()


# In[25]:


#Conduct univariable analysis - find outliers in each column and check for normal distribution

from scipy import stats

# Univariate Analysis
def univariate_analysis(column):
    # Histogram
    plt.figure(figsize=(10, 5))
    sns.histplot(df[column], kde=True)
    plt.title(f'Univariate Analysis - {column}')
    plt.show()

# Iteration through columns
for column in df.columns:
    univariate_analysis(column)


# In[26]:


# Outliers in each column

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
print(IQR)


# In[27]:


# Pairwire Correlations

correlation_matrix = df.corr()
plt.figure(figsize=(10, 5))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Pairwise Correlation Matrix')
plt.show()


# In[28]:


# Numerical Data Types 
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns


# In[29]:


# Group Variables
grouped_variables = [numerical_columns[i:i+3] for i in range(0, len(numerical_columns), 3)]


# In[30]:


# Box plots for each Group
for group in grouped_variables:
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df[group])
    plt.title('Box plots for {}'.format(', '.join(group)))
    plt.show()


# In[31]:


# Multivariate Analysis to find duplicated columns
duplicated_columns = df.T.duplicated().any()

if duplicated_columns:
    print("Duplicated Columns:")
    print(df.loc[:, duplicated_columns].columns)
else:
    print("No Duplicated Columns found.")


# In[ ]:




