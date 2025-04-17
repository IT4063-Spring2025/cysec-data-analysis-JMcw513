#!/usr/bin/env python
# coding: utf-8

# ## In-Class Activity - Cyber Security Data Analysis 
# This notebook will guide you through the process of analyzing a cyber security dataset. Follow the TODO tasks to complete the assignment.
# 

# # Step 1: Importing the required libraries
# 
# TODO: Import the necessary libraries for data analysis and visualization.

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
pd.options.display.max_rows = 999
warnings.filterwarnings("ignore")


# # Step 2: Loading the dataset
# 
# TODO: Load the given dataset.

# In[3]:


df = "./Data/CySecData.csv"


# # Step 3: Display the first few rows of the dataset
# TODO: Import the necessary libraries for data analysis and visualization.

# In[4]:


import pandas as pd
df = pd.read_csv("./Data/CySecData.csv")
df.head(5)


# # Step 4: Initial info on the dataset.
# 
# TODO: Provide a summary of the dataset.

# In[4]:


df.info()


# # Step 5: Creating dummy variables
# TODO: Create dummy variables for the categorical columns except for the label column "class".

# In[5]:


dfDummies = pd.get_dummies(df, columns=["class"], drop_first=True)
dfDummies.head(5)



# # Step 6: Dropping the target column
# TODO: Drop the target column 'class' from the dataset.

# In[6]:


dfDummies.drop("class_normal", axis=1, inplace=True)
dfDummies.head(5)


# # Step 7: Importing the Standard Scaler
# TODO: Import the `StandardScaler` from `sklearn.preprocessing`.

# In[7]:


from sklearn.preprocessing import StandardScaler
dfDummies.columns


# # Step 8: Scaling the dataset
# TODO: Scale the dataset using the `StandardScaler`.

# In[8]:


numeric_dfDummies = dfDummies.select_dtypes(include=['number'])

scaler = StandardScaler()
dfNormalized = scaler.fit_transform(numeric_dfDummies)
dfNormalized = pd.DataFrame(dfNormalized, columns=numeric_dfDummies.columns)
dfNormalized.head(5)


# # Step 9: Splitting the dataset
# TODO: Split the dataset into features (X) and target (y).

# In[39]:


from sklearn.model_selection import train_test_split
X = dfNormalized
y = df['class']



# # Step 10: Importing the required libraries for the models
# TODO: Import the necessary libraries for model training and evaluation.

# In[31]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

dfDummies.columns


# # Step 11: Defining the models (Logistic Regression, Support Vector Machine, Random Forest)
# TODO: Define the models to be evaluated.

# In[13]:


models = []
models.append(('LR', LogisticRegression()))
models.append(('SVM', SVC()))
models.append(('RandomForestClassifier', RandomForestClassifier()))




# # Step 12: Evaluating the models
# TODO: Evaluate the models using 10 fold cross-validation and display the mean and standard deviation of the accuracy.
# Hint: Use Kfold cross validation and a loop

# In[41]:


#Evaluate the models using 10 fold cross-validation
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

results = []
names = []
for name, model in models:
    kfold = cross_val_score(model, X, y, cv=10, scoring='accuracy')
    results.append(kfold)
    names.append(name)
    msg = "%s: %f (%f)" % (name, kfold.mean(), kfold.std())
    print(msg)



# # Step 13: Converting the notebook to a script
# TODO: Convert the notebook to a script using the `nbconvert` command.

# In[43]:


#get_ipython().system('jupyter nbconvert --to python notebook.ipynb')

