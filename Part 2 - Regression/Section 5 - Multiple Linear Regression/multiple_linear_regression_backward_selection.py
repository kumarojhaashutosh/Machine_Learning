# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 16:08:24 2019

@author: Ashutosh
"""
#importing the libraries:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset:
data = pd.read_csv('50_Startups.csv')

X = data.iloc[:,:-1]
y = data.iloc[:,4]

# Dealing with categorical attribute:
X = pd.get_dummies(X, drop_first = True)

# Prepend the intercepts
X = np.append(np.ones((50,1)).astype(int), values = X, axis = 1)

#splitting the dataset for training and testing:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state=0)

'''# Scaling Data:
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)'''

#Using backward elimination to reduce the number of features:
import statsmodels.formula.api as sm
sl = 0.05 #significance level
features = [i for i in range(X.shape[1])]
X_opt = X_train[:,features]
regressor_ols = sm.OLS(y_train,X_opt).fit()

while(max(regressor_ols.pvalues.values)>sl):
    max_pval_feature_idx = np.argmax(regressor_ols.pvalues.values)
    features.pop(max_pval_feature_idx)
    X_opt = X_train[:,features]
    regressor_ols = sm.OLS(y_train,X_opt).fit()
 
y_pred = regressor_ols.predict(X_test[:,features])

# check accuracy of prediction
from sklearn.metrics import r2_score
print('R2_Score: {}'.format(r2_score(y_test,y_pred)))

