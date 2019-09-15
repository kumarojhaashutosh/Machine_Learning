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

#splitting the dataset for training and testing:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state=0)

# Scaling Data:
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#importing algorithm and training:
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)

# check accuracy of prediction
from sklearn.metrics import r2_score
print('R2_Score: {}'.format(r2_score(y_test,y_pred)))

