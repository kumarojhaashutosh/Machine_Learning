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
data = pd.read_csv('Salary_Data.csv')

X = data.iloc[:,:-1]
y = data.iloc[:,1]

#splitting the dataset for training and testing:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)

#importing algorithm and training:
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
m = regressor.coef_
c = regressor.intercept_

#visualization:
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,m*X_train+c,color='blue')
plt.ylabel('Y (Actual Salary)')
plt.xlabel('X (Experience)')
plt.legend()
plt.title('Regression Line (Training Set)')
plt.show()

plt.scatter(X_test,y_test,color='red')
plt.plot(X_test,m*X_test+c,color='blue')
plt.ylabel('Y (Predicted Salary)')
plt.xlabel('X (Experience)')
plt.legend()
plt.title('Regression Line (Test Set)')
plt.show()

#analyze prediction accuracy:
from sklearn.metrics import r2_score
print('R2_Score: {}'.format(r2_score(y_test,y_pred)))