# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 10:40:33 2019

@author: kumar
"""
#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2] #must be a matrix and not a vector for scaling
y = dataset.iloc[:,2:]

#Since the dataset is small, we do not split it into testing and training
#Since SVR does take care of data scaling, scale the data
from sklearn.preprocessing import StandardScaler
scale_X = StandardScaler()
scale_y = StandardScaler()

scaled_X = scale_X.fit_transform(X)
scaled_y = scale_y.fit_transform(y).flatten()

#import algorithm
from sklearn.svm import SVR
regressor = SVR(epsilon = 0.1)
regressor.fit(scaled_X, scaled_y)
y_pred = scale_y.inverse_transform(regressor.predict(scaled_X))
prediction = np.round(scale_y.inverse_transform(regressor.predict(scale_X.transform(6.5))).item())

#visualization
plt.scatter(6.5,160000,color='green',s = 150)
plt.scatter(6.5, scale_y.inverse_transform(regressor.predict(scale_X.transform(6.5))),s=30,color = 'cyan')
plt.legend(['Actual Salary: 160000','Predicted Salary: '+str(prediction)])
plt.scatter(X,y,color='red')
plt.plot(X,y_pred,color='blue')
plt.xlabel("Position (level)")
plt.ylabel("Salary")
plt.title('Truth or Bluff')
plt.show()

