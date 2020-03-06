# -*- coding: utf-8 -*-
###
Simple Linear Regression equation:         y = b0+b1x         
Multiple Linear Regression equation:         y= b0+b1x+ b2x2+ b3x3+....+ bnxn         

Polynomial Regression equation:         y= b0+b1x + b2x2+ b3x3+....+ bnxn         


Steps for Polynomial Regression:

Data Pre-processing
Build a Linear Regression model and fit it to the dataset
Build a Polynomial Regression model and fit it to the dataset
Visualize the result for Linear Regression and Polynomial Regression model.
Predicting the output.


###

#Data Pre-processing
# importing libraries  
import numpy as nm  
import matplotlib.pyplot as plt  
import pandas as pd  
  
#importing datasets  
data_set= pd.read_csv('Position_Salaries.csv')  
  
#Extracting Independent and dependent Variable  
x= data_set.iloc[:, 1:2].values  
y= data_set.iloc[:, 2].values  

#Fitting the Linear Regression to the dataset  
from sklearn.linear_model import LinearRegression  
lin_regs= LinearRegression()  
lin_regs.fit(x,y)

 #Fitting the Polynomial regression to the dataset  
from sklearn.preprocessing import PolynomialFeatures  
poly_regs= PolynomialFeatures(degree= 2)  
x_poly= poly_regs.fit_transform(x)  
lin_reg_2 =LinearRegression()  
lin_reg_2.fit(x_poly, y)  

#Visulaizing the result for Linear Regression model  
plt.scatter(x,y,color="blue")  
plt.plot(x,lin_regs.predict(x), color="red")  
plt.title("Bluff detection model(Linear Regression)")  
plt.xlabel("Position Levels")  
plt.ylabel("Salary")  
plt.show() 

#Visulaizing the result for Polynomial Regression  
plt.scatter(x,y,color="blue")  
plt.plot(x, lin_reg_2.predict(poly_regs.fit_transform(x)), color="red")  
plt.title("Bluff detection model(Polynomial Regression)")  
plt.xlabel("Position Levels")  
plt.ylabel("Salary")  
plt.show()  

#Predicting the final result with the Linear Regression model
lin_pred = lin_regs.predict([[6.5]])  
print(lin_pred)  

#Predicting the final result with the Polynomial Regression model
poly_pred = lin_reg_2.predict(poly_regs.fit_transform([[6.5]]))  
print(poly_pred)  

