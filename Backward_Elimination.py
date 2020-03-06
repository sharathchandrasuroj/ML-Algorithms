# -*- coding: utf-8 -*-

###Steps of Backward Elimination
Below are some main steps which are used to apply backward elimination process:

Step-1: Firstly, We need to select a significance level to stay in the model. (SL=0.05)

Step-2: Fit the complete model with all possible predictors/independent variables.

Step-3: Choose the predictor which has the highest P-value, such that.

If P-value >SL, go to step 4.
Else Finish, and Our model is ready.
Step-4: Remove that predictor.

Step-5: Rebuild and fit the model with the remaining variables.
###

#step1
#Preparation of Backward Elimination
import statsmodels.api as smf  

x = np.append(arr = np.ones((50,1)).astype(int), values=x, axis=1)  

#step2 P > |t|

x_opt=x [:, [0,1,2,3,4,5]]  
regressor_OLS=sm.OLS(endog = y, exog=x_opt).fit()  
regressor_OLS.summary() 

#trail n error
x_opt=x[:, [0,2,3,4,5]]  
regressor_OLS=sm.OLS(endog = y, exog=x_opt).fit()  
regressor_OLS.summary()  

x_opt= x[:, [0,3,4,5]]  
regressor_OLS=sm.OLS(endog = y, exog=x_opt).fit()  
regressor_OLS.summary()  

x_opt=x[:, [0,3,5]]  
regressor_OLS=sm.OLS(endog = y, exog=x_opt).fit()  
regressor_OLS.summary()  

x_opt=x[:, [0,3]]  
regressor_OLS=sm.OLS(endog = y, exog=x_opt).fit()  
regressor_OLS.summary()


#Estimating the performance#

Below is the code for Building Multiple Linear Regression model by only using R&D spend:

# importing libraries  
import numpy as nm  
import matplotlib.pyplot as mtp  
import pandas as pd  
  
#importing datasets  
data_set= pd.read_csv('50_CompList1.csv')  
  
#Extracting Independent and dependent Variable  
x_BE= data_set.iloc[:, :-1].values  
y_BE= data_set.iloc[:, 1].values  
  
  
# Splitting the dataset into training and test set.  
from sklearn.model_selection import train_test_split  
x_BE_train, x_BE_test, y_BE_train, y_BE_test= train_test_split(x_BE, y_BE, test_size= 0.2, random_state=0)  
  
#Fitting the MLR model to the training set:  
from sklearn.linear_model import LinearRegression  
regressor= LinearRegression()  
regressor.fit(nm.array(x_BE_train).reshape(-1,1), y_BE_train)  
  
#Predicting the Test set result;  
y_pred= regressor.predict(x_BE_test)  
  
#Cheking the score  
print('Train Score: ', regressor.score(x_BE_train, y_BE_train))  
print('Test Score: ', regressor.score(x_BE_test, y_BE_test))  