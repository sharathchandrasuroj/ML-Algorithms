# ML-Alorithms
✅Getting the dataset
✅Importing libraries
✅Importing datasets
✅Finding Missing Data                       
✅Encoding Categorical Data
✅Splitting dataset into training and test set
✅Feature scaling


✅ **Getting the dataset**
To create a machine learning model, the first thing we required is a dataset as a machine learning model completely works on data. The collected data for a particular problem in a proper format is known as the dataset.CSV(Comma-Separated Values) HTML and XLSV files

✅ **Importing libraries**
✔Numpy: Numpy Python library is used for including any type of mathematical operation in the code
✔Matplotlib:#pyplot  matplotlib, which is a Python 2D plotting library
✔Pandas: used for importing and managing the datasets. It is an open-source data manipulation and analysis library
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

✅ **Importing datasets**
First set working directory
◼ read_csv function:  used to read a csv file and performs various operations on it. Using this function, we can read a csv file locally as well as through an URL.
data_set= pd.read_csv('Dataset.csv')  

◼ Extracting independent variable:

To extract an independent variable, we will use iloc[ ] method of Pandas library. It is used to extract the required rows and columns from the dataset.
x= data_set.iloc[:,:-1].values  

◼ Extracting dependent variable:

To extract dependent variables, again, we will use Pandas .iloc[] method.
y= data_set.iloc[:,3].values 


✅**Finding Missing Data** 
◼ By deleting the particular row: delete the specific row or column which consists of null values. But this way is not so efficient and removing data may lead to loss of information which will not give the accurate output.

◼ By calculating the mean: calculate the mean of that column or row which contains any missing value and will put it on the place of missing value.

To handle missing values, use Scikit-learn library 
#handling missing data (Replacing missing data with the mean value)  
from sklearn.preprocessing import Imputer  
imputer= Imputer(missing_values ='NaN', strategy='mean', axis = 0)  

#Fitting imputer object to the independent variables x.   
imputerimputer= imputer.fit(x[:, ])  
#Replacing missing data with the calculated mean value  
x[:, ]= imputer.transform(x[:, ])  


✅**Encoding Categorical Data**
use LabelEncoder() class from preprocessing library.

***Catgorical data***
ML only alows numeric values
To delete categorial values we will Label Encoder


***Dummy Variables:***

Dummy variables are those variables which have values 0 or 1. The 1 value gives the presence of that variable in a particular column, and rest variables become 0. With dummy encoding, we will have a number of columns equal to the number of categories.

OneHotEncoder class of preprocessing library.
from sklearn.preprocessing import LabelEncoder, OneHotEncoder  
label_encoder_x= LabelEncoder()  
x[:, 0]= label_encoder_x.fit_transform(x[:, 0])  
#Encoding for dummy variables  
onehot_encoder= OneHotEncoder(categorical_features= [0])    
x= onehot_encoder.fit_transform(x).toarray()  

✅**Splitting dataset into training and test set**
**Training Set:** A subset of dataset to train the machine learning model, and we already know the output.

**Test set:** A subset of dataset to test the machine learning model, and by using the test set, model predicts the output.

For splitting the dataset, we will use the below lines of code:

from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.2, random_state=0)  


✅**Feature scaling**
It is a technique to standardize the independent variables of the dataset in a specific range. In feature scaling, we put our variables in the same range and in the same scale so that no any variable dominate the other variable.
 A machine learning model is based on **Euclidean distance**, and if we do not scale the variable, then it will cause some issue in our machine learning model.
 
 
 Standardization
X` = (X-mean(X))/ Std Deviation
`
Normalization

X` = (X-min(X)) / max(X)-min(X)
`

from sklearn.preprocessing import StandardScaler  
Now, we will create the object of **StandardScaler** class for independent variables or features. And then we will fit and transform the training dataset.

st_x= StandardScaler()  
x_train= st_x.fit_transform(x_train)  

For test dataset, we will directly apply **transform()** function instead of **fit_transform()** because it is already done in training set.
x_test= st_x.transform(x_test)  




********Code*********
# importing libraries  
import numpy as nm  
import matplotlib.pyplot as mtp  
import pandas as pd  
  
# importing datasets  
data_set= pd.read_csv('Dataset.csv')  
  
# Extracting Independent Variable  
x= data_set.iloc[:, :-1].values  
  
# Extracting Dependent variable  
y= data_set.iloc[:, 3].values  
  
# handling missing data(Replacing missing data with the mean value)  
from sklearn.preprocessing import Imputer  
imputer= Imputer(missing_values ='NaN', strategy='mean', axis = 0)  
  
# Fitting imputer object to the independent varibles x.   
imputerimputer= imputer.fit(x[:, 1:3])  
  
# Replacing missing data with the calculated mean value  
x[:, 1:3]= imputer.transform(x[:, 1:3])  
  
# for categorical Variable  
from sklearn.preprocessing import LabelEncoder, OneHotEncoder  
label_encoder_x= LabelEncoder()  
x[:, 0]= label_encoder_x.fit_transform(x[:, 0])  
  
# Encoding for dummy variables  
onehot_encoder= OneHotEncoder(categorical_features= [0])    
x= onehot_encoder.fit_transform(x).toarray()  
  
# encoding for purchased variable  
labelencoder_y= LabelEncoder()  
y= labelencoder_y.fit_transform(y)  
  
# Splitting the dataset into training and test set.  
from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.2, random_state=0)  
  
# Feature Scaling of datasets  
from sklearn.preprocessing import StandardScaler  
st_x= StandardScaler()  
x_train= st_x.fit_transform(x_train)  
x_test= st_x.transform(x_test)
