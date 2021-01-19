#Q- To explore supervised Machine learning 
#importing libraries 
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv('http://bit.ly/w-data')
X = dataset.iloc[:,:-1].values
Y= dataset.iloc[:,1].values #Y is dependant vector. 

#splitting into training & testing
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=1/3, random_state=0)

#fitting simple linear regression to the training set
from sklearn.linear_model import LinearRegression 
regressor= LinearRegression()
regressor.fit(X_train , Y_train)

#predicting test set result 
Y_pred= regressor.predict(X_test)

#Visualizing training set result
plt.scatter(X_train , Y_train , color='red')
plt.plot(X_train , regressor.predict(X_train), color='blue')
plt.title('hours vs score (training set)')
plt.xlabel('hours')
plt.ylabel('score')
plt.show()

#Visualizing test set result
plt.scatter(X_test , Y_test , color='red')
plt.plot(X_train , regressor.predict(X_train), color='blue')
plt.title('hours vs score (test set)')
plt.xlabel('hours')
plt.ylabel('score')
plt.show()
