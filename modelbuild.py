import pandas as pd
import numpy as np
#import csv
import matplotlib.pyplot as plt
import seaborn as sns

#read the datset
sales=pd.read_csv("C:\\Users\\Donatus\\Documents\\SalesProjectDeploy\\sales_train.csv")
#dropping duplicates
new_sales=sales.drop_duplicates()
new_sales=new_sales[new_sales['item_price'] < 40000]
new_sales.date=pd.to_datetime(new_sales['date'])
neww_sales=new_sales.drop('date', axis=1)
#Modelling part

#ML models
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import cross_val_score
X=neww_sales.drop('item_cnt_day',axis=1) 
y=neww_sales['item_cnt_day']
#split the data
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.4, random_state=0)

regressor = LinearRegression()  
regressor.fit(X_train, y_train)
#Intercept
print(regressor.intercept_)
#Slope
print(regressor.coef_)
#Predict
pred1 = regressor.predict(X_test)

#actual value and predicted value
newpredictions = pd.DataFrame({'Actual': y_test, 'Predicted': pred1})
newpredictions

# saving the model 
import pickle 
pickle_out = open("classifier.pkl", mode = "wb") 
pickle.dump(regressor, pickle_out) 
pickle_out.close()

# loading the trained model
pickle_in = open('classifier.pkl', 'rb') 
classifier = pickle.load(pickle_in)