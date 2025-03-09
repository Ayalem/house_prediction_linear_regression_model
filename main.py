import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy

from skimage.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


training=pd.read_csv("train.csv")
features=["LotFrontage","YrSold","LotArea","TotRmsAbvGrd","GrLivArea","GarageCars","1stFlrSF"]
#checking for missing values:
print(training.isna().sum())
#lot frontage has 259 missing values let's fill them with the mean:
training['LotFrontage']=training['LotFrontage'].fillna(training['LotFrontage'].mean())



#load dataset
x=np.array(training[features])
y=np.array(training.iloc[:,-1])
print(x[:5])
print(y[:5])


#1.splitting data set:
#training set size:60%:
x_train,x_,y_train,y_=train_test_split(x,y,test_size=0.4,random_state=1)
#split the remaining data into 2 subsets :cross-validation set and the  testing set:
x_cv,x_test,y_cv,y_test=train_test_split(x_,y_,test_size=0.5,random_state=1)
del x_,y_
print("x_train shape:", x_train.shape)
print("x_cv shape:", x_cv.shape)
print("x_test shape:", x_test.shape)
print("y_train shape:", y_train.shape)
print("y_cv shape:", y_cv.shape)
print("y_test shape:", y_test.shape)
#2.adding polynomial features:
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.linear_model import SGDRegressor
train_mses=[]
cv_mses=[]
models=[]
polys=[]
scalers=[]
for degree in range(1,11):
    poly=PolynomialFeatures(degree,include_bias=False)
    polys.append(poly)
    x_train_mapped=poly.fit_transform(x_train)
    x_cv_mapped=poly.transform(x_cv)
    #scaling /normalizing:
    scaler=StandardScaler()
    scalers.append(scaler)
    x_train_mapped_scaled=scaler.fit_transform(x_train_mapped)
    x_cv_mapped_scaled=scaler.transform(x_cv_mapped)
    #train our model:
    model=SGDRegressor()
    model.fit(x_train_mapped_scaled,y_train)
    models.append(model)
    #compute training mse:
    yhat=model.predict(x_train_mapped_scaled)
    train_mse=mean_squared_error(yhat,y_train)
    train_mses.append(train_mse)
    #test with the cross-validation set:
    yhat=model.predict(x_cv_mapped_scaled)
    cv_mse=mean_squared_error(yhat,y_cv)
    cv_mses.append(cv_mse)
#choose the best model based on the mse  of the cv set:
plt.plot(cv_mses,label="cv mse")
plt.plot(train_mses,label="train mse")
plt.legend()
plt.show()
degree=np.argmin(cv_mses)+1#since index starts at 0
print("the best model has degree of:",degree)
#let's test it with cv set:
x_test_mapped=polys[degree-1].transform(x_test)
x_test_mapped_scaled=scalers[degree-1].transform(x_test_mapped)
yhat=models[degree-1].predict(x_test_mapped_scaled)
test_mse=mean_squared_error(y_test,yhat)
print("test set mse:",test_mse)
best_model=models[degree-1]
import joblib

# Assuming 'model' is your trained model
joblib.dump(best_model, 'house_price_model.pkl')  # Save the model
best_poly = polys[degree - 1]
joblib.dump(best_poly, 'polynomial_features.pkl')

# Save the scaler used to scale the features
best_scaler = scalers[degree - 1]
joblib.dump(best_scaler, 'scaler.pkl')




