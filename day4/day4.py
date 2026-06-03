import pandas as pd 
from sklearn.linear_model import LinearRegression,LassoCV,RidgeCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
import os 
pd.set_option('display.max_columns', None)
df1=pd.read_csv(r"H:\PROGRAMMING\HAAROON\MACHINELEARNING\day4(new)\train.csv")
df2=pd.read_csv(r'H:\PROGRAMMING\HAAROON\MACHINELEARNING\day4(new)\test.csv')
#eda analysis 
df1.info()
df2.info()
print(df1.head())
print(df2.head())
print(df1.isnull().sum())
print(df2.isnull().sum()) 
print(df1.info())
df1=df1.drop(columns=['Alley','PoolQC','Fence','MiscFeature'])
df2=df2.drop(columns=['Alley','PoolQC','Fence','MiscFeature'])  
df1.fillna(df1.mode().iloc[0],inplace=True)
df2.fillna(df2.mode().iloc[0],inplace=True)
df1=pd.get_dummies(df1,drop_first=True)
df2=pd.get_dummies(df2,drop_first=True)
print("sale price:",df1['SalePrice'])
#dividing into X and y 
X_train=df1.drop(columns=['Id','SalePrice'])
y_train=df1['SalePrice']
X_test=df2.drop(columns=['Id','SalePrice'])
y_test=df2['SalePrice']
#models 
model1=LinearRegression()
model2=LassoCV(alphas=[0.1,1.0,10.0],cv=5)
model3=RidgeCV(alphas=[0.1,1.0,10.0],cv=5)
model4=RandomForestRegressor(max_depth=10,n_estimators=100,random_state=42)
model5=XGBRegressor(max_depth=10,n_estimators=100,random_state=42)
#scaling the data 
sc=StandardScaler()
X_train_scaled=sc.fit_transform(X_train)
X_test_scaled=sc.transform(X_test)
#fitting the models 
model1.fit(X_train_scaled,y_train)
model2.fit(X_train_scaled,y_train)
model3.fit(X_train_scaled,y_train)
model4.fit(X_train,y_train)
model5.fit(X_train,y_train)
#predictions 
y_pred1=model1.predict(X_test_scaled)
y_pred2=model2.predict(X_test_scaled)
y_pred3=model3.predict(X_test_scaled)
y_pred4=model4.predict(X_test)
y_pred5=model5.predict(X_test)
#evaluating the models
print("Linear Regression:")
print("MSE:",mean_squared_error(y_test,y_pred1))
print("R2 Score:",r2_score(y_test,y_pred1))
print("MAE:",mean_absolute_error(y_test,y_pred1))
print("\nLasso Regression:")
print("MSE:",mean_squared_error(y_test,y_pred2))
print("R2 Score:",r2_score(y_test,y_pred2))
print("MAE:",mean_absolute_error(y_test,y_pred2))
print("\nRidge Regression:")
print("MSE:",mean_squared_error(y_test,y_pred3))
print("R2 Score:",r2_score(y_test,y_pred3))
print("MAE:",mean_absolute_error(y_test,y_pred3))
print("\nRandom Forest Regressor:")
print("MSE:",mean_squared_error(y_test,y_pred4))
print("R2 Score:",r2_score(y_test,y_pred4))
print("MAE:",mean_absolute_error(y_test,y_pred4))
print("\nXGBoost Regressor:")
print("MSE:",mean_squared_error(y_test,y_pred5))
print("R2 Score:",r2_score(y_test,y_pred5))
print("MAE:",mean_absolute_error(y_test,y_pred5))