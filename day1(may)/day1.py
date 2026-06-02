import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import  StandardScaler
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,recall_score,precision_score
import numpy as np
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
#load data set 
df=pd.read_csv(r'H:\PROGRAMMING\HAAROON\MACHINELEARNING\day1(may)\train.csv')
# print(df.head())
# print(df.isnull().sum())
# print(df.describe())
# Handle missing values using SimpleImputer
imputer = SimpleImputer(strategy='mean',missing_values=np.nan)
df['Age']=imputer.fit_transform(df[['Age']])
df['Embarked'] = df['Embarked'].ffill()
df_encoded = pd.get_dummies(df,columns=['Sex','Embarked'],drop_first=True)
#splitting X and Y 
X  = df_encoded.drop(['Survived','Name','Ticket','Cabin'],axis=1)
y = df_encoded['Survived']
#spliting data into train and test
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
#scaling data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
#creating models 
model1 = LogisticRegression()
model2= DecisionTreeClassifier(max_depth=5)
model3 = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)
#fitting models
model1.fit(X_train_scaled,y_train)
model2.fit(X_train_scaled,y_train)
model3.fit(X_train_scaled,y_train)  
#predicting results using test data 
y_pred1 = model1.predict(X_test_scaled)
y_pred2 = model2.predict(X_test_scaled)
y_pred3 = model3.predict(X_test_scaled)
#evaluating models
print("model 1 logistic regression")
print("accuracy score:",accuracy_score(y_test,y_pred1))
print("confusion matrix:",confusion_matrix(y_test,y_pred1))
print("f1 score:",f1_score(y_test,y_pred1))
print("recall score:",recall_score(y_test,y_pred1))
print("-----------------------------------")
print("model 2 decision tree")
print("accuracy score:",accuracy_score(y_test,y_pred2))
print("confusion matrix:",confusion_matrix(y_test,y_pred2))
print("f1 score:",f1_score(y_test,y_pred2))
print("recall score:",recall_score(y_test,y_pred2))
print("-----------------------------------")
print("model 3 random forest")
print("accuracy score:",accuracy_score(y_test,y_pred3))
print("confusion matrix:",confusion_matrix(y_test,y_pred3))
print("f1 score:",f1_score(y_test,y_pred3))
print("recall score:",recall_score(y_test,y_pred3))
