import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,precision_score,recall_score,f1_score

pd.set_option('display.max_columns', None)
df=pd.read_csv(r'H:\PROGRAMMING\HAAROON\MACHINELEARNING\day3(new)\WA_Fn-UseC_-Telco-Customer-Churn.csv')
#eda 
# print(df.head())
# df.info()
# df.describe()
#print(df.isnull().sum())
df=df.convert_dtypes()
#encoding
df['TotalCharges']=pd.to_numeric(df['TotalCharges'],errors='coerce')
df = pd.get_dummies(df, drop_first=True)
X=df.drop('Churn',axis=1)
y=df['Churn']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
#scaling 
sc=StandardScaler()
X_train_scaled=sc.fit_transform(X_train)
X_test_scaled=sc.transform(X_test)
#models
model1=XGBClassifier()
model2=LogisticRegression(class_weight='balanced')
model3=RandomForestClassifier(max_depth=5)
model4=SVC()
model5=KNeighborsClassifier(n_neighbors=5)
#fitting models 
model1.fit(X_train,y_train)
model2.fit(X_train_scaled,y_train)
model3.fit(X_train,y_train)
model4.fit(X_train_scaled,y_train)
model5.fit(X_train_scaled,y_train)
#predicting with models
y_pred1=model1.predict(X_test)
y_pred2=model2.predict(X_test_scaled)
y_pred3=model3.predict(X_test)
y_pred4=model4.predict(X_test_scaled)
y_pred5=model5.predict(X_test_scaled)
#feature importance 
randomforest_importances=model3.feature_importances_
xgboost_importances=model1.feature_importances_
feature_importance_df1=pd.DataFrame({'Feature':X.columns,'Importance':randomforest_importances}).sort_values(by='Importance',ascending=False)
feature_importance_df2=pd.DataFrame({'Feature':X.columns,'Importance':xgboost_importances}).sort_values(by='Importance',ascending=False)    
#evaluation
print("XGBoost ")
print("classification report : ",classification_report(y_test,y_pred1))
print("confusion matrix : ",confusion_matrix(y_test,y_pred1))
print("accuracy score : ",accuracy_score(y_test,y_pred1))
print("precision score : ",precision_score(y_test,y_pred1))
print("recall score : ",recall_score(y_test,y_pred1))
print("f1 score : ",f1_score(y_test,y_pred1))
print("Logistic Regression ")
print("classification report : ",classification_report(y_test,y_pred2))
print("confusion matrix : ",confusion_matrix(y_test,y_pred2))
print("accuracy score : ",accuracy_score(y_test,y_pred2))
print("precision score : ",precision_score(y_test,y_pred2))
print("recall score : ",recall_score(y_test,y_pred2))
print("f1 score : ",f1_score(y_test,y_pred2))
print("Random Forest ")
print("classification report : ",classification_report(y_test,y_pred3))
print("confusion matrix : ",confusion_matrix(y_test,y_pred3))
print("accuracy score : ",accuracy_score(y_test,y_pred3))
print("precision score : ",precision_score(y_test,y_pred3))
print("recall score : ",recall_score(y_test,y_pred3))
print("f1 score : ",f1_score(y_test,y_pred3))
print("SVC ")
print("classification report : ",classification_report(y_test,y_pred4))
print("confusion matrix : ",confusion_matrix(y_test,y_pred4))
print("accuracy score : ",accuracy_score(y_test,y_pred4))
print("precision score : ",precision_score(y_test,y_pred4))
print("recall score : ",recall_score(y_test,y_pred4))
print("f1 score : ",f1_score(y_test,y_pred4))
print("KNN ")
print("classification report : ",classification_report(y_test,y_pred5))
print("confusion matrix : ",confusion_matrix(y_test,y_pred5))
print("accuracy score : ",accuracy_score(y_test,y_pred5))
print("precision score : ",precision_score(y_test,y_pred5))
print("recall score : ",recall_score(y_test,y_pred5))
print("f1 score : ",f1_score(y_test,y_pred5))
#using perdict_proba with supporting alg 
y_pred_proba1=model1.predict_proba(X_test)[:,1]
y_pred_proba2=model2.predict_proba(X_test_scaled)[:,1]
y_pred_proba3=model3.predict_proba(X_test)[:,1]
y_pred_proba4=model4.decision_function(X_test_scaled)
print("XGBoost predicted probabilities : ",y_pred_proba1)
print("Logistic Regression predicted probabilities : ",y_pred_proba2)
print("Random Forest predicted probabilities : ",y_pred_proba3)
print("SVC decision function values : ",y_pred_proba4)
