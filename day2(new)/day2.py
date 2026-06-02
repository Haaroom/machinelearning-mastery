import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import  RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score,classification_report,roc_auc_score
pd.set_option('display.max_columns', None)
df=pd.read_csv(r'H:\PROGRAMMING\HAAROON\MACHINELEARNING\day2(new)\creditcard.csv')
# df.head()
# df.info()
# df.describe()
# df.isnull().sum()
X=df.drop('Class',axis=1)
y=df['Class']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test = sc.transform(X_test)
model1=LogisticRegression(class_weight='balanced')
model2=RandomForestClassifier(class_weight='balanced')
model3=XGBClassifier()
model4=DecisionTreeClassifier()
model1.fit(X_train,y_train)
model2.fit(X_train,y_train)
model3.fit(X_train,y_train)
model4.fit(X_train,y_train)
print("Logistic Regression")
y_pred1=model1.predict(X_test)
print(confusion_matrix(y_test,y_pred1))
print("Precision Score:",precision_score(y_test,y_pred1))
print("Recall Score:",recall_score(y_test,y_pred1))
print("F1 Score:",f1_score(y_test,y_pred1))
print(classification_report(y_test,y_pred1))
print("ROC AUC Score:",roc_auc_score(y_test,y_pred1))
print("Random Forest Classifier")
y_pred2=model2.predict(X_test)
print(confusion_matrix(y_test,y_pred2))
print("Precision Score:",precision_score(y_test,y_pred2))
print("Recall Score:",recall_score(y_test,y_pred2))
print("F1 Score:",f1_score(y_test,y_pred2))
print(classification_report(y_test,y_pred2))
print("ROC AUC Score:",roc_auc_score(y_test,y_pred2))
print("XGB Classifier")
y_pred3=model3.predict(X_test)
print(confusion_matrix(y_test,y_pred3))
print("Precision Score:",precision_score(y_test,y_pred3))
print("Recall Score:",recall_score(y_test,y_pred3))
print("F1 Score:",f1_score(y_test,y_pred3))
print(classification_report(y_test,y_pred3))
print("ROC AUC Score:",roc_auc_score(y_test,y_pred3))
print("Decision Tree Classifier")
y_pred4=model4.predict(X_test)  
print(confusion_matrix(y_test,y_pred4))
print("Precision Score:",precision_score(y_test,y_pred4))
print("Recall Score:",recall_score(y_test,y_pred4))
print("F1 Score:",f1_score(y_test,y_pred4))
print(classification_report(y_test,y_pred4))
print("ROC AUC Score:",roc_auc_score(y_test,y_pred4))
#feature importance 
feature_importance=model2.feature_importances_
feature_importance_df=pd.DataFrame({'Feature':X.columns,'Importance':feature_importance})
#predict_proba's of models 
y_prob1=model1.predict_proba(X_test)
y_prob2=model2.predict_proba(X_test)
y_prob3=model3.predict_proba(X_test)
y_prob4=model4.predict_proba(X_test)
#maybe ill try accuracies and stuffs with this later 
#now threshold tuning 
threshold=0.5
y_pred_threshold1=(y_prob1[:,1]>=threshold).astype(int)
y_pred_threshold2=(y_prob2[:,1]>=threshold).astype(int)
y_pred_threshold3=(y_prob3[:,1]>=threshold).astype(int)
y_pred_threshold4=(y_prob4[:,1]>=threshold).astype(int)




