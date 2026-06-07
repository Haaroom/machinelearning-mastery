from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score,recall_score,precision_score,confusion_matrix,classification_report,accuracy_score

data=load_breast_cancer()
X=data.data
y=data.target
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,stratify=y,test_size=0.2)
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
model1=KNeighborsClassifier(n_neighbors=3)
model2=SVC(kernel='rbf')
model3=GaussianNB()
model1.fit(X_train,y_train)
model2.fit(X_train,y_train)
model3.fit(X_train,y_train)
y_pred1=model1.predict(X_test)
y_pred2=model2.predict(X_test)
y_pred3=model3.predict(X_test)
def evaluate(y_true, y_pred):
    print("Accuracy ", accuracy_score(y_true, y_pred))
    print("Precision ",precision_score(y_true, y_pred))
    print("Recall ", recall_score(y_true, y_pred))
    print("F1 ", f1_score(y_true, y_pred))
    print("confusion matrix ", confusion_matrix(y_true, y_pred))
    print("classification_report ",classification_report(y_true, y_pred))
evaluate(y_train,y_pred1)
evaluate(y_train,y_pred2)
evaluate(y_train,y_pred3)







