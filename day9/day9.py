from  sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix,f1_score,accuracy_score,recall_score,precision_score
from sklearn.decomposition import PCA
# Load the wine data
X, y = load_wine(return_X_y=True)
#eda 
print(X.shape)
print(y.shape)
#splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#scaling the data 
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
# Train the SVM model
model1 = SVC(kernel='linear', random_state=42, decision_function_shape='ovr')
model2 = SVC(kernel='poly', random_state=42, decision_function_shape='ovr')
model3=SVC(kernel='sigmoid', random_state=42, decision_function_shape='ovr')
model4=SVC(kernel='rbf', random_state=42, decision_function_shape='ovr')
model5 = SVC(kernel='linear', random_state=42, decision_function_shape='ovo')
model6 = SVC(kernel='poly', random_state=42, decision_function_shape='ovo')
model7=SVC(kernel='sigmoid', random_state=42, decision_function_shape='ovo')
model8=SVC(kernel='rbf', random_state=42, decision_function_shape='ovo')

#fitting the models 
model1.fit(X_train, y_train)
model2.fit(X_train, y_train)
model3.fit(X_train, y_train)
model4.fit(X_train, y_train)
model5.fit(X_train, y_train)
model6.fit(X_train, y_train)
model7.fit(X_train, y_train)
model8.fit(X_train, y_train)
# Predicting the test set results
y_pred1 = model1.predict(X_test)
y_pred2 = model2.predict(X_test)
y_pred3 = model3.predict(X_test)
y_pred4 = model4.predict(X_test)
y_pred5 = model5.predict(X_test)
y_pred6 = model6.predict(X_test)
y_pred7 = model7.predict(X_test)
y_pred8 = model8.predict(X_test)
# Evaluating the models
print("Model 1 (Linear, OVR) Accuracy:", accuracy_score(y_test, y_pred1),"\nClassification Report:\n", classification_report(y_test, y_pred1), "\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred1))
print("Model 2 (Polynomial, OVR) Accuracy:", accuracy_score(y_test, y_pred2),"\nClassification Report:\n", classification_report(y_test, y_pred2), "\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred2))
print("Model 3 (Sigmoid, OVR) Accuracy:", accuracy_score(y_test, y_pred3),"\nClassification Report:\n", classification_report(y_test, y_pred3), "\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred3))
print("Model 4 (RBF, OVR) Accuracy:", accuracy_score(y_test, y_pred4),"\nClassification Report:\n", classification_report(y_test, y_pred4), "\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred4))
print("Model 5 (Linear, OVO) Accuracy:", accuracy_score(y_test, y_pred5),"\nClassification Report:\n", classification_report(y_test, y_pred5), "\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred5))
print("Model 6 (Polynomial, OVO) Accuracy:", accuracy_score(y_test, y_pred6),"\nClassification Report:\n", classification_report(y_test, y_pred6), "\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred6))
print("Model 7 (Sigmoid, OVO) Accuracy:", accuracy_score(y_test, y_pred7),"\nClassification Report:\n", classification_report(y_test, y_pred7), "\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred7))
print("Model 8 (RBF, OVO) Accuracy:", accuracy_score(y_test, y_pred8),"\nClassification Report:\n", classification_report(y_test, y_pred8), "\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred8))
# using pca 
pca = PCA(n_components=2)
pca.fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
# Train the SVM model on PCA-transformed data
#fitting the models (pca)
model1.fit(X_train_pca, y_train)
model2.fit(X_train_pca, y_train)
model3.fit(X_train_pca, y_train)
model4.fit(X_train_pca, y_train)
model5.fit(X_train_pca, y_train)
model6.fit(X_train_pca, y_train)
model7.fit(X_train_pca, y_train)
model8.fit(X_train_pca, y_train)
# Predicting the test set results (pca)
y_pred1 = model1.predict(X_test_pca)
y_pred2 = model2.predict(X_test_pca)
y_pred3 = model3.predict(X_test_pca)
y_pred4 = model4.predict(X_test_pca)
y_pred5 = model5.predict(X_test_pca)
y_pred6 = model6.predict(X_test_pca)
y_pred7 = model7.predict(X_test_pca)
y_pred8 = model8.predict(X_test_pca)
# Evaluating the models (pca)
print("Model 1 (Linear, OVR) Accuracy:", accuracy_score(y_test, y_pred1),"\nClassification Report:\n", classification_report(y_test, y_pred1), "\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred1))
print("Model 2 (Polynomial, OVR) Accuracy:", accuracy_score(y_test, y_pred2),"\nClassification Report:\n", classification_report(y_test, y_pred2), "\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred2))
print("Model 3 (Sigmoid, OVR) Accuracy:", accuracy_score(y_test, y_pred3),"\nClassification Report:\n", classification_report(y_test, y_pred3), "\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred3))
print("Model 4 (RBF, OVR) Accuracy:", accuracy_score(y_test, y_pred4),"\nClassification Report:\n", classification_report(y_test, y_pred4), "\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred4))
print("Model 5 (Linear, OVO) Accuracy:", accuracy_score(y_test, y_pred5),"\nClassification Report:\n", classification_report(y_test, y_pred5), "\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred5))
print("Model 6 (Polynomial, OVO) Accuracy:", accuracy_score(y_test, y_pred6),"\nClassification Report:\n", classification_report(y_test, y_pred6), "\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred6))
print("Model 7 (Sigmoid, OVO) Accuracy:", accuracy_score(y_test, y_pred7),"\nClassification Report:\n", classification_report(y_test, y_pred7), "\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred7))
print("Model 8 (RBF, OVO) Accuracy:", accuracy_score(y_test, y_pred8),"\nClassification Report:\n", classification_report(y_test, y_pred8), "\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred8))
