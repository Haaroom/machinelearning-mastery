from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,recall_score,classification_report,confusion_matrix,precision_score
import matplotlib.pyplot as plt 
from sklearn.datasets import load_wine
data = load_wine()
X= data.data
y=data.target
X_train,X_test,y_train,y_test = train_test_split(X,y,stratify=y,random_state=42,shuffle=True)
def evaluate(y_pred,y_test):
    print("accuracy_score: ",accuracy_score(y_test,y_pred,avetage="weighted"))
    print("precision_score: ",precision_score(y_test,y_pred,"weighted"))
    print("recall_score: ",recall_score(y_test,y_pred,"weighted"))
    print("confusion matrix: ",confusion_matrix(y_test,y_pred))
    print("classification_report: ",classification_report(y_test,y_pred))
    print("f1_score: ",f1_score(y_test,y_pred))
def train_DecisionTree(X_train,X_test,y_train,y_test):
    for depth in range(1,11):
        model = DecisionTreeClassifier(max_depth=depth)
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test,y_test)
        print(f"model with depth {depth}'s evalaution")
        evaluate(y_pred)
        plt.bar(model.feature_names,model.feature_importances_)
        plt.title("Feature Importance")
        plt.xlabel("features")
        plt.ylabel("importance")
        plt.show()
        print("plotting the tree")
        figure=plot_tree(model,max_depth=depth,feature_names=model.feature_names)
        figure.show()
        

    


