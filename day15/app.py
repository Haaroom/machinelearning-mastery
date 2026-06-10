from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report, confusion_matrix, precision_score
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd  

# Load data correctly
data = load_wine()
X = data.data
y = data.target
feature_names = data.feature_names  # Fixed: extracted valid feature names

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y, shuffle=True)
tree_depths = [10, 50, 100, 200]

def evaluate(y_pred, y_test):
    # Fixed: Removed non-existent average parameter from accuracy_score
    print("accuracy_score: ", accuracy_score(y_test, y_pred))
    print("precision_score: ", precision_score(y_test, y_pred, average="weighted"))
    print("recall_score: ", recall_score(y_test, y_pred, average="weighted"))
    print("confusion matrix: \n", confusion_matrix(y_test, y_pred))
    print("classification_report: \n", classification_report(y_test, y_pred))
    # Fixed: Added average='weighted' to support multiclass targets
    print("f1_score: ", f1_score(y_test, y_pred, average="weighted"))

for i in tree_depths:
    model1 = RandomForestClassifier(max_depth=i, n_estimators=100, oob_score=True, random_state=42)
    model2 = ExtraTreesClassifier(n_estimators=100, oob_score=True, random_state=42, bootstrap=True, max_depth=i)
    model3 = AdaBoostClassifier(n_estimators=100, random_state=42)
    model4 = XGBClassifier(max_depth=i)
    
    model1.fit(X_train, y_train)
    model2.fit(X_train, y_train)
    model3.fit(X_train, y_train)
    model4.fit(X_train, y_train)
    
    y_pred1 = model1.predict(X_test)
    y_pred2 = model2.predict(X_test)
    y_pred3 = model3.predict(X_test)
    y_pred4 = model4.predict(X_test)
    
    importance1 = model1.feature_importances_
    importance2 = model2.feature_importances_
    importance3 = model3.feature_importances_
    importance4 = model4.feature_importances_
    
    # Fixed: Using feature_names instead of X.columns
    df_importance1 = pd.DataFrame({'Feature': feature_names, 'Importance': importance1}).sort_values(by='Importance', ascending=False)
    df_importance2 = pd.DataFrame({'Feature': feature_names, 'Importance': importance2}).sort_values(by='Importance', ascending=False)
    df_importance3 = pd.DataFrame({'Feature': feature_names, 'Importance': importance3}).sort_values(by='Importance', ascending=False)
    df_importance4 = pd.DataFrame({'Feature': feature_names, 'Importance': importance4}).sort_values(by='Importance', ascending=False)
    
    print("\n" + "="*40)
    print(f" RESULTS FOR MAX DEPTH: {i} ")
    print("="*40)
    print("\n--- Random Forest ---")
    evaluate(y_pred1, y_test)
    print("\n--- Extra Trees ---")
    evaluate(y_pred2, y_test)
    print("\n--- AdaBoost ---")
    evaluate(y_pred3, y_test)
    print("\n--- XGBoost ---")
    evaluate(y_pred4, y_test)
    
    # Fixed: Adjusted subplot layout configuration
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Plot 1: Random Forest (Fixed axis mapping to axes[0])
    sns.barplot(x='Importance', y='Feature', data=df_importance1, palette='Blues_r', ax=axes[0], edgecolor='black')
    axes[0].set_title('Random Forest Feature Importance', fontsize=12, weight='bold', pad=10)
    axes[0].set_xlabel('Importance Score')  
    axes[0].set_ylabel('Features')     
    axes[0].grid(axis='x', linestyle='--', alpha=0.5)
    
    # Plot 2: Extra Trees (Fixed axis mapping to axes[1])
    sns.barplot(x='Importance', y='Feature', data=df_importance2, palette='Blues_r', ax=axes[1], edgecolor='black')
    axes[1].set_title('Extra Trees Feature Importance', fontsize=12, weight='bold', pad=10)
    axes[1].set_xlabel('Importance Score')  
    axes[1].set_ylabel('')     
    axes[1].grid(axis='x', linestyle='--', alpha=0.5)
    
    # Plot 3: AdaBoost (Fixed axis mapping to axes[2])
    sns.barplot(x='Importance', y='Feature', data=df_importance3, palette='Blues_r', ax=axes[2], edgecolor='black')
    axes[2].set_title('AdaBoost Feature Importance', fontsize=12, weight='bold', pad=10)
    axes[2].set_xlabel('Importance Score')  
    axes[2].set_ylabel('Features')     
    axes[2].grid(axis='x', linestyle='--', alpha=0.5)
    
    # Plot 4: XGBoost (Fixed axis mapping to axes[3])
    sns.barplot(x='Importance', y='Feature', data=df_importance4, palette='Blues_r', ax=axes[3], edgecolor='black')
    axes[3].set_title('XGBoost Feature Importance', fontsize=12, weight='bold', pad=10)
    axes[3].set_xlabel('Importance Score')  
    axes[3].set_ylabel('')     
    axes[3].grid(axis='x', linestyle='--', alpha=0.5)  
    plt.suptitle(f'Comparative Feature Importance Analysis Across 4 Models (Max Depth: {i})', fontsize=16, weight='bold', y=0.98)
    plt.tight_layout()
    plt.show()
