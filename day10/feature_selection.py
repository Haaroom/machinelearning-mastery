# feature_selection.py
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def remove_high_correlation(df, threshold=0.85):
    df_numeric = df.select_dtypes(include=[np.number])
    corr_matrix = df_numeric.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return df.drop(columns=to_drop)
def variance_threshold(df, threshold=0.0):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(df[numeric_cols])
    kept_numeric_cols = numeric_cols[selector.get_support()]
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
    return df[list(non_numeric_cols) + list(kept_numeric_cols)]
def select_k_best(X, y, k=5):
    selector = SelectKBest(score_func=f_classif, k=k)
    X_new = selector.fit_transform(X, y)
    kept_features = X.columns[selector.get_support()]
    return pd.DataFrame(X_new, columns=kept_features, index=X.index)
def rfe_selection(X, y, n_features=5):
    estimator = LogisticRegression(max_iter=1000, solver='liblinear')
    selector = RFE(estimator, n_features_to_select=n_features, step=1)
    X_new = selector.fit_transform(X, y)
    kept_features = X.columns[selector.get_support()]
    return pd.DataFrame(X_new, columns=kept_features, index=X.index)
def importance_selection(X, y, threshold="mean"):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    importances = model.feature_importances_
    feature_imp_df = pd.Series(importances, index=X.columns)
    if threshold == "mean":
        cutoff = feature_imp_df.mean()
    else:
        cutoff = threshold    
    kept_features = feature_imp_df[feature_imp_df >= cutoff].index
    return X[kept_features]
