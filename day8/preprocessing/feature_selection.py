import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif

def remove_high_correlation(df, threshold=0.85):
    df_copy = df.copy()
    corr_matrix = df_copy.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
    return df_copy.drop(columns=to_drop)

def variance_filter(df, threshold=0.01):
    df_copy = df.copy()
    variances = df_copy.var()
    low_variance_cols = variances[variances <= threshold].index
    return df_copy.drop(columns=low_variance_cols)

def select_k_best_features(X, y, k=10):
    selector = SelectKBest(score_func=f_classif, k=k)
    X_new = selector.fit_transform(X, y)
    selected_features = selector.get_feature_names_out(input_features=X.columns)
    return pd.DataFrame(X_new, columns=selected_features, index=X.index)
