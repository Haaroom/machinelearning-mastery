import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer

def standard_scale(df, columns):
    df_copy = df.copy()
    sc = StandardScaler()
    df_copy[columns] = sc.fit_transform(df_copy[columns])
    return df_copy 
def minmax_scale(df, columns):
    df_copy = df.copy()
    mc = MinMaxScaler()
    df_copy[columns] = mc.fit_transform(df_copy[columns])
    return df_copy 
def robust_scale(df, columns):
    df_copy = df.copy()
    sc = RobustScaler()
    df_copy[columns] = sc.fit_transform(df_copy[columns])
    return df_copy 
def normalize_data(df, columns):
    df_copy = df.copy()
    sc = Normalizer()
    df_copy[columns] = sc.fit_transform(df_copy[columns])
    return df_copy
