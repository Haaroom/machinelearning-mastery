import pandas as pd 
def missing_report(df):
    return df.isna().sum()

def fill_mean(df, columns):
    return df[columns].fillna(df[columns].mean())
def fill_median(df, columns):
    return df[columns].fillna(df[columns].median())
def fill_mode(df, columns):
    return df[columns].fillna(df[columns].mode())
def fill_constant(df, columns, value):
    return df[columns].fillna(value)
def drop_missing_rows(df):
    return df[columns].dropna()
def drop_missing_columns(df, threshold=0.5):
    return df.drop(axis=1,threshold=int(len(df)(1-threshold)))
