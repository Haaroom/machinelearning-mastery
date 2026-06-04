import pandas as pd 
from itertools import combinations
from sklearn.preprocessing import PolynomialFeatures
class FeatureCreation:
    def __init__(self,df,columns,bins):
        self.df=df
        self.columns=columns
        self.bins=bins
    def equal_width_binning(self,n_bins,labels=None):
        df1=self.df.copy()
        for col in self.columns:
            df1[f'binned_{col}']=pd.cut(df1[col], bins=n_bins, labels=labels)
        return df1
    def equal_frequncy_binning(self,n_bins,labels=None):
        df1=self.df.copy()
        for col in self.columns:
            df1[f'binned_{col}']=pd.qcuts(df1[col],bins=n_bins,labels=labels,duplicates="drop")
        return df1
    def create_interaction(self,operation="mulitply"):
        df1=self.df.copy()
        for col1,col2 in combinations(self.columns,2):
            new_col = f'{col1}_{operation}_{col2}'
            df1[new_col]=df1[col1]+df1[col2] if operation.strip().lower()=="add" else  df1[col1]-df1[col2] if operation.strip().lower()=="subtract" else df1[col1]*df1[col2] if operation.strip().lower()=="multiply" else df1[col1]/df1[col2] if operation.strip().lower() == "divide" else "operation now defined"
            return df1
    def polynomial_features(self,degree):
        df1=self.df.copy()
        pf=PolynomialFeatures(degree=degree)
        polyarray=pf.fit_transform(df1[self.columns])
        feature_names = pf.get_feature_names_out(self.columns)
        df_poly = pd.DataFrame(polyarray, columns=feature_names, index=df1.index)
        df_rest = df1.drop(columns=self.columns)
        return pd.concat([df_rest, df_poly], axis=1)

    def extract_datetime_features(df, column):
        df1 = df.copy()
        df1[column] = pd.to_datetime(df1[column], errors='coerce')
        df1[f'{column}_year'] = df1[column].dt.year
        df1[f'{column}_month'] = df1[column].dt.month
        df1[f'{column}_day'] = df1[column].dt.day
        df1[f'{column}_weekday'] = df1[column].dt.weekday  
        df1[f'{column}_quarter'] = df1[column].dt.quarter
        return df

