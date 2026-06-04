import numpy as np
import pandas as pd 
from sklearn.preprocessing import PowerTransformer, QuantileTransformer

class Transformation:
    def __init__(self, df, columns):
        self.df = df
        self.columns = columns
    def log_transform(self):
        df1 = self.df.copy()
        df1[self.columns] = np.log1p(df1[self.columns])
        return df1
    def sqrt_transformation(self): 
        df1 = self.df.copy()
        df1[self.columns] = np.sqrt(df1[self.columns])
        return df1 
    def power_transformation(self):
        df1 = self.df.copy()
        pt = PowerTransformer()
        df1[self.columns] = pt.fit_transform(df1[self.columns])
        return df1
    def quantile_transformer(self):
        df1 = self.df.copy()
        qt = QuantileTransformer()
        df1[self.columns] = qt.fit_transform(df1[self.columns])
        return df1
