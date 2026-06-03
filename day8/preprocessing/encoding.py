import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder

def label_encode(df, columns):
    df_copy = df.copy()
    for col in columns:
        encoder = LabelEncoder()
        df_copy[col] = encoder.fit_transform(df_copy[col].astype(str))
    return df_copy
def one_hot_encode(df, columns):
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_array = encoder.fit_transform(df[columns])
    encoded_cols = encoder.get_feature_names_out(columns)
    encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols, index=df.index)
    return pd.concat([df.drop(columns=columns), encoded_df], axis=1)
def ordinal_encode(df, columns, categories):
    encoder = OrdinalEncoder(categories=categories)
    df_copy = df.copy()
    df_copy[columns] = encoder.fit_transform(df_copy[columns])
    return df_copy
def frequency_encode(df, column):
    freq = df[column].value_counts(normalize=True)
    df_copy = df.copy()
    df_copy[f'{column}_freq'] = df_copy[column].map(freq)
    return df_copy
