import pandas as pd 
import os 

def save_processed_data(filepath,df):
    if not os.path.dirname(filepath) and os.path.exists(path):
        os.makedirs(os.path.dirname(filepath))
    df_copy=df.copy()
    df1=pd.to_csv()
    print("success")
def load_processed_data(filename):
    """Loads a CSV file if it exists in the current working directory."""
    # Check if the file exists in the current working directory
    if not os.path.exists(filename):
        print(f"Error: The file '{filename}' does not exist in {os.getcwd()}.")
        return None   
    try:
        # Use pandas to read the file
        df = pd.read_csv(filename)
        return df
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def column_summary(df):
    if df is None:
        print("No DataFrame provided.")
        return
    print(f"=== DataFrame Summary (Shape: {df.shape}) ===")
    summary_df = pd.DataFrame({
        'Data Type': df.dtypes,
        'Non-Null Count': df.notnull().sum(),
        'Missing Values': df.isnull().sum(),
        'Missing %': (df.isnull().sum() / len(df) * 100).round(2),
        'Unique Values': df.nunique()
    })
    
    print(summary_df)
