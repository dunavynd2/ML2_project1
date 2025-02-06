import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

def clean_dataset(df, scale_type=None):
    df.fillna(df.median(numeric_only=True), inplace=True)  
    df.fillna("Unknown", inplace=True)  
    
    df.drop_duplicates(inplace=True)
    
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip().str.lower()
    
    for col in df.select_dtypes(include=['datetime64']).columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col])
            except ValueError:
                pass
    
    for col in df.select_dtypes(include=['number']).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = np.clip(df[col], lower_bound, upper_bound)
    
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    
    if scale_type == 'standard':
        scaler = StandardScaler()
        df[df.select_dtypes(include=['number']).columns] = scaler.fit_transform(df.select_dtypes(include=['number']))
    elif scale_type == 'minmax':
        scaler = MinMaxScaler()
        df[df.select_dtypes(include=['number']).columns] = scaler.fit_transform(df.select_dtypes(include=['number']))
    
    return df

df = pd.read_csv("AirbnbListings.csv") 
df_cleaned = clean_dataset(df, scale_type='standard') 
print(df_cleaned.head())

df_cleaned.to_csv("AirbnbListings_Cleaned.csv", index=False)