import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os



def get_data(file_path):
    '''Loads data from a given location.'''
    print(f'Getting data from {file_path}')
    df_raw = pd.read_csv(file_path)
    return df_raw

def rename_columns(df, columnName):
    '''Rename columns in a given pandas dataframe'''
    return df.rename(columns = columnName)

def drop_duplicates(df):
    '''Drop duplicates in a given pandas dataframe'''
    return df.drop_duplicates()

def find_outliers_IQR(df,numeric_col):
   '''Find outliers for a given pandas dataset's numeric columns and return the outliers'''
   q1=df[numeric_col].quantile(0.25)
   q3=df[numeric_col].quantile(0.75)
   IQR=q3-q1
   outliers = df[((df[numeric_col]<(q1-1.5*IQR)) | (df[numeric_col]>(q3+1.5*IQR)))]
   return outliers

def export_csv(df, fileName):
    '''Export pandas data frame into csv'''
    return df.to_csv(fileName, index=False)