import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read the chunk and get columns from the datafields dictionary
# if Oldest=True, take the oldest instance, otherwise newest
def get_columns_from_chunk(chunk, datafields, oldest=False):
    selected_columns = {}
    for field_name, instances in datafields.items():
        instance_key = min(instances) if oldest else max(instances)
        selected_columns[field_name] = instances[instance_key]
    
    # Select only the necessary columns from the chunk
    filtered_chunk = chunk[list(selected_columns.values())].rename(columns={
        v: k for k, v in selected_columns.items()
    })
    
    return filtered_chunk

# from the given "fields" list, convert all columns where date is in range, to 0 or 1 instead of a date.
# Having date as not NA implies a person was diagnosed with said condition
def convert_date_to_binary(df, fields):
    start_date = pd.Timestamp("1950-01-01")
    end_date = pd.Timestamp("2030-01-01")
    
    for col in fields:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
            mask = (df[col] >= start_date) & (df[col] <= end_date)
    
            df[col] = np.where(mask, 1, 0)
    
    return df

# Prints the number of rows with NA values for each column.
def count_na_in_dataframe(df):
    na_counts = df.isna().sum().sort_values(ascending=False)
    
    for column, na_count in na_counts.items():
        print(f"{column}: {na_count} missing values")

