import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Optional, Union

# Read the chunk and get columns from the datafields dictionary
# if Oldest=True, take the oldest instance, otherwise newest
def get_columns_from_chunk(chunk: pd.DataFrame, datafields: List[str], oldest: bool = False) -> pd.DataFrame:
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
def convert_date_to_binary(df: pd.DataFrame, fields: List[str]) -> pd.DataFrame:
    start_date = pd.Timestamp("1950-01-01")
    end_date = pd.Timestamp("2030-01-01")
    
    for col in fields:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
            mask = (df[col] >= start_date) & (df[col] <= end_date)
    
            df[col] = np.where(mask, 1, 0)
    
    return df

# Prints the number of rows with NA values for each column.
def count_na_in_dataframe(df: pd.DataFrame) -> None:
    na_counts = df.isna().sum().sort_values(ascending=False)
    
    for column, na_count in na_counts.items():
        print(f"{column}: {na_count} missing values")

# Prints number of rows with NA or negative values.
def count_na_and_negative(df: pd.DataFrame) -> None:
    na_counts = df.isna().sum()
    negative_counts = (df < 0).sum(numeric_only=True)
    total_counts = (na_counts + negative_counts).sort_values(ascending=False)
    
    for column, count in total_counts.items():
        print(f"{column}: {count} missing or negative values")

# map education into 4 groups as specified in the map
def map_education_levels(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        1: 3,  # College or University degree -> Level 3
        2: 2,  # A levels/AS levels or equivalent -> Level 2
        3: 1,  # O levels/GCSEs or equivalent -> Level 1
        4: 1,  # CSEs or equivalent -> Level 1
        5: 2,  # NVQ or HND or HNC or equivalent -> Level 2
        6: 2,  # Other professional qualifications -> Level 2
        -7: 0, # No education
        -3: pd.NA  # Convert to NA
    }
    df["Education"] = df["Education"].map(mapping)
    return df

# returns the number of rows in the dataframe with more than `x` NA features
def count_rows_with_na_greater_than(df: pd.DataFrame, x: int) -> int:
    na_counts = df.isna().sum(axis=1)
    return (na_counts > x).sum()

# drops all rows with more than `x` NA features
def drop_rows_with_na_greater_than(df: pd.DataFrame, x: int) -> pd.DataFrame:
    na_counts = df.isna().sum(axis=1)
    return df[na_counts <= x]

# maps vascular problems by severity (unused)
def map_vascular_levels(df: pd.DataFrame) -> pd.DataFrame:
    col_name = "Report of vascular problems"
    # Replace -7 with 0 and -3 with NA
    df[col_name] = df[col_name].replace({-7: 0, -3: pd.NA})
    
    # Map severity levels
    severity_mapping = {
        1: 1,  # Heart attack
        2: 1,  # Angina
        3: 1,  # Stroke
        4: 1   # High blood pressure
    }
    
    df[col_name] = df[col_name].map(lambda x: severity_mapping.get(x, x))
    
    return df

# Instead of mapping vascular problems from 0 to 4, convert to categorical features
def one_hot_encode_vascular_problems(df: pd.DataFrame) -> pd.DataFrame:
    column_name = "Report of vascular problems"

    df[['Heart Attack', 'Angina', 'Stroke', 'High Blood Pressure']] = 0

    df.loc[df[column_name] == 1, 'Heart Attack'] = 1
    df.loc[df[column_name] == 2, 'Angina'] = 1
    df.loc[df[column_name] == 3, 'Stroke'] = 1
    df.loc[df[column_name] == 4, 'High Blood Pressure'] = 1

    df = df.drop(column_name, axis=1)

    return df


