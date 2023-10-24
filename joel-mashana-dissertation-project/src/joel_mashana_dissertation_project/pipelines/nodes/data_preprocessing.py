import pandas as pd

from kedro.pipeline import node


def filter_rows_based_on_conditions(df, conditions):
    """
    Filter rows in a dataframe based on the given conditions.

    Parameters:
    - df (DataFrame): The input dataframe.
    - conditions (str): A string that specifies the conditions for filtering.

    Returns:
    - DataFrame: A filtered dataframe.
    """
    return df.query(conditions)

def filter_data_on_supplychain_finance(data):
    
    # Convert 'Start date' and 'End date' columns to datetime format by inferring the format and coercing errors
    data['Start date'] = pd.to_datetime(data['Start date'], infer_datetime_format=True, errors='coerce')
    data['End date'] = pd.to_datetime(data['End date'], infer_datetime_format=True, errors='coerce')

    # Filter out rows with NaT values in the "Start date" or "End date" columns
    data = data.dropna(subset=['Start date', 'End date'])

    # Filter the dataset where 'Supply-chain financing offered' is True
    conditions = "`Supply-chain financing offered` == True"
    filtered_data = filter_rows_based_on_conditions(data, conditions)

    # Sort the filtered data by 'Start date'
    sorted_data = filtered_data.sort_values(by='Start date')
    
    return sorted_data  
