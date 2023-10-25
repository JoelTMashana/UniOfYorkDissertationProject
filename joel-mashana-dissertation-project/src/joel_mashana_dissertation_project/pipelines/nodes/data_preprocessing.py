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

    # Assert that all rows in the filtered_data have 'Supply-chain financing offered' set to True
    condition_met = all(filtered_data['Supply-chain financing offered']), "Not all rows have 'Supply-chain financing offered' set to True"
    # Sort the filtered data by 'Start date'
    
    if condition_met:
        sorted_data = filtered_data.sort_values(by='Start date')
        print("Data filtered such that 'Supply-chain financing offered' is True for each record")
        return sorted_data
    else:
        print("Not all rows have 'Supply-chain financing offered' set to True")
        return None 
    
    
    return sorted_data  
