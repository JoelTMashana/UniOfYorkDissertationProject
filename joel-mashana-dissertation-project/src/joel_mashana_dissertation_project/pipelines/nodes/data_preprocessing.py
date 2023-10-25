import pandas as pd
from collections import defaultdict
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



def extract_payment_periods(data):
    # Need to consider flattening and asserting that all periods are unique
    # Filter data for records starting from 2017 onwards
  
    data['Start date'] = pd.to_datetime(data['Start date'], errors='coerce')
    data['End date'] = pd.to_datetime(data['End date'], errors='coerce')
 
    periods_from_2017 = data[data['Start date'].dt.year >= 2017]

    # Initialise a dictionary to store periods by year
    payment_periods = defaultdict(list)

    for _, row in periods_from_2017.iterrows():
        # Get the years for start and end dates
        start_year = row['Start date'].year
        end_year = row['End date'].year

        # Convert the start and end dates to "Year Month" format
        start_date_str = row['Start date'].strftime('%Y %b').upper()
        end_date_str = row['End date'].strftime('%Y %b').upper()
        
        # Create the period tuple
        period = f"({start_date_str}, {end_date_str})"
        
        # Add the period to the start year's list
        if period not in payment_periods[start_year]:
            payment_periods[start_year].append(period)
        
        # If the period spans multiple years, add to the end year's list as well
        if start_year != end_year:
            if period not in payment_periods[end_year]:
                payment_periods[end_year].append(period)
    
    return dict(payment_periods)


def create_period_column(data):
    """Add 'Period' column to the data."""
    data['Start date'] = pd.to_datetime(data['Start date'], errors='coerce')
    data['End date'] = pd.to_datetime(data['End date'], errors='coerce')
    data['Period'] = data['Start date'].dt.strftime('%Y %b').str.upper() + " - " + data['End date'].dt.strftime('%Y %b').str.upper()
    return data


