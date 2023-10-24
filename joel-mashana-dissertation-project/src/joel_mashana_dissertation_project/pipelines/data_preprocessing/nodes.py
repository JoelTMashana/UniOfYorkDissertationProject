import pandas as pd

def clean_payment_practise_data(data: pd.DataFrame) -> pd.DataFrame:
    # 
    cleaned_data = data
    return cleaned_data

def clean_monthly_gdp_data(data: pd.DataFrame) -> pd.DataFrame:
    # Remove the rows CDID, PreUnit, Unit, Release Date, Next Release, Important Notes
    # Changed column name title to date
    cleaned_data = data
    return cleaned_data

def clean_inflation_rates_data(data: pd.DataFrame) -> pd.DataFrame:
    # Format Date Changed column name to 'Date'
    # Format dates from 16-jun-21 to 2021 JUN - example
    # Where there are more than one record per month, use the average
    # -- Consider whats actually more valuable here. The rate or the rate change? 
    #    if rate change, this is a step for the feature engineering step.
    cleaned_data = data
    return cleaned_data

def combine_datasets(payment_data: pd.DataFrame, gdp_data: pd.DataFrame, inflation_data: pd.DataFrame) -> pd.DataFrame:
    # Your combining logic here
    combined_data = payment_data  # Just a placeholder
    return combined_data

def feature_engineering(data: pd.DataFrame) -> pd.DataFrame:
    # Your feature engineering logic here
    feature_engineering_data = data
    return feature_engineering_data
