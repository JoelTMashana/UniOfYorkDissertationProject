import pandas as pd
from collections import defaultdict
from kedro.pipeline import node
from sklearn.impute import SimpleImputer, KNNImputer


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

def filter_data_from_given_year(df, year):
    filtered_df = df[df['Start date'] >= f'{year}-01-01'] 
    return filtered_df

def filter_data_on_supplychain_finance(data, year):
    
    # Convert 'Start date' and 'End date' columns to datetime format by inferring the format and coercing errors
    data['Start date'] = pd.to_datetime(data['Start date'], infer_datetime_format=True, errors='coerce')
    data['End date'] = pd.to_datetime(data['End date'], infer_datetime_format=True, errors='coerce')

    # Filter out rows with NaT values in the "Start date" or "End date" columns
    data = data.dropna(subset=['Start date', 'End date'])

    data = filter_data_from_given_year(data, year)
    
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

    # Re-order columns to move 'Period' to the leftmost position
    columns = ['Period'] + [col for col in data if col != 'Period']
    data = data[columns]
    return data


def remove_redundant_columns(data):
    data = data.drop(['Start date', 'End date',  'Filing date', 'URL'], axis=1)
    return data

def anonymise_data(data):
    data = data.drop(['Company', 'Company number',  'Report Id'], axis=1)
    return data

def encode_column(data, columns_to_encode):
    for column_name in columns_to_encode:
        data[column_name] = data[column_name].apply(lambda x: 1 if x == True or x == 'TRUE' else (0 if x == False or x == 'FALSE' else x))

        # Handle circumstance where column is made up of TRUE or FALSE and empty cells
        # -- assumption
        if data[column_name].isnull().all() or (data[column_name] == 0).all():
            data[column_name].fillna(1, inplace=True)

        elif data[column_name].isnull().all() or (data[column_name] == 1).all():
            data[column_name].fillna(0, inplace=True)

        else:
            data[column_name].fillna(0, inplace=True)
    return data


def align_columns(data, column_one, column_two):
    mask = (data[column_one] == 0) & (data[column_two].isna())
    data.loc[mask, column_two] = 0
    return data




### code related to the inflation rates 
def convert_date_format(df, column_name, format='%d-%b-%y'):
    df[column_name] = pd.to_datetime(df[column_name], format=format)
    return df


#### Commented out in pipenline therefore not in use for now
# start_date='2017-01-01'
def prepare_inflation_data(data, start_date='2017-01-01'):
    data['Date Changed'] = pd.to_datetime(data['Date Changed'], format='%d-%b-%y')

    # data = data[data['Date Changed'] >= pd.to_datetime(start_date)]

    data.sort_values('Date Changed', inplace=True)

    data.set_index('Date Changed', inplace=True)

    monthly_data = data.resample('MS').ffill()

    monthly_data.reset_index(inplace=True)
    
    monthly_data['Date Changed'] = monthly_data['Date Changed'].dt.strftime('%Y-%b').str.upper()


    # print ('Monthly Data ')
    # print(monthly_data)
    return monthly_data


def flatten_periods(periods):
    flat_data = [period[1:-1].replace(", ", " - ") for period_list in periods.values() for period in period_list]
    df = pd.DataFrame(flat_data, columns=["Period"])
    return df

def get_average_inflation_for_periods(data, periods):
    """
    Calculate the average inflation rate for a list of periods.
    """
    periods_flattened = periods = flatten_periods(periods)
    print('Periods flattened')
    print(periods_flattened)
    mean_rates = {}
    print(data.columns)
    for period in periods_flattened['Period']:
        start_date, end_date = period.split(" - ")
        start_date = pd.Timestamp(start_date) # Do i need timestamp??
        end_date = pd.Timestamp(end_date)

        ## Must verify this logic and ensure that it is actually getting means for correct periods
        ## to me looks like its not actually comparing the period but instead date col only has one date
        ## possibly just the start date.
        ## Need to print or output the inflation df being used to verify.
        ## Print mean rates dict too to see what it looks like.
        mask = (data["Date Changed"] >= start_date) & (data["Date Changed"] <= end_date)
        filtered_data = data[mask]
        
        mean_rate = filtered_data["Rate"].mean()
        
        mean_rates[period] = mean_rate

    return mean_rates


## Related to GDP 
def gdp_remove_headers(data):

    # Rename 'Title' column to 'Date'
    data.rename(columns={'Title': 'Date'}, inplace=True)

    # Drop rows before '1997 JAN'
    index_to_drop = data[data['Date'] == '1997 JAN'].index[0]
    data_cleaned = data.loc[index_to_drop:].reset_index(drop=True)

    return data_cleaned



def calculate_gdp_averages_for_period(data, start_date, end_date):
    data['Date'] = pd.to_datetime(data['Date'], format='%Y %b')
    
    period_data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]
    
    # Calculate the average 
    averages = period_data.mean().drop('Date')  # Drop the 'Date' column as it's not needed in averages
    
    return averages

def process_gdp_averages(data, payment_periods):
    all_averages = []
    all_periods = []
    for year_periods in payment_periods.values():
        for period in year_periods:
            start, end = period[1:-1].split(', ')  # Split the period string into start and end dates
            all_periods.append(f"{start} - {end}")
            averages_for_period = calculate_gdp_averages_for_period(data, start, end)
            all_averages.append(averages_for_period)
    
    # Convert all averages to a DataFrame
    averages_df = pd.DataFrame(all_averages)
    averages_df.insert(0, 'Period', all_periods)
    print('averages_df')
    print(averages_df)
    return averages_df


## Concerning data combination

def combine_datasets(payment_practices, gdp_averages):
    combined_df = pd.merge(payment_practices, gdp_averages, on="Period", how="left")

    return combined_df


def convert_float_columns_to_int(data):
    """
    Converts columns with float values to integers if all values in the column are whole numbers.
    """
    for column in data.select_dtypes(include=['float']):
        if (data[column].dropna() % 1 == 0).all():
            data[column] = data[column].astype(pd.Int64Dtype())
    return data




### Handling missing values 

def mean_imputation(data, exclude_column='Period'):
    data_to_impute = data.drop(columns=[exclude_column])
    
    imputer = SimpleImputer(strategy='mean')
    imputed_data = pd.DataFrame(imputer.fit_transform(data_to_impute), columns=data_to_impute.columns)
    
    imputed_data[exclude_column] = data[exclude_column].values
    
    # Reorder the columns 
    imputed_data = imputed_data[data.columns]
    
    return imputed_data



