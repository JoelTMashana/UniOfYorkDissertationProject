# Standard libraries
import os
import random

# Third-party libraries
# Data Manipulation
import numpy as np
import pandas as pd

# Data Visualisation
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning - General
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as make_pipeline_imb

# Statistics and Model Evaluation
import statsmodels.api as sm
from scipy.stats import expon, reciprocal

# Visualisation for Machine Learning
from yellowbrick.cluster import KElbowVisualizer

# Neural Networks and Deep Learning
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from scikeras.wrappers import KerasClassifier

# Hyperparameter Tuning
import kerastuner as kt

# Model Interpretability
import shap

# Local application/library specific imports
from collections import defaultdict
from kedro.pipeline import node


from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate



def filter_rows_based_on_conditions(df, conditions):
    """
    Filter rows in a dataframe based on the given conditions.
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
    averages = period_data.mean().drop('Date') 
    
    return averages

def process_gdp_averages(data, payment_periods):
    all_averages = []
    all_periods = []
    for year_periods in payment_periods.values():
        for period in year_periods:
            start, end = period[1:-1].split(', ')  
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

def mean_imputation(data, exclude_column):
    data_to_impute = data.drop(columns=[exclude_column])
    
    imputer = SimpleImputer(strategy='mean')
    imputed_data = pd.DataFrame(imputer.fit_transform(data_to_impute), columns=data_to_impute.columns)
    
    imputed_data[exclude_column] = data[exclude_column].values
    
    # Reorder the columns 
    imputed_data = imputed_data[data.columns]
    
    assert not imputed_data.isnull().any().any(), "There should be no null values after imputation."
    assert set(imputed_data.columns) == set(data.columns), "The columns should match the original data."

    return imputed_data


# Consider using K neighbours for 



## Handling Outliers 

def robust_scale_column(data, column_name):
    """
    Apply RobustScaler to a specified column in a DataFrame.
    """
    
    if column_name not in data.columns:
        raise ValueError(f"Column {column_name} not found in the DataFrame.")

    data_to_scale = data[column_name].values.reshape(-1, 1)

    scaler = RobustScaler()

    scaled_data = scaler.fit_transform(data_to_scale)

    # Assign the scaled data back to df
    scaled_column_name = f"{column_name}_scaled"
    data[scaled_column_name] = scaled_data.flatten()

    return data


# Related to creating the target variable

def find_optimal_clusters(data, column_to_cluster):
    feature = data[[column_to_cluster]]
    model = KMeans(random_state=0)
    visualiser = KElbowVisualizer(model, k=(2,10), metric='silhouette', timings=False) # May need to use metric for the actual algo too
    visualiser.fit(feature)
    # visualiser.show()

    return visualiser.elbow_value_


def perform_kmeans_clustering(data, column_to_cluster):
    optimal_number_of_clusters = find_optimal_clusters(data,  column_to_cluster)
    kmeans = KMeans(n_clusters=optimal_number_of_clusters, random_state=0)
    data['Clusters'] = kmeans.fit_predict(data[[column_to_cluster]])

    data['Risk Level'] = kmeans.labels_ + 1 # assign risk levels, account for 0 index
    data = data.drop(['Clusters', '% Invoices not paid within agreed terms'], axis=1)

    
    assert 'Risk Level' in data.columns, "Risk Level column does not exist."
    assert 'Clusters' not in data.columns, "Clusters column should not exist after dropping it."
    assert '% Invoices not paid within agreed terms' not in data.columns, "Error: '% Invoices not paid within agreed terms' column still exists in the dataset."

    assert data['Risk Level'].nunique() == optimal_number_of_clusters, (
        "The number of unique values in the Risk Level column is not equal to the optimal number of clusters."
    )
    return data


### Dimensionality reduction 

def scale_and_apply_pca(data, n_components, columns_to_exclude, target_column):
    X = data.drop(columns=[target_column])
    X=  data.drop(columns=[columns_to_exclude])
    y = data[target_column]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

     # Combine the principal components with the target variable
    principal_components = pd.DataFrame(X_pca)
    principal_components[target_column] = y
    
    return principal_components


#### Scale 

def standard_scale_data(data, columns_to_exclude, target_column):
    # X = data.drop(columns=[target_column] + columns_to_exclude)
    X = data.drop(columns=[target_column])
    X=  data.drop(columns=[columns_to_exclude])
    y = data[target_column]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=data.index)

    X_scaled_df[target_column] = y

    return X_scaled_df

# ###### ML Algorithms


def main_split_train_test_validate(data, target_column, columns_to_exclude):
    # Initial number of columns
    initial_number_of_columns = data.shape[1]

    X = data.drop(columns=[columns_to_exclude, target_column]) 
    y = data[target_column]

    X_train_temp, X_test, y_train_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_validate, y_train, y_validate = train_test_split(X_train_temp, y_train_temp, test_size=0.25, random_state=42)  

    # Assertions
    assert X_train.shape[1] ==  initial_number_of_columns - 2, "X_train should have one less column than the initial dataset"
    assert X_validate.shape[1] ==  initial_number_of_columns - 2, "X_validate should have one less column than the initial dataset"
    assert X_test.shape[1] ==  initial_number_of_columns - 2, "X_test should have one less column than the initial dataset"
    
    assert 'Risk Level' not in X_train.columns, "Risk Level should not be in X_train"
    assert 'Risk Level' not in X_validate.columns, "Risk Level should not be in X_validate"
    assert 'Risk Level' not in X_test.columns, "Risk Level should not be in X_test"

    assert y_train.ndim == 1, "y_train should only have one column"
    assert y_validate.ndim == 1, "y_validate should only have one column"
    assert y_test.ndim == 1, "y_test should only have one column"

    assert y_train.name == 'Risk Level', "y_train should have the column name 'Risky Level'"
    assert y_validate.name == 'Risk Level', "y_validate should have the column name 'Risky Level'"
    assert y_test.name == 'Risk Level', "y_test should have the column name 'Risky Level'"

    return X_train, X_validate, X_test, y_train, y_validate, y_test




def split_train_test_validate(data, target_column):
    X = data
    y = data[target_column]
    X = data.drop(columns=[target_column])

    X_train_temp, X_test, y_train_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train, X_validate, y_train, y_validate = train_test_split(X_train_temp, y_train_temp, test_size=0.25, random_state=42)  

    return X_train, X_validate, X_test, y_train, y_validate, y_test





def split_train_test_validate_smote_applied(data, target_column, columns_to_exclude):
    X = data.drop(columns=[columns_to_exclude, target_column]) 
    y = data[target_column]

    X_train_temp, X_test, y_train_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train, X_validate, y_train, y_validate = train_test_split(X_train_temp, y_train_temp, test_size=0.25, random_state=42)  

    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    return X_train_smote, X_validate, X_test, y_train_smote, y_validate, y_test


def split_train_test_validate_smote_applied_varied_splits(data, target_column, columns_to_exclude, split_num):
    X = data.drop(columns=[columns_to_exclude, target_column]) 
    y = data[target_column]

    X_train_temp, X_test, y_train_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=split_num)
    X_train, X_validate, y_train, y_validate = train_test_split(X_train_temp, y_train_temp, test_size=0.25, random_state=split_num)  

    smote = SMOTE(random_state=split_num)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    return X_train_smote, X_validate, X_test, y_train_smote, y_validate, y_test


def split_train_test_validate_rfe(data, target_column, columns_to_exclude):
    X = data
    # X = data.drop(columns=[columns_to_exclude])
    # X = data.drop(columns=[target_column])
    y = data[target_column]
    X = data.drop(columns=[columns_to_exclude, target_column]) # Check if made same mistake else where
    assert 'Period' not in X.columns, "Error: 'Period' column still exists in the dataset."
    assert '% Invoices not paid within agreed terms' not in X.columns, "Error: '% Invoices not paid within agreed terms' column still exists in the dataset."

    X_train_temp, X_test, y_train_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train, X_validate, y_train, y_validate = train_test_split(X_train_temp, y_train_temp, test_size=0.25, random_state=42)  

    return X_train, X_validate, X_test, y_train, y_validate, y_test

def train_decision_tree(X_train, y_train, X_validate, y_validate, model_name, number_of_iterations):
    
    param_dist = {
        "max_depth": [3, 5, 10, 15, 20, None],
        "min_samples_split": range(2, 50),
        "min_samples_leaf": range(1, 50),
        "criterion": ["gini", "entropy"]
    }

    decision_tree = DecisionTreeClassifier(random_state=42)
    random_search = RandomizedSearchCV(decision_tree, param_distributions=param_dist, 
                                       n_iter=number_of_iterations, cv=10, random_state=42)
    random_search.fit(X_train, y_train)

    best_params = random_search.best_params_
    best_model = random_search.best_estimator_

    predictions = best_model.predict(X_validate)

    print_model_name(model_name)
    accuracy = calculate_accuracy(y_validate, predictions)
    report = store_and_print_classification_report(y_validate, predictions)
    auc = print_auc(best_model, X_validate, y_validate)

    return best_model, pd.DataFrame({'accuracy': [accuracy]}), pd.DataFrame({'auc': [auc]}), pd.DataFrame({'report': [report]}), pd.DataFrame({'best params': [best_params]}) 


def get_hyperparameter_ranges(random_search, top_percentage=0.2):
    
    results_df = pd.DataFrame(random_search.cv_results_)
    top_results = results_df.sort_values('rank_test_score').head(int(len(results_df) * top_percentage))

    continuous_params_df = pd.DataFrame({
        'min_samples_split': [
            round(top_results['param_min_samples_split'].quantile(0.25)), 
            round(top_results['param_min_samples_split'].quantile(0.75))
        ],
        'min_samples_leaf': [
            round(top_results['param_min_samples_leaf'].quantile(0.25)), 
            round(top_results['param_min_samples_leaf'].quantile(0.75))
        ]
    })

   # Discrete parameters - selecting the most frequent value
    max_depth_value = top_results['param_max_depth'].value_counts().idxmax()
    criterion_value = top_results['param_criterion'].value_counts().idxmax()

    discrete_params_df = pd.DataFrame({
        'max_depth': [max_depth_value],
        'criterion': [criterion_value]
    })


    return continuous_params_df, discrete_params_df

def get_hyperparameter_ranges(random_search, top_percentage=0.2):
    
    results_df = pd.DataFrame(random_search.cv_results_)
    top_results = results_df.sort_values('rank_test_score').head(int(len(results_df) * top_percentage))

    continuous_params_df = pd.DataFrame({
        'min_samples_split': [
            round(top_results['param_min_samples_split'].quantile(0.25)), 
            round(top_results['param_min_samples_split'].quantile(0.75))
        ],
        'min_samples_leaf': [
            round(top_results['param_min_samples_leaf'].quantile(0.25)), 
            round(top_results['param_min_samples_leaf'].quantile(0.75))
        ]
    })

   # Discrete parameters - selecting the most frequent value
    max_depth_value = top_results['param_max_depth'].value_counts().idxmax()
    criterion_value = top_results['param_criterion'].value_counts().idxmax()

    discrete_params_df = pd.DataFrame({
        'max_depth': [max_depth_value],
        'criterion': [criterion_value]
    })


    return continuous_params_df, discrete_params_df



def train_decision_tree_with_random_search(X_train, y_train, X_validate, y_validate, model_name, number_of_iterations):
    
    ## Still need to apply smote and standard scaling
    param_dist = {
        "max_depth": [3, 5, 10, 15, 20, None],
        "min_samples_split": range(2, 50),
        "min_samples_leaf": range(1, 50),
        "criterion": ["gini", "entropy"]
    }

    decision_tree = DecisionTreeClassifier(random_state=42)
    random_search = RandomizedSearchCV(decision_tree, param_distributions=param_dist, 
                                       n_iter=number_of_iterations, cv=2, random_state=42)
    random_search.fit(X_train, y_train)

    best_model = random_search.best_estimator_
    predictions = best_model.predict(X_validate)

    print_model_name(model_name)
    accuracy = calculate_accuracy(y_validate, predictions)
    auc = print_auc(best_model, X_validate, y_validate)
    f1 = print_and_return_f1_score(y_validate, predictions)
    precision = print_and_return_precision(y_validate, predictions)
    recall = print_and_return_recall(y_validate, predictions)

    

    # Get hyperparameter ranges
    continuous_params_df, discrete_params_df = get_hyperparameter_ranges(random_search)

    

    return {
        'metrics': {
            'accuracy': accuracy,
            'auc': auc,
            'f1_score': f1,
            'precision': precision,
            'recall': recall
        },
        'continuous_params': continuous_params_df,
        'discrete_params': discrete_params_df
    }


def train_logistic_regression(X_train, y_train, X_validate, y_validate, model_name, number_of_iterations):

    param_dist = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2', 'elasticnet', 'none'],
        'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
        'class_weight': [None, 'balanced'],
        'l1_ratio': [None, 0.5, 0.75, 1]
    }

    logistic_regression_model = LogisticRegression(random_state=42)

    random_search = RandomizedSearchCV(logistic_regression_model, param_distributions=param_dist, 
                                   n_iter=number_of_iterations, cv=10, random_state=42)
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    random_search.fit(X_train_smote, y_train_smote)

    best_params = random_search.best_params_
    best_model = random_search.best_estimator_

    y_pred = best_model.predict(X_validate)

    print_model_name(model_name)
    accuracy = calculate_accuracy(y_validate, y_pred)
    report =  store_and_print_classification_report(y_validate, y_pred)
    auc = print_auc(best_model, X_validate, y_validate)

    return best_model, pd.DataFrame({'accuracy': [accuracy]}), pd.DataFrame({'auc': [auc]}), pd.DataFrame({'report': [report]}),  pd.DataFrame({'best params': [best_params]})


def train_svm(X_train, y_train, X_validate, y_validate, model_name, number_of_iterations):
    #not working
    param_dist = {
        'C': reciprocal(0.1, 10),
        'kernel': ['linear', 'sigmoid'],
        'degree': [2, 3, 4, 5]
    }
    svm_model = SVC()
    random_search = RandomizedSearchCV(svm_model, param_distributions=param_dist, 
                                    n_iter=number_of_iterations, cv=5, verbose=2, random_state=42, n_jobs=-1)
    
    random_search.fit(X_train, y_train)

    best_params = random_search.best_params_
    best_model = random_search.best_estimator_

    y_pred = best_model.predict(X_validate)

    print_model_name(model_name)

    accuracy = calculate_accuracy(y_validate, y_pred)
    report =  store_and_print_classification_report(y_validate, y_pred)
    auc = print_auc(best_model, X_validate, y_validate)

    return best_model, pd.DataFrame({'accuracy': [accuracy]}), pd.DataFrame({'auc': [auc]}), pd.DataFrame({'report': [report]}),  pd.DataFrame({'best params': [best_params]})

def get_hyperparameter_ranges_svm(random_search, top_percentage=0.2):
    results_df = pd.DataFrame(random_search.cv_results_)
    top_results = results_df.sort_values('rank_test_score').head(int(len(results_df) * top_percentage))

    # Continuous parameters - C value
    continuous_params_df = pd.DataFrame({
        'C': [
            round(top_results['param_C'].quantile(0.25), 2), 
            round(top_results['param_C'].quantile(0.75), 2)
        ]
    })

    # Discrete parameters - Kernel type
    kernel_value = top_results['param_kernel'].value_counts().idxmax()

    discrete_params_df = pd.DataFrame({
        'kernel': [kernel_value]
    })

    return continuous_params_df, discrete_params_df



def train_svm_with_random_search(X_train, y_train, X_validate, y_validate, model_name, number_of_iterations):
   
    param_dist = {
        'kernel': ['linear','sigmoid'],
        'C': [0.1, 1, 10, 100, 1000],
    }
    svm_model = SVC(probability=True)
    random_search = RandomizedSearchCV(svm_model, param_distributions=param_dist, 
                                    n_iter=number_of_iterations, cv=2, verbose=2, random_state=42, n_jobs=-1)
    
    random_search.fit(X_train, y_train)

    best_params = random_search.best_params_
    best_model = random_search.best_estimator_

    predictions = best_model.predict(X_validate)

    print_model_name(model_name)
    accuracy = calculate_accuracy(y_validate, predictions)
    auc = print_auc(best_model, X_validate, y_validate)
    f1 = print_and_return_f1_score(y_validate, predictions)
    precision = print_and_return_precision(y_validate, predictions)
    recall = print_and_return_recall(y_validate, predictions)

      # Get hyperparameter ranges
    continuous_params_df, discrete_params_df = get_hyperparameter_ranges_svm(random_search)
    print("Continuous Hyperparameters:", continuous_params_df)
    print("Discrete Hyperparameters:", discrete_params_df)

    return {
        'metrics': {
            'accuracy': accuracy,
            'auc': auc,
            'f1_score': f1,
            'precision': precision,
            'recall': recall
        },
        'continuous_params': continuous_params_df,
        'discrete_params': discrete_params_df
    }


def train_logistic_regression_for_rfe(X_train, y_train, X_validate, y_validate, model_name, number_of_features_to_select):
    logistic_regression_model = LogisticRegression(random_state=42)

    smote = SMOTE(random_state=42) # For now use this but eventually use the node
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    # RFE
    rfe = RFE(estimator=logistic_regression_model, n_features_to_select=number_of_features_to_select)
    rfe.fit(X_train_smote, y_train_smote)

    # Selecting features based on RFE
    # X_train_rfe = X_train_smote[:, rfe.support_]
    # X_validate_rfe = X_validate[:, rfe.support_]

    X_train_rfe = X_train_smote.iloc[:, rfe.support_]
    X_validate_rfe = X_validate.iloc[:, rfe.support_]


    logistic_regression_model.fit(X_train_rfe, y_train_smote)

    y_pred = logistic_regression_model.predict(X_validate_rfe)

    print_model_name(model_name)
    accuracy = calculate_accuracy(y_validate, y_pred)
    report = store_and_print_classification_report(y_validate, y_pred)
    auc = print_auc(logistic_regression_model, X_validate_rfe, y_validate)

    # Identifying selected features
    selected_features = pd.DataFrame({
        'Feature': X_train.columns[rfe.support_],
        'Ranking': rfe.ranking_[rfe.support_]
    })
   
    return logistic_regression_model, pd.DataFrame({'accuracy': [accuracy]}), pd.DataFrame({'auc': [auc]}), pd.DataFrame({'report': [report]}), selected_features

def train_ann(X_train, y_train, X_validate, y_validate, model_name, important_features_df):

    important_features = important_features_df[important_features_df['Ranking'] == 1]['Feature'].tolist()

    X_train = X_train[important_features]
    X_validate = X_validate[important_features]


    model = Sequential([
        Dense(30, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')  # Binary
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    # For binary cross entropy, may need to change labels in pre processing
    y_train_smote = y_train_smote.replace({1: 0, 2: 1})
    y_validate = y_validate.replace({1: 0, 2: 1}) 
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train_smote)
    X_validate_scaled = scaler.transform(X_validate)
   
    model.fit( X_train_scaled, y_train_smote, epochs=10, batch_size=64)

    y_pred_probs = model.predict(X_validate_scaled).ravel()
    y_pred = np.round(y_pred_probs)

    print_model_name(model_name)
    accuracy = calculate_accuracy(y_validate, y_pred)
    report =  store_and_print_classification_report(y_validate, y_pred)
    auc = print_auc_tf(model, X_validate_scaled, y_validate)

    loss, accuracy = model.evaluate( X_validate_scaled , y_validate)
    print(f"Loss: {loss}, Accuracy: {accuracy}")


    return model#, pd.DataFrame({'accuracy': [accuracy]}), pd.DataFrame({'auc': [auc]})



def train_decision_tree_experimental(X_train, y_train, X_validate, y_validate, model_name, exclude_column, important_features):
    
    ## Important features from Logistic regression rfe
    important_features = important_features[important_features['Ranking'] == 1]['Feature'].tolist()

    X_train = X_train[important_features]
    X_validate = X_validate[important_features]
    
    # X_train = X_train.drop(columns=exclude_column)
    # X_validate = X_validate.drop(columns=exclude_column)

    assert 'Period' not in X_train.columns, "Period Column should not be in the training set"
    assert 'Period' not in X_validate.columns, "Period Column should not be in the validation set"

    decision_tree = DecisionTreeClassifier(random_state=42)

    decision_tree.fit(X_train, y_train)


    predictions = decision_tree.predict(X_validate)

    print_model_name(model_name)
    accuracy = calculate_accuracy(y_validate, predictions)
    auc = print_auc(decision_tree, X_validate, y_validate)

    confusion_matrix_values = print_and_return_confusion_matrix(y_validate, predictions)
    f1 = print_and_return_f1_score(y_validate, predictions)
    precision = print_and_return_precision(y_validate, predictions)
    recall = print_and_return_recall(y_validate, predictions)

    tn, fp, fn, tp = confusion_matrix_values.ravel()

    report = store_and_print_classification_report(y_validate, predictions)
    return {
        'accuracy': accuracy,
        'auc': auc,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'confusion_matrix_tp': tp,
        'confusion_matrix_tn': tn,
        'confusion_matrix_fp': fp,
        'confusion_matrix_fn': fn,
    }

def train_logistic_regression_experimental(X_train, y_train, X_validate, y_validate, model_name, exclude_column):
  
    X_train = X_train.drop(columns=exclude_column)
    X_validate = X_validate.drop(columns=exclude_column)

    logistic_regression_model = LogisticRegression(random_state=42)

    logistic_regression_model.fit(X_train, y_train)

    predictions = logistic_regression_model.predict(X_validate)

    print_model_name(model_name)
    accuracy = calculate_accuracy(y_validate, predictions)
    auc = print_auc(logistic_regression_model, X_validate, y_validate)

    confusion_matrix_values = print_and_return_confusion_matrix(y_validate, predictions)
    f1 = print_and_return_f1_score(y_validate, predictions)
    precision = print_and_return_precision(y_validate, predictions)
    recall = print_and_return_recall(y_validate, predictions)

    tn, fp, fn, tp = confusion_matrix_values.ravel()

    return {
        'accuracy': accuracy,
        'auc': auc,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'confusion_matrix_tp': tp,
        'confusion_matrix_tn': tn,
        'confusion_matrix_fp': fp,
        'confusion_matrix_fn': fn,
    }

def train_svm_experimental(X_train, y_train, X_validate, y_validate, model_name, exclude_column):

    X_train = X_train.drop(columns=exclude_column)
    X_validate = X_validate.drop(columns=exclude_column)

    svm_model = SVC(kernel='linear', probability=True)    
    svm_model.fit(X_train, y_train)
    predictions = svm_model.predict(X_validate)

    print_model_name(model_name)
    accuracy = calculate_accuracy(y_validate, predictions)
    auc = print_auc( svm_model, X_validate, y_validate)

    confusion_matrix_values = print_and_return_confusion_matrix(y_validate, predictions)
    f1 = print_and_return_f1_score(y_validate, predictions)
    precision = print_and_return_precision(y_validate, predictions)
    recall = print_and_return_recall(y_validate, predictions)

    print("Confusion Matrix:\n", confusion_matrix_values)
    tn, fp, fn, tp = confusion_matrix_values.ravel()

    return {
        'accuracy': accuracy,
        'auc': auc,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'confusion_matrix_tp': tp,
        'confusion_matrix_tn': tn,
        'confusion_matrix_fp': fp,
        'confusion_matrix_fn': fn,
    }


def train_ann_experimental(X_train, y_train, X_validate, y_validate, model_name, exclude_column):


    X_train = X_train.drop(columns=exclude_column)
    X_validate = X_validate.drop(columns=exclude_column)

    np.random.seed(42)
    random.seed(42)
    tensorflow.random.set_seed(42)
    os.environ['PYTHONHASHSEED'] = str(42)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

    model = Sequential([
        Dense(30, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')  # Binary
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


    # For binary cross entropy, may need to change labels in pre processing
    y_train = y_train.replace({1: 0, 2: 1})
    y_validate = y_validate.replace({1: 0, 2: 1}) 

    model.fit( X_train, y_train, epochs=10, batch_size=64)

    y_pred_probs = model.predict(X_validate).ravel()
    predictions = np.round(y_pred_probs)

    print_model_name(model_name)
    accuracy = calculate_accuracy(y_validate, predictions)
    auc = print_auc_tf( model, X_validate, y_validate)

    confusion_matrix_values = print_and_return_confusion_matrix(y_validate, predictions)
    f1 = print_and_return_f1_score(y_validate, predictions)
    precision = print_and_return_precision(y_validate, predictions)
    recall = print_and_return_recall(y_validate, predictions)

    tn, fp, fn, tp = confusion_matrix_values.ravel()


    return {
        'accuracy': accuracy,
        'auc': auc,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'confusion_matrix_tp': tp,
        'confusion_matrix_tn': tn,
        'confusion_matrix_fp': fp,
        'confusion_matrix_fn': fn,
    }


def get_hyperparameter_ranges_ann(random_search, top_percentage=0.2):
    
    results_df = pd.DataFrame(random_search.cv_results_)
    top_results = results_df.sort_values('rank_test_score').head(int(len(results_df) * top_percentage))


    print('The Structure of the top results for hyperparameters')
    print(top_results)

    continuous_params_df = pd.DataFrame({
        'neurons': [
            round(top_results['param_neurons'].quantile(0.25)), 
            round(top_results['param_neurons'].quantile(0.75))
        ]
    })

   # Discrete parameters - selecting the most frequent value
    optimiser_value = top_results['optimizer'].value_counts().idxmax()
    activation_value = top_results['activation'].value_counts().idxmax()

    discrete_params_df = pd.DataFrame({
        'max_depth': [optimiser_value],
        'criterion': [activation_value]
    })


    return continuous_params_df, discrete_params_df


# def create_model(optimizer='adam', neurons=10, activation='relu', input_shape=None):
#     model = Sequential([
#         Dense(neurons, activation=activation, input_shape=(input_shape,)),
#         Dense(neurons, activation=activation),
#         Dense(1, activation='sigmoid')  # Binary
#     ])
#     model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
#     return model



# def train_ann_with_random_search(X_train, y_train, X_validate, y_validate, model_name, number_of_iterations):
    
#     scaler = StandardScaler()
#     X_train  = scaler.fit_transform(X_train)
#     X_validate  = scaler.transform(X_validate)

#     smote = SMOTE(random_state=42)
#     X_train_smote_and_scaled, y_train_smote = smote.fit_resample(X_train, y_train)

#     np.random.seed(42)
#     random.seed(42)
#     tensorflow.random.set_seed(42)
#     os.environ['PYTHONHASHSEED'] = str(42)
#     os.environ['TF_DETERMINISTIC_OPS'] = '1'

#     # Need to remember to standard scale
#     # And ensure, use smote

#     input_shape = X_train.shape[1]

#     # Define the hyperparameter space
#     param_dist = {
#         'optimizer': ['adam', 'sgd'],
#         'neurons': [10, 16, 30],
#         'activation': ['relu', 'tanh']
#     }

#     # Create the KerasClassifier
#     model = KerasClassifier(build_fn=lambda: create_model(input_shape=input_shape), epochs=10, batch_size=64, verbose=0)

#     # Random search
#     random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, 
#                                        n_iter=number_of_iterations, cv=2, random_state=42)
#     random_search.fit(X_train_smote_and_scaled, y_train_smote)

#     # Best model For predictions and metrics
#     best_model = random_search.best_estimator_.model

    
#     # For binary cross entropy, may need to change labels in pre processing
#     y_train_smote = y_train_smote.replace({1: 0, 2: 1})
#     y_validate = y_validate.replace({1: 0, 2: 1}) 

#     best_model.fit( X_train_smote_and_scaled, y_train_smote, epochs=10, batch_size=64)

#     y_pred_probs = best_model.predict(X_validate).ravel()
#     predictions = np.round(y_pred_probs)

#     print_model_name(model_name)
#     accuracy = calculate_accuracy(y_validate, predictions)
#     auc = print_auc_tf(best_model, X_validate, y_validate)

#     f1 = print_and_return_f1_score(y_validate, predictions)
#     precision = print_and_return_precision(y_validate, predictions)
#     recall = print_and_return_recall(y_validate, predictions)

    
#     continuous_params_df, discrete_params_df = get_hyperparameter_ranges(random_search)

#     print('Best Hyperparmeters For ANN')
#     print('Continous: ')
#     print(continuous_params_df)

#     print('Discrete')
#     print(discrete_params_df)

#     return {
#         'metrics': {
#             'accuracy': accuracy,
#             'auc': auc,
#             'f1_score': f1,
#             'precision': precision,
#             'recall': recall
#         },
#         'continuous_params': continuous_params_df,
#         'discrete_params': discrete_params_df
#     }



def get_best_hyperparameters_ann(top_trials): # 
    activation_list = []
    optimiser_list = []
    num_layers_list = []
    all_units_lists = {}

    # Extract hyperparameters from top trials
    for trial in top_trials:
        hps = trial.hyperparameters
        num_layers = hps.get('num_layers')
        num_layers_list.append(num_layers)

        for i in range(num_layers):
            units_key = 'units_' + str(i)
            activation_key = 'activation_' + str(i)

            if units_key not in all_units_lists:
                all_units_lists[units_key] = []
            all_units_lists[units_key].append(hps.get(units_key))

            activation_list.append(hps.get(activation_key))
        
        optimiser_list.append(hps.get('optimizer'))
        print(trial.hyperparameters.values)

    # Calculate ranges or most common values
    units_ranges = {key: [np.percentile(values, 25), np.percentile(values, 75)] for key, values in all_units_lists.items()}
    most_common_activation = max(set(activation_list), key=activation_list.count)
    most_common_optimiser = max(set(optimiser_list), key=optimiser_list.count)
    num_layers_range = [min(num_layers_list), max(num_layers_list)]  # range of number of layers

    hyperparameter_ranges_df = pd.DataFrame({
        'num_layers_range': [num_layers_range],
        'optimizer': [most_common_optimiser],
        'activation': [most_common_activation]
    })

    for key, value in units_ranges.items():
        hyperparameter_ranges_df[key] = [value]

    return hyperparameter_ranges_df




def train_ann_with_random_search(X_train, y_train, X_validate, y_validate, model_name, number_of_iterations): # Main Function
   
    # input_shape = X_train.shape[1]
    def create_model(hp):
        model = Sequential()
        
        num_layers = hp.Int('num_layers', 1, 20)
        print(f"Creating model with {num_layers} layers")  

        for i in range(num_layers):
            units = hp.Int('units_' + str(i), min_value=15, max_value=150, step=10)
            activation = hp.Choice('activation_' + str(i), values=['relu', 'tanh'])
            if i == 0:
                print(f"Adding layer {i+1} with {units} units, {activation} activation, and input shape {X_train.shape[1]}") 
                model.add(Dense(units=units, activation=activation, input_shape=(X_train.shape[1],)))
            else:
                print(f"Adding layer {i+1} with {units} units and {activation} activation")  
                model.add(Dense(units=units, activation=activation))

        model.add(Dense(1, activation='sigmoid'))
        print("Adding output layer with sigmoid activation")  

        optimizer = hp.Choice('optimizer', values=['adam', 'sgd'])
        print(f"Compiling model with {optimizer} optimizer")  
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        # model.batch_size = hp.Int('batch_size', min_value=16, max_value=128, step=16)

        return model



    
    scaler = StandardScaler()
    X_train  = scaler.fit_transform(X_train)
    X_validate  = scaler.transform(X_validate)

    

    smote = SMOTE(random_state=42)
    X_train_smote_and_scaled, y_train_smote = smote.fit_resample(X_train, y_train)

    np.random.seed(42)
    random.seed(42)
    tensorflow.random.set_seed(42)
    os.environ['PYTHONHASHSEED'] = str(42)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

 

    tuner = kt.RandomSearch(
        create_model,
        objective='val_accuracy',
        max_trials=number_of_iterations,
        executions_per_trial=1,
        directory='my_dir',
        project_name='ann_tuning'
    )

    tunerNumLayersAdded = kt.RandomSearch(
        create_model,
        objective='val_accuracy',
        max_trials=number_of_iterations,
        executions_per_trial=1,
        directory='new_dir', 
        project_name='ann_tuning_num_layers_added' 
    )

    tunerNumLayersAddedTwo = kt.RandomSearch(
        create_model,
        objective='val_accuracy',
        max_trials=number_of_iterations,
        executions_per_trial=1,
        directory='new_dir_two', 
        project_name='ann_tuning_num_layers_added_two' 
    )

    tunerNumLayersAddedThree = kt.RandomSearch(
        create_model,
        objective='val_accuracy',
        max_trials=number_of_iterations,
        executions_per_trial=1,
        directory='new_dir_three', 
        project_name='ann_tuning_num_layers_added_three' 
    )
    tunerNumLayersAddedFour = kt.RandomSearch(
        create_model,
        objective='val_accuracy',
        max_trials=number_of_iterations,
        executions_per_trial=1,
        directory='new_dir_four', 
        project_name='ann_tuning_num_layers_added_four' 
    )
    
    y_train_smote = y_train_smote.replace({1: 0, 2: 1})
    y_validate = y_validate.replace({1: 0, 2: 1}) 

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    tunerNumLayersAddedFour.search(X_train_smote_and_scaled, y_train_smote, 
                 epochs=100,  # large number of epochs and let early stopping halt the training
                 validation_data=(X_validate, y_validate),
                 callbacks=[early_stopping])

    top_percentage=0.2
    top_trials = tunerNumLayersAddedFour.oracle.get_best_trials(int(number_of_iterations * top_percentage))

    hyperparameter_ranges_df = get_best_hyperparameters_ann(top_trials)

    # Get the best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"Best hyperparameters: {best_hps.values}")

    # Build the model with the best hyperparameters and train it on the data
    best_model = tuner.hypermodel.build(best_hps)
    # best_batch_size = best_hps.get('batch_size')

    best_model.fit(X_train_smote_and_scaled, y_train_smote, 
                #    batch_size=best_batch_size, 
                   epochs=100,  
                   validation_data=(X_validate, y_validate),
                   callbacks=[early_stopping])


    y_pred_probs = best_model.predict(X_validate).ravel()
    predictions = np.round(y_pred_probs)

    print_model_name(model_name)
    accuracy = calculate_accuracy(y_validate, predictions)
    auc = print_auc_tf(best_model, X_validate, y_validate)

    f1 = print_and_return_f1_score(y_validate, predictions)
    precision = print_and_return_precision(y_validate, predictions)
    recall = print_and_return_recall(y_validate, predictions)

    best_hyperparameters_df = pd.DataFrame([best_hps.values])

    return {
        'metrics': {
            'accuracy': accuracy,
            'auc': auc,
            'f1_score': f1,
            'precision': precision,
            'recall': recall
        },
        'best_hyperparameters': hyperparameter_ranges_df 
    }

def train_ann_with_grid_search(X_train, y_train, X_validate, y_validate, model_name):
    # Define the ANN model
    
    # np.random.seed(42)
    # random.seed(42)
    # tensorflow.random.set_seed(42)
    # os.environ['PYTHONHASHSEED'] = str(42)
    # os.environ['TF_DETERMINISTIC_OPS'] = '1'

    param_grid = {
    'num_layers': [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    'optimizer': ['sgd'],
    'activation': ['tanh'],
    'units_4': [100, 110, 120, 137.5],
    'units_5': [60, 80, 100, 115],
    'units_6': [35, 50, 65, 80],
    'units_7': [25, 50, 72.5],
    'units_8': [67.5, 85, 105],
    'units_9': [35, 60, 90, 125],
    'units_10': [60, 80, 100, 125],
    'units_11': [77.5, 85, 100, 102.5],
    'units_12': [27.5, 40, 52.5],
    'units_13': [40, 60, 75, 90],
    'units_14': [30, 35, 40],
    'units_15': [67.5, 70, 72.5],
    'units_16': [65],
    'units_17': [85],
    'units_18': [135]
}
    
    def create_model(num_layers, optimizer='sgd', activation='tanh', **kwargs):
        model = Sequential()
        for i in range(4, num_layers + 4):  
            units = kwargs.get(f'units_{i}', 50)  
            if i == 4:  
                model.add(Dense(units=units, activation=activation, input_shape=(X_train.shape[1],)))
            else:
                model.add(Dense(units=units, activation=activation))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
        return model

    
    # Scaling and SMOTE
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_validate_scaled = scaler.transform(X_validate)
    
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

    # KerasClassifier wrapper
    model = KerasClassifier(
        build_fn=create_model,
        epochs=100,
        verbose=1,
        num_layers=5,  
        optimizer='sgd',  
        activation='tanh',  
        units_4= [100, 110, 120, 137.5],
        units_5= [60, 80, 100, 115],
        units_6= [35, 50, 65, 80],
        units_7= [25, 50, 72.5],
        units_8= [67.5, 85, 105],
        units_9= [35, 60, 90, 125],
        units_10= [60, 80, 100, 125],
        units_11= [77.5, 85, 100, 102.5],
        units_12= [27.5, 40, 52.5],
        units_13= [40, 60, 75, 90],
        units_14= [30, 35, 40],
        units_15= [67.5, 70, 72.5],
        units_16= [65],
        units_17= [85],
        units_18= [135] 
    )
    y_train_smote = y_train_smote.replace({1: 0, 2: 1})
    y_validate = y_validate.replace({1: 0, 2: 1}) 
    # Grid Search
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=10, scoring='accuracy', verbose=1, error_score='raise')
    grid_result = grid.fit(X_train_smote, y_train_smote)
    print('Grid Result')
    print(grid_result.cv_results_)
    # Best model
    # Check if best model is found
    print('Best estimator direct access')
    print(grid_result.best_estimator_.model_)

    best_model = grid_result.best_estimator_.model_
    print('Best model found.')
    print(type(best_model))



    # Evaluation
    y_pred_probs = best_model.predict(X_validate_scaled).ravel()
    predictions = np.round(y_pred_probs)
    print(model_name)
    accuracy = calculate_accuracy(y_validate, predictions)
    # auc = roc_auc_score(y_validate, y_pred_probs)
    auc = print_auc_tf(best_model, X_validate, y_validate)
    f1 = print_and_return_f1_score(y_validate, predictions)
    precision = print_and_return_precision(y_validate, predictions)
    recall = print_and_return_recall(y_validate, predictions)

    best_hyperparameters_df = pd.DataFrame([grid_result.best_params_])
    return {
        'metrics': {
            'accuracy': accuracy,
            'auc': auc,
            'f1_score': f1,
            'precision': precision,
            'recall': recall
        },
        'best_hyperparameters': best_hyperparameters_df,
        'best_model': best_model
    }



def train_ann_experimental_scaled(X_train, y_train, X_validate, y_validate, model_name):

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_validate_scaled = scaler.transform(X_validate)

    np.random.seed(42)
    random.seed(42)
    tensorflow.random.set_seed(42)
    os.environ['PYTHONHASHSEED'] = str(42)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

    model = Sequential([
        Dense(30, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')  # Binary
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    y_train = y_train.replace({1: 0, 2: 1})
    y_validate = y_validate.replace({1: 0, 2: 1}) 

    model.fit(X_train_scaled, y_train, epochs=10, batch_size=64)

    y_pred_probs = model.predict(X_validate_scaled).ravel()
    predictions = np.round(y_pred_probs)

    print_model_name(model_name)
    accuracy = calculate_accuracy(y_validate, predictions)
    auc = print_auc_tf(model, X_validate_scaled, y_validate)
    confusion_matrix_values = print_and_return_confusion_matrix(y_validate, predictions)
    f1 = print_and_return_f1_score(y_validate, predictions)
    precision = print_and_return_precision(y_validate, predictions)
    recall = print_and_return_recall(y_validate, predictions)
    tn, fp, fn, tp = confusion_matrix_values.ravel()

    return {
        'accuracy': accuracy,
        'auc': auc,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'confusion_matrix_tp': tp,
        'confusion_matrix_tn': tn,
        'confusion_matrix_fp': fp,
        'confusion_matrix_fn': fn,
    }



def train_decision_tree_experimental_scaled(X_train, y_train, X_validate, y_validate, model_name):
    
    # # Drop excluded columns
    # X_train = X_train.drop(columns=exclude_column)
    # X_validate = X_validate.drop(columns=exclude_column)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_validate_scaled = scaler.transform(X_validate)

    assert 'Period' not in X_train.columns, "Period Column should not be in the training set"
    assert 'Period' not in X_validate.columns, "Period Column should not be in the validation set"

    decision_tree = DecisionTreeClassifier(random_state=42)

    decision_tree.fit(X_train_scaled, y_train)


    predictions = decision_tree.predict(X_validate_scaled)

    print_model_name(model_name)
    accuracy = calculate_accuracy(y_validate, predictions)
    auc = print_auc(decision_tree, X_validate_scaled, y_validate)

    confusion_matrix_values = print_and_return_confusion_matrix(y_validate, predictions)
    f1 = print_and_return_f1_score(y_validate, predictions)
    precision = print_and_return_precision(y_validate, predictions)
    recall = print_and_return_recall(y_validate, predictions)

    tn, fp, fn, tp = confusion_matrix_values.ravel()

    report = store_and_print_classification_report(y_validate, predictions)
    return {
        'accuracy': accuracy,
        'auc': auc,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'confusion_matrix_tp': tp,
        'confusion_matrix_tn': tn,
        'confusion_matrix_fp': fp,
        'confusion_matrix_fn': fn,
    }

def train_svm_experimental_scaled(X_train, y_train, X_validate, y_validate, model_name):

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_validate_scaled = scaler.transform(X_validate)

    svm_model = SVC(kernel='linear', probability=True)    
    svm_model.fit(X_train_scaled , y_train)
    predictions = svm_model.predict(X_validate_scaled)

    print_model_name(model_name)
    accuracy = calculate_accuracy(y_validate, predictions)
    auc = print_auc( svm_model, X_validate_scaled, y_validate)

    confusion_matrix_values = print_and_return_confusion_matrix(y_validate, predictions)
    f1 = print_and_return_f1_score(y_validate, predictions)
    precision = print_and_return_precision(y_validate, predictions)
    recall = print_and_return_recall(y_validate, predictions)

    print("Confusion Matrix:\n", confusion_matrix_values)
    tn, fp, fn, tp = confusion_matrix_values.ravel()

    return {
        'accuracy': accuracy,
        'auc': auc,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'confusion_matrix_tp': tp,
        'confusion_matrix_tn': tn,
        'confusion_matrix_fp': fp,
        'confusion_matrix_fn': fn,
    }


def train_ann_experimental_feature_selected(X_train, y_train, X_validate, y_validate, model_name, important_features):

    ## Important features from Logistic regression rfe
    important_features = important_features[important_features['Ranking'] == 1]['Feature'].tolist()

    X_train = X_train[important_features]
    X_validate = X_validate[important_features]

    # dropped in split
    # X_train = X_train.drop(columns=exclude_column)
    # X_validate = X_validate.drop(columns=exclude_column)

    np.random.seed(42)
    random.seed(42)
    tensorflow.random.set_seed(42)
    os.environ['PYTHONHASHSEED'] = str(42)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

    model = Sequential([
        Dense(30, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')  # Binary
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


    # For binary cross entropy, may need to change labels in pre processing
    y_train = y_train.replace({1: 0, 2: 1})
    y_validate = y_validate.replace({1: 0, 2: 1}) 

    model.fit( X_train, y_train, epochs=10, batch_size=64)

    y_pred_probs = model.predict(X_validate).ravel()
    predictions = np.round(y_pred_probs)

    print_model_name(model_name)
    accuracy = calculate_accuracy(y_validate, predictions)
    auc = print_auc_tf( model, X_validate, y_validate)

    confusion_matrix_values = print_and_return_confusion_matrix(y_validate, predictions)
    f1 = print_and_return_f1_score(y_validate, predictions)
    precision = print_and_return_precision(y_validate, predictions)
    recall = print_and_return_recall(y_validate, predictions)

    tn, fp, fn, tp = confusion_matrix_values.ravel()


    return {
        'accuracy': accuracy,
        'auc': auc,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'confusion_matrix_tp': tp,
        'confusion_matrix_tn': tn,
        'confusion_matrix_fp': fp,
        'confusion_matrix_fn': fn,
    }



def train_logistic_regression_experimental_rfe(X_train, y_train, X_validate, y_validate, model_name, number_of_features_to_select, exclude_column):
    X_train = X_train.drop(columns=exclude_column)
    X_validate = X_validate.drop(columns=exclude_column)
    
    logistic_regression_model = LogisticRegression(random_state=42)

    # RFE
    rfe = RFE(estimator=logistic_regression_model, n_features_to_select=number_of_features_to_select)
    rfe.fit(X_train, y_train)

    X_train_rfe = X_train.iloc[:, rfe.support_]
    X_validate_rfe = X_validate.iloc[:, rfe.support_]

    # logistic_regression_model.fit(X_train_rfe, y_train)

    # predictions = logistic_regression_model.predict(X_validate_rfe)

    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_train_rfe_smote, y_train_smote = smote.fit_resample(X_train_rfe, y_train)

    logistic_regression_model.fit(X_train_rfe_smote, y_train_smote)
    predictions = logistic_regression_model.predict(X_validate_rfe)

    # Identifying selected features
    selected_features = pd.DataFrame({
        'Feature': X_train.columns[rfe.support_],
        'Ranking': rfe.ranking_[rfe.support_]
    })
   
    print_model_name(model_name)
    accuracy = calculate_accuracy(y_validate, predictions)
    auc = print_auc_tf(logistic_regression_model, X_validate_rfe, y_validate)

    confusion_matrix_values = print_and_return_confusion_matrix(y_validate, predictions)
    f1 = print_and_return_f1_score(y_validate, predictions)
    precision = print_and_return_precision(y_validate, predictions)
    recall = print_and_return_recall(y_validate, predictions)

    tn, fp, fn, tp = confusion_matrix_values.ravel()

    metrics = {
        'accuracy': accuracy,
        'auc': auc,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'confusion_matrix_tp': tp,
        'confusion_matrix_tn': tn,
        'confusion_matrix_fp': fp,
        'confusion_matrix_fn': fn
    }
    
    return metrics, selected_features


### Interpretability ###################################################



### Final Mdoels #######################################################



def get_best_hyperparameters_decision_tree(grid_search): # Main Support
    best_params = grid_search.best_params_

    print(best_params)
    best_hyperparameters_df = pd.DataFrame(best_params, index=[0])

    return best_hyperparameters_df

def train_decision_tree_with_grid_search(X_train, y_train, X_validate, y_validate, model_name): # Main Function   

    param_grid = {
        "decisiontreeclassifier__max_depth": [20],
        "decisiontreeclassifier__min_samples_split": range(10, 38),
        "decisiontreeclassifier__min_samples_leaf": range(33, 43),
        "decisiontreeclassifier__criterion": ["entropy"]
    }

    pipeline = make_pipeline_imb(StandardScaler(), SMOTE(random_state=42), 
                                 DecisionTreeClassifier(random_state=42))
    # GridSearchCV uses statified crossvalidation by default.
    grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=10, scoring='accuracy', verbose=1)

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    predictions = best_model.predict(X_validate)

    print_model_name(model_name)
    accuracy = calculate_accuracy(y_validate, predictions)
    auc = print_auc(best_model, X_validate, y_validate)
    f1 = print_and_return_f1_score(y_validate, predictions)
    precision = print_and_return_precision(y_validate, predictions)
    recall = print_and_return_recall(y_validate, predictions)

    

    # Get hyperparameter ranges
    best_params_df = get_best_hyperparameters_decision_tree(grid_search)
    confusion_matrix_values = print_and_return_confusion_matrix(y_validate, predictions)

    tn, fp, fn, tp = confusion_matrix_values.ravel()
    # Store actual max depth 
    actual_max_depth = grid_search.best_estimator_.named_steps['decisiontreeclassifier']
    actual_max_depth = actual_max_depth.tree_.max_depth

    print('Decision Tree depth:', actual_max_depth)
    
    extracted_model = best_model.named_steps['decisiontreeclassifier']
    explainer = shap.Explainer(extracted_model, X_train)

    # Compute SHAP values
    shap_values = explainer(X_train)
    print(type(shap_values))
    print(shap_values.shape)


    return {
        'metrics': {
            'accuracy': accuracy,
            'auc': auc,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'confusion_matrix_tp': tp,
            'confusion_matrix_tn': tn,
            'confusion_matrix_fp': fp,
            'confusion_matrix_fn': fn,
            'decision_tree_actual_max_depth': actual_max_depth
        },
        'best_hyperparameters': best_params_df,
        'best_model': best_model
    }








### Evaluation ##########################################################

def print_model_name(model_name):
    print(f"Evaluation Metrics: {model_name}")

def calculate_accuracy(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    return accuracy

def store_and_print_classification_report(y_test, y_pred):
    report = classification_report(y_test, y_pred)
    print(f"Classification Report:\n{report}")
    return report

def print_and_return_confusion_matrix(y_test, y_pred):
    matrix = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:\n{matrix}")
    return matrix

def print_and_return_f1_score(y_test, y_pred):
    f1 = f1_score(y_test, y_pred)
    print(f"F1 Score: {f1}")
    return f1

def print_and_return_precision(y_test, y_pred):
    precision = precision_score(y_test, y_pred)
    print(f"Precision: {precision}")
    return precision

def print_and_return_recall(y_test, y_pred):
    recall = recall_score(y_test, y_pred)
    print(f"Recall: {recall}")
    return recall

def print_auc(model, X_test, y_test):
    """
    Prints the AUC for the given model and test data.
    """
    # Probabilities for the positive class
    probas = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, probas)

    print(f"AUC: {roc_auc}")
    return roc_auc

def print_auc_tf(model, X_test, y_test):
    """
    Prints the AUC for the given model and test data.
    """
    # Probabilities for the positive class
    probas = model.predict(X_test).ravel()

    roc_auc = roc_auc_score(y_test, probas)

    print(f"AUC: {roc_auc}")
    return roc_auc


#### Sampling

def smote_oversample_minority_class(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    return X_train_smote, y_train_smote




accuracy_scorer = make_scorer(calculate_accuracy)
auc_scorer = make_scorer(print_auc, needs_proba=True, needs_threshold=False)
f1_scorer = make_scorer(print_and_return_f1_score)
precision_scorer = make_scorer(print_and_return_precision)
recall_scorer = make_scorer(print_and_return_recall)

def evaluate_decision_tree_depths(X_train, y_train, initial_max_depth, min_depth): # Main function
    results = []

    # Define the cross-validation 10-fold stratified
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    # Define the scoring functions
    scoring = {
        'accuracy': accuracy_scorer,
        'auc': auc_scorer,
        'f1': f1_scorer,
        'precision': precision_scorer,
        'recall': recall_scorer
    }
    for depth in range(initial_max_depth, min_depth - 1, -1):
        pipeline = make_pipeline_imb(
            StandardScaler(),
            SMOTE(random_state=42),
            DecisionTreeClassifier(
                criterion='entropy',
                max_depth=depth,
                min_samples_split=10,
                min_samples_leaf=42,
                random_state=42
            )
        )

        # Evaluate the model using cross-validation
        cv_results = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring)

        # Store the results
        results.append({
            'max_depth': depth,
            'cv_accuracy': np.mean(cv_results['test_accuracy']),
            'cv_auc': np.mean(cv_results['test_auc']),
            'cv_f1': np.mean(cv_results['test_f1']),
            'cv_precision': np.mean(cv_results['test_precision']),
            'cv_recall': np.mean(cv_results['test_recall'])
        })
        results_df = pd.DataFrame(results)

    return results_df
