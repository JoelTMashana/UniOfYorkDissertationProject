import pandas as pd
from collections import defaultdict
from kedro.pipeline import node
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from yellowbrick.cluster import KElbowVisualizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import tensorflow 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import statsmodels.api as sm
from sklearn.model_selection import RandomizedSearchCV


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
    data = data.drop(['Clusters'], axis=1)

    
    assert 'Risk Level' in data.columns, "Risk Level column does not exist."
    assert 'Clusters' not in data.columns, "Clusters column should not exist after dropping it."
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




# ###### ML Algorithms

def split_train_test_validate(data, target_column):
    X = data
    y = data[target_column]
    X = data.drop(columns=[target_column])

    X_train_temp, X_test, y_train_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train, X_validate, y_train, y_validate = train_test_split(X_train_temp, y_train_temp, test_size=0.25, random_state=42)  

    return X_train, X_validate, X_test, y_train, y_validate, y_test

def train_decision_tree(X_train, y_train, X_validate, y_validate, model_name, number_of_iterations):
    
    param_dist = {
        "max_depth": [3, 10, None],
        "min_samples_split": range(2, 11),
        "min_samples_leaf": range(1, 11),
        "criterion": ["gini", "entropy"]
    }
    # X = data.drop(target_column, axis=1)
    # y = data[target_column]

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    decision_tree = DecisionTreeClassifier(random_state=42)
    random_search = RandomizedSearchCV(decision_tree, param_distributions=param_dist, 
                                       n_iter=number_of_iterations, cv=5, random_state=42)
    random_search.fit(X_train, y_train)

    best_model = random_search.best_estimator_

    predictions = best_model.predict(X_validate)

    print_model_name(model_name)
    accuracy = calculate_accuracy(y_validate, predictions)
    report = store_and_print_classification_report(y_validate, predictions)
    auc = print_auc(best_model, X_validate, y_validate)

    return best_model, pd.DataFrame({'accuracy': [accuracy]}), pd.DataFrame({'auc': [auc]}), pd.DataFrame({'report': [report]})


def train_logistic_regression(data, target_column, model_name):
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    logistic_regression_model = LogisticRegression()

    logistic_regression_model.fit(X_train, y_train)

    # Predicting on the test data
    y_pred = logistic_regression_model.predict(X_test)

    print_model_name(model_name)
    calculate_accuracy(y_test, y_pred)
    store_and_print_classification_report(y_test, y_pred)
    print_auc(logistic_regression_model, X_test, y_test)

    return logistic_regression_model



def train_svm(data, target_column, model_name):
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    svm_model = SVC(probability=True,random_state=42)

    svm_model.fit(X_train, y_train)

    y_pred = svm_model.predict(X_test)

    print_model_name(model_name)
    calculate_accuracy(y_test, y_pred)
    store_and_print_classification_report(y_test, y_pred)
    print_auc(svm_model, X_test, y_test)

    return svm_model


def train_ann(data, target_column, columns_to_exclude, model_name):
    X = data.drop(target_column, axis=1)
    X=  data.drop(columns=[columns_to_exclude])
    y = data[target_column]


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')  # Binary
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=10, batch_size=32)

    print_model_name(model_name)
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Loss: {loss}, Accuracy: {accuracy}")


    return model




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
    # probabilities for the positive class
    probas = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, probas)

    print(f"AUC: {roc_auc}")
    return roc_auc
