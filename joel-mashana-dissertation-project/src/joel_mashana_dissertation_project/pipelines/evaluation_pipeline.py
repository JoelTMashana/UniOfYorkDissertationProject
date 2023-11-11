from kedro.pipeline import Pipeline, node
from .nodes.data_preprocessing import (filter_data_on_supplychain_finance, extract_payment_periods, create_period_column,
                                       remove_redundant_columns, anonymise_data, encode_column, align_columns,
                                       prepare_inflation_data, get_average_inflation_for_periods,
                                       gdp_remove_headers, process_gdp_averages, combine_datasets, convert_float_columns_to_int,
                                       mean_imputation, robust_scale_column, perform_kmeans_clustering, scale_and_apply_pca,
                                       train_decision_tree
                                       )


def create_pipeline(**kwargs):


    return Pipeline(
        [ 

        ]
    )
