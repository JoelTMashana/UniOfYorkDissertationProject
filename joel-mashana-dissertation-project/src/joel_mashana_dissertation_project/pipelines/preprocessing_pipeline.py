from kedro.pipeline import Pipeline, node
from .nodes.data_preprocessing import (filter_data_on_supplychain_finance, extract_payment_periods, create_period_column,
                                       remove_redundant_columns, anonymise_data, encode_column, align_columns,
                                       prepare_inflation_data, get_average_inflation_for_periods,
                                       gdp_remove_headers, process_gdp_averages, combine_datasets, convert_float_columns_to_int,
                                       mean_imputation, robust_scale_column, perform_kmeans_clustering, scale_and_apply_pca,
                                       train_decision_tree, train_logistic_regression, train_svm, train_ann, calculate_accuracy,
                                       split_train_test_validate
                                       )
def create_pipeline(**kwargs):
    
    filter_buyer_payment_practises_on_supply_chain_finance_node = node(
                filter_data_on_supplychain_finance,
                inputs= {
                   "data": "buyer_payment_behaviour_in",
                   "year": "params:earliest_year_filter"
                },
                outputs="buyer_payment_practices_out",
                name="filter_buyer_payment_practises_on_supply_chain_finance_node"
            )
    create_period_column_node = node(
            create_period_column,  
            inputs="buyer_payment_practices_out",
            outputs="buyer_payment_practices_with_period_col", 
            name="create_period_column_node" 
            )
    extract_payment_periods_node = node(
            extract_payment_periods,
            inputs="buyer_payment_practices_out",  
            outputs="buyer_payment_practices_payment_periods_out",
            name="extract_payment_periods_node"
            )
    
    remove_redundant_columns_node = node(
            remove_redundant_columns,
            inputs="buyer_payment_practices_with_period_col",
            outputs="buyer_payment_practices_payment_redudent_columns_removed",
            name="remove_redundant_columns_node"  
    )
    
    anonymise_data_node = node(
        anonymise_data,
        inputs="buyer_payment_practices_payment_redudent_columns_removed",
        outputs="buyer_payment_practices_anonymised",
        name="anonymise_data_node"
    )

    encode_column_payments_made_in_the_reporting_period = node(
        encode_column,
        inputs={
        "data": "buyer_payment_practices_anonymised",
        "columns_to_encode": "params:columns_to_encode"
        },
        outputs="buyer_payment_practices_boolean_columns_encoded",
        name="encode_column_payments_made_in_the_reporting_period_node"
    )

    align_columns_node = node(
        align_columns,
        inputs={
            "data": "buyer_payment_practices_boolean_columns_encoded",
            "column_one": "params:column_one",
            "column_two": "params:column_two",
        },

        outputs="buyer_payment_practices_filtered_encoded_final",
        name="align_columns_node"
    )

    prepare_inflation_data_node = node(
        prepare_inflation_data,
        inputs={
            "data": "inflation_rates"
            # "start_date": "params:inflation_rate_data_start_date"  
        },
        outputs="inflation_rates_prepared_for_average_caculation",
        name="prepare_inflation_data_node"
    )

    inflation_rates_averages_node = node(
        get_average_inflation_for_periods,
        inputs={
            "data": "inflation_rates_prepared_for_average_caculation",
            "periods": "buyer_payment_practices_payment_periods_out"
        },
        outputs="inflation_rates_averages_forward_filled",
        name="inflation_rates_averages_node"
    )


    monthly_gdp_headers_removed_node = node(
        gdp_remove_headers,
        inputs="monthly_gdp",
        outputs="monthly_gdp_headers_removed",
        name = "monthly_gdp_headers_removed_node"
    )

    calculate_monthly_gdp_averages_node = node(
        process_gdp_averages,
        inputs={
            "data": "monthly_gdp_headers_removed",
            "payment_periods": "buyer_payment_practices_payment_periods_out"
        },
        outputs="monthly_gdp_averages",
        name="calculate_monthly_gdp_averages_node"
    )

    combine_datasets_node = node(
        combine_datasets,
        inputs = {
            "payment_practices": "buyer_payment_practices_filtered_encoded_final",
            "gdp_averages": "monthly_gdp_averages"
        },
        outputs="combined_data_set",
        name = "combine_datasets_node"
    )

    convert_payment_practise_column_data_to_floats_node = node(
        convert_float_columns_to_int,
        inputs="combined_data_set",
        outputs="combined_data_with_appropriate_cols_converted_to_integers",
        name="convert_payment_practise_column_data_to_floats_node"
    )

    peform_mean_imputation_on_combined_dataset_node = node(
        mean_imputation,
        inputs= {
            "data": "combined_data_with_appropriate_cols_converted_to_integers",
            "exclude_column": "params:columns_to_exclude_for_imputation"
        },
        outputs="combined_data_set_mean_imputed",
        name="peform_mean_imputation_on_combined_dataset_node"
    )

    robust_scale_percentage_invoices_not_paid_on_agreed_terms_column_node = node(
        robust_scale_column,
        inputs= {
            "data": "combined_data_set_mean_imputed",
            "column_name": "params:columns_to_include_for_robust_scaling",
        },
        outputs="combined_data_set_invoices_missed_robust_scaled",
        name="robust_scale_percentage_invoices_not_paid_on_agreed_terms_column_node"
    )

    determine_and_assign_risk_levels_node = node(
        perform_kmeans_clustering,
        inputs = {
            "data": "combined_data_set_mean_imputed", # Needs to change to scaled data pontentially
            "column_to_cluster": "params:column_for_clustering"
        },
        outputs = "combined_data_set_with_risk_levels",
        name="determine_and_assign_risk_levels_node"
    )

  
    apply_principle_component_analysis_node = node(
        scale_and_apply_pca,
        inputs= {
            "data": "combined_data_set_with_risk_levels",
            "n_components": "params:number_of_components_to_retain",
            "target_column": "params:target",
            "columns_to_exclude": "params:columns_to_exclude"
        },
        outputs="combined_data_set_pca_applied",
        name="apply_principle_component_analysis_node"
    )

    split_data_train_test_validate_node = node(
        split_train_test_validate,
         inputs = {
            "data": "combined_data_set_pca_applied",
            "target_column": "params:target"
        },
        outputs = ["X_train","X_validate", " X_test", "y_train", "y_validate", "y_test"],
        name="split_data_train_test_validate_node"
    )

    execute_decision_tree_node = node(
        train_decision_tree,
        inputs={
            "X_train": "X_train",
            "y_train": "y_train",
            "X_validate": "X_validate",
            "y_validate": "y_validate",
            "model_name":  "params:decision_tree",
            "number_of_iterations": "params:number_of_iterations_randomised_search"
        },
        outputs=[
            "decision_tree_model", 
            "decision_tree_performance_metric_accuracy",
            "decision_tree_performance_metric_auc",
            "decision_tree_performance_metric_report"
            ],
        name="execute_decision_tree_node"
    )

    # calculate_decision_tree_accuracy_node = node(
    #     calculate_accuracy,
    #     inputs={
    #         "data": "combined_data_set_pca_applied",
    #         "target_column": "params:target",
    #         "model_name":  "params:decision_tree"
    #     },
    #     outputs="decision_tree_model",
    #     name="execute_decision_tree_node"
    # )

    execute_logistic_regression_node = node(
        train_logistic_regression,
        inputs={
            "data": "combined_data_set_pca_applied",
            "target_column": "params:target",
            "model_name":  "params:logistic_regression"
        },
        outputs="logistic_regression_model",
        name="execute_logistic_regression_node"
    )

    execute_svm_node = node(
    func=train_svm,
    inputs={
        "data": "combined_data_set_pca_applied",
        "target_column": "params:target",
        "model_name": "params:svm"
    },
    outputs="svm_model",
    name="execute_svm_node"
    )

    execute_ann_node = node(
        train_ann,
        inputs={
            "data": "combined_data_set_with_risk_levels",
            "target_column": "params:target",
            "columns_to_exclude": "params:columns_to_exclude",
            "model_name":  "params:ann"
        },
        outputs="ann_model",
        name="execute_ann_node"
    )


    return Pipeline(
        [ 
           filter_buyer_payment_practises_on_supply_chain_finance_node,
           create_period_column_node,
           extract_payment_periods_node,
           remove_redundant_columns_node,
           anonymise_data_node,
           encode_column_payments_made_in_the_reporting_period,
           align_columns_node,
        #    prepare_inflation_data_node,
        #    inflation_rates_averages_node,
           monthly_gdp_headers_removed_node,
           calculate_monthly_gdp_averages_node,
           combine_datasets_node,
           convert_payment_practise_column_data_to_floats_node,
           peform_mean_imputation_on_combined_dataset_node,
        #    robust_scale_percentage_invoices_not_paid_on_agreed_terms_column_node,
           determine_and_assign_risk_levels_node,
           apply_principle_component_analysis_node,
           split_data_train_test_validate_node,
           ### Excute models
           execute_decision_tree_node,
        #    execute_logistic_regression_node,
        #    execute_svm_node,
        #    execute_ann_node,
       
        ]
    )
