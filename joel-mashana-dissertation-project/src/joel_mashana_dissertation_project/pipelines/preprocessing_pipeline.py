from kedro.pipeline import Pipeline, node
from .nodes.data_preprocessing import (filter_data_on_supplychain_finance, extract_payment_periods, create_period_column,
                                       remove_redundant_columns, anonymise_data, encode_column, align_columns,
                                       prepare_inflation_data, get_average_inflation_for_periods,
                                       gdp_remove_headers, process_gdp_averages, combine_datasets, convert_float_columns_to_int,
                                       mean_imputation, robust_scale_column, perform_kmeans_clustering, scale_and_apply_pca,
                                       train_decision_tree, train_logistic_regression, train_svm, train_ann, calculate_accuracy,
                                       split_train_test_validate,  smote_oversample_minority_class, standard_scale_data,
                                       train_logistic_regression_for_rfe, split_train_test_validate_rfe, train_decision_tree_experimental,
                                       train_svm_experimental, train_ann_experimental,
                                       train_logistic_regression_experimental_rfe, train_logistic_regression_experimental, split_train_test_validate_smote_applied,
                                       train_ann_experimental_feature_selected, train_ann_experimental_scaled, train_decision_tree_experimental_scaled,
                                       train_svm_experimental_scaled, split_train_test_validate_smote_applied_varied_splits, main_split_train_test_validate,
                                       train_decision_tree_with_random_search, train_svm_with_random_search, train_ann_with_random_search,
                                       train_decision_tree_with_grid_search, evaluate_decision_tree_depths, train_ann_with_grid_search, train_ann_with_fixed_hyperparameters,
                                       evaluate_and_return_final_decision_tree_model, train_svm_with_grid_search, evaluate_and_return_final_svm_model
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

    
    determine_and_assign_risk_levels_buyer_data_node = node(
        perform_kmeans_clustering,
        inputs = {
            "data": "buyer_payment_practices_filtered_encoded_final", 
            "column_to_cluster": "params:column_for_clustering"
        },
        outputs = "buyer_payement_practise_data_with_risk_levels",
        name="determine_and_assign_risk_levels_buyer_data_node"
    )

    mean_imputation_for_experimental_data_node = node(
        mean_imputation,
        inputs= {
            "data": "buyer_payement_practise_data_with_risk_levels",
            "exclude_column": "params:columns_to_exclude_for_imputation"
        },
        outputs="buyer_payment_practice_dataset_mean_imputed",
        name="mean_imputation_for_experimental_data_node"
    )

    mean_imputation_node = node(
        mean_imputation,
        inputs= {
            "data": "buyer_payement_practise_data_with_risk_levels",
            "exclude_column": "params:columns_to_exclude_for_imputation"
        },
        outputs="buyer_payment_practice_dataset_mean_imputed",
        name="mean_imputation_node"
    )

    initial_data_splitting_for_experiments = node (
        split_train_test_validate,
         inputs = {
            "data": "buyer_payment_practice_dataset_mean_imputed",
            "target_column": "params:target"
        },
        outputs = ["X_train_experimental","X_validate_experimental", " X_test_experimental", "y_train_experimental", "y_validate_experimental", "y_test_experimental"],
        name="initial_data_splitting_for_experiments"
    )

    initial_data_splitting_for_smote_applied = node (
        split_train_test_validate_smote_applied,
        inputs = {
            "data": "buyer_payment_practice_dataset_mean_imputed",
            "target_column": "params:target",
            "columns_to_exclude": "params:columns_to_exclude"
        },
        outputs = ["X_train_experimental","X_validate_experimental", " X_test_experimental", "y_train_experimental", "y_validate_experimental", "y_test_experimental"],
        name="initial_data_splitting_for_smote_applied"
    )

    # experiment_decision_tree_buyer_data_only = node (
    #     train_decision_tree_experimental,
    #     inputs={
    #         "X_train": "X_train_experimental",
    #         "y_train": "y_train_experimental",
    #         "X_validate": "X_validate_experimental",
    #         "y_validate": "y_validate_experimental",
    #         "model_name":  "params:decision_tree",
    #         "exclude_column": "params:period"
    #     },
    #     outputs="metrics",
    #     name="execute_decision_tree_node"
    # )

    experiment_logistic_regression_buyer_data_only = node (
        train_logistic_regression_experimental,
        inputs={
            "X_train": "X_train_experimental",
            "y_train": "y_train_experimental",
            "X_validate": "X_validate_experimental",
            "y_validate": "y_validate_experimental",
            "model_name":  "params:logistic_regression",
            "exclude_column": "params:period"
        },
        outputs="metrics",
        name="experiment_logistic_regression_buyer_data_only"
    )

    experiment_svm_buyer_data_only = node (
        train_svm_experimental,
        inputs={
            "X_train": "X_train_experimental",
            "y_train": "y_train_experimental",
            "X_validate": "X_validate_experimental",
            "y_validate": "y_validate_experimental",
            "model_name":  "params:svm",
            "exclude_column": "params:period"
        },
        outputs="metrics",
        name="experiment_svm_buyer_data_only"
    )
    
    experiment_ann_buyer_data_only = node (
        train_ann_experimental,
        inputs={
            "X_train": "X_train_experimental",
            "y_train": "y_train_experimental",
            "X_validate": "X_validate_experimental",
            "y_validate": "y_validate_experimental",
            "model_name":  "params:ann",
            "exclude_column": "params:period"
        },
        outputs="metrics",
        name="experiment_ann_buyer_data_only"
    )


    experiment_decision_tree_buyer_data_only_scaled = node (
        train_decision_tree_experimental_scaled,
        inputs={
            "X_train": "X_train_experimental",
            "y_train": "y_train_experimental",
            "X_validate": "X_validate_experimental",
            "y_validate": "y_validate_experimental",
            "model_name":  "params:decision_tree"
        },
        outputs="metrics",
        name="experiment_decision_tree_buyer_data_only_scaled"
    )

    experiment_logistic_regression_buyer_data_only_scaled  = node (
        train_logistic_regression_experimental,
        inputs={
            "X_train": "X_train_experimental",
            "y_train": "y_train_experimental",
            "X_validate": "X_validate_experimental",
            "y_validate": "y_validate_experimental",
            "model_name":  "params:logistic_regression",
            "exclude_column": "params:period"
        },
        outputs="metrics",
        name="experiment_logistic_regression_buyer_data_only_scaled"
    )

    experiment_svm_buyer_data_only_scaled = node (
        train_svm_experimental_scaled,
        inputs={
            "X_train": "X_train_experimental",
            "y_train": "y_train_experimental",
            "X_validate": "X_validate_experimental",
            "y_validate": "y_validate_experimental",
            "model_name":  "params:svm"
        },
        outputs="metrics",
        name="experiment_svm_buyer_data_only_scaled"
    )
    
    experiment_ann_buyer_data_only_scaled  = node (
        train_ann_experimental_scaled,
        inputs={
            "X_train": "X_train_experimental",
            "y_train": "y_train_experimental",
            "X_validate": "X_validate_experimental",
            "y_validate": "y_validate_experimental",
            "model_name":  "params:ann"
        },
        outputs="metrics",
        name="experiment_ann_buyer_data_only_scaled"
    )
    






    combined_data_splitting_for_experiments = node (
        split_train_test_validate_smote_applied,
         inputs = {
            "data": "combined_data_set_with_risk_levels",
            "target_column": "params:target",
            "columns_to_exclude": "params:columns_to_exclude"
        },
        outputs = ["X_train_experimental_gdp_data_included","X_validate_experimental_gdp_data_included", " X_test_experimental_gdp_data_included", "y_train_experimental_gdp_data_included", "y_validate_experimental_gdp_data_included", "y_test_experimental_gdp_data_included"],
        name="combined_data_splitting_for_experiments"
    )

    experiments_recursive_feature_elimination_node = node(
        train_logistic_regression_experimental_rfe,
        inputs={
            "X_train": "X_train_experimental_gdp_data_included",
            "y_train": "y_train_experimental_gdp_data_included",
            "X_validate": "X_validate_experimental_gdp_data_included",
            "y_validate": "y_validate_experimental_gdp_data_included",
            "model_name": "params:logistic_regression",
            "number_of_features_to_select": "params:number_of_features_to_select",
            "exclude_column": "params:period"
        },
        outputs=[
            "metrics",
            "logistic_regression_model_rfe_experimental"
        ],
        name="experiments_recursive_feature_elimination_node"
    )



    experiment_decision_tree_combinded_dataset_node = node (
        train_decision_tree_experimental,
        inputs={
            "X_train": "X_train_experimental_gdp_data_included",
            "y_train": "y_train_experimental_gdp_data_included",
            "X_validate": "X_validate_experimental_gdp_data_included",
            "y_validate": "y_validate_experimental_gdp_data_included",
            "model_name":  "params:decision_tree",
            "exclude_column": "params:period",
            "important_features": "logistic_regression_model_rfe_experimental"
        },
        outputs="metrics",
        name="experiment_decision_tree_combinded_dataset_node"
    )
    
    experiment_ann_combinded_dataset_node = node (
        train_ann_experimental_feature_selected,
        inputs={
            "X_train": "X_train_experimental_gdp_data_included",
            "y_train": "y_train_experimental_gdp_data_included",
            "X_validate": "X_validate_experimental_gdp_data_included",
            "y_validate": "y_validate_experimental_gdp_data_included",
            "model_name":  "params:ann",
            "important_features": "logistic_regression_model_rfe_experimental"

        },
        outputs="metrics",
        name="experiment_ann_combinded_dataset_node"
    )
    



    experiment_varied_train_test_validate_split_node = node (
        split_train_test_validate_smote_applied_varied_splits,
        inputs = {
            "data": "buyer_payment_practice_dataset_mean_imputed",
            "target_column": "params:target",
            "columns_to_exclude": "params:period",
            "split_num": "params:split_num"
        },
        outputs = ["X_train_experimental_split_variations","X_validate_experimental_split_variations", " X_test_experimental_split_variations", "y_train_experimental_split_variations", "y_validate_experimental_split_variations", "y_test_experimental_split_variations"],
        name="experiment_varied_train_test_validate_split_node"
    )


    
    experiment_decision_tree_buyer_data_only_scaled_varied_splits = node (
        train_decision_tree_experimental_scaled,
        inputs={
            "X_train": "X_train_experimental_split_variations",
            "y_train": "y_train_experimental_split_variations",
            "X_validate": "X_validate_experimental_split_variations",
            "y_validate": "y_validate_experimental_split_variations",
            "model_name":  "params:decision_tree"
        },
        outputs="metrics",
        name="experiment_decision_tree_buyer_data_only_scaled_varied_splits"
    )

    experiment_svm_buyer_data_only_scaled_varied_splits = node (
        train_svm_experimental_scaled,
        inputs={
            "X_train": "X_train_experimental_split_variations",
            "y_train": "y_train_experimental_split_variations",
            "X_validate": "X_validate_experimental_split_variations",
            "y_validate": "y_validate_experimental_split_variations",
            "model_name":  "params:svm"
        },
        outputs="metrics",
        name="experiment_svm_buyer_data_only_scaled_varied_splits"
    )
    
    experiment_ann_buyer_data_only_scaled_varied_splits  = node (
        train_ann_experimental_scaled,
        inputs={
            "X_train": "X_train_experimental_split_variations",
            "y_train": "y_train_experimental_split_variations",
            "X_validate": "X_validate_experimental_split_variations",
            "y_validate": "y_validate_experimental_split_variations",
            "model_name":  "params:ann"
        },
        outputs="metrics",
        name="experiment_ann_buyer_data_only_scaled_varied_splits"
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
            "decision_tree_performance_metric_report",
            "decision_tree_optimal_hyperparameters"
            ],
        name="execute_decision_tree_node"
    )


    execute_logistic_regression_node = node(
        train_logistic_regression,
        inputs={
            "X_train": "X_train_smote",
            "y_train": "y_train_smote",
            "X_validate": "X_validate",
            "y_validate": "y_validate",
            "model_name":  "params:logistic_regression",
            "number_of_iterations": "params:number_of_iterations_randomised_search"
        },
        outputs=[
            "logistic_regression_model",
            "logistic_regression_performance_metric_accuracy",
            "logistic_regression_performance_metric_auc",
            "logistic_regression_performance_metric_report",
            "logistic_regression_optimal_hyperparameters"
                 ],
        name="execute_logistic_regression_node"
    )

    smote_oversample_minority_class_node = node(
         smote_oversample_minority_class,
         inputs={
            "X_train": "X_train",
            "y_train": "y_train",
        },
        outputs= ["X_train_smote", "y_train_smote"]
    )

    execute_svm_node = node(
        func=train_svm,
        inputs={
            "X_train": "X_train_smote",
            "y_train": "y_train_smote",
            "X_validate": "X_validate",
            "y_validate": "y_validate",
            "model_name":  "params:svm",
            "number_of_iterations": "params:number_of_iterations_randomised_search"
        },
        outputs=[
            "svm_model",
            "svm_model_metric_accuracy",
            "svm_model_metric_auc",
            "svm_model_metric_report",
            "svm_model_optimal_hyperparameters"
        ],
        name="execute_svm_node"
    )

    execute_ann_node = node(
        train_ann,
        inputs={
            # "X_train": "X_train_ann",
            # "y_train": "y_train_ann",
            # "X_validate": "X_validate_ann",
            # "y_validate": "y_validate_ann",
            "X_train": "X_train_for_rfe",
            "y_train": "y_train_for_rfe",
            "X_validate": "X_validate_for_rfe",
            "y_validate": "y_validate_for_rfe",
            "model_name":  "params:ann",
            "important_features_df": "logistic_regression_model_rfe_selected_features"
        },
        outputs="ann_model",
        name="execute_ann_node"
    )

    scale_data_for_ann_node = node(
        standard_scale_data,
        inputs={
            "data": "combined_data_set_with_risk_levels",
            "target_column": "params:target",
            "columns_to_exclude": "params:columns_to_exclude",
        },
        outputs="data_standard_scaled_for_ann",
        name="scale_data_for_ann_node"
    )

    split_data_train_test_validate_for_ann_node = node(
        split_train_test_validate,
        inputs = {
            "data": "combined_data_set_pca_applied",
            "target_column": "params:target"
        },
        outputs = ["X_train_ann","X_validate_ann", " X_test_ann", "y_train_ann", "y_validate_ann", "y_test_ann"],
        name="split_data_train_test_validate_for_ann_node"
    )

    split_data_train_test_validate_rfe_node = node(
        split_train_test_validate_rfe,
         inputs = {
            "data": "combined_data_set_with_risk_levels",
            "target_column": "params:target",
            "columns_to_exclude": "params:columns_to_exclude"
        },
        outputs = ["X_train_for_rfe","X_validate_for_rfe", " X_test_for_rfe", "y_train_for_rfe", "y_validate_for_rfe", "y_test_for_rfe"],
        name="split_data_train_test_validate_rfe_node"
    )

    train_test_validate_split_node = node(
        main_split_train_test_validate,
        inputs = {
            "data": "buyer_payment_practice_dataset_mean_imputed",
            "target_column": "params:target",
            "columns_to_exclude": "params:columns_to_exclude"
        },
        outputs= ["X_train_main","X_validate_main", "X_test_main", "y_train_main", "y_validate_main", "y_test_main"]

    )

    recursive_feature_elimination_node = node(
        train_logistic_regression_for_rfe,
        inputs={
            "X_train": "X_train_for_rfe",
            "y_train": "y_train_for_rfe",
            "X_validate": "X_validate_for_rfe",
            "y_validate": "y_validate_for_rfe",
            "model_name": "params:logistic_regression",
            "number_of_features_to_select": "params:number_of_features_to_select"
        },
        outputs= [
            "logistic_regression_model_rfe",
            "logistic_regression_model_rfe_performance_metric_accuracy",
            "logistic_regression_model_rfe_performance_metric_auc",
            "logistic_regression_model_rfe_performance_metric_report",
            "logistic_regression_model_rfe_selected_features"
        ],
        name="recursive_feature_elimination_node"
    )
    
    find_optimal_hyperparameter_ranges_for_decision_tree_node = node(
        train_decision_tree_with_random_search,
        inputs={
            "X_train": "X_train_main",
            "y_train": "y_train_main",
            "X_validate": "X_validate_main",
            "y_validate": "y_validate_main",
            "model_name": "params:decision_tree",
            "number_of_iterations": "params:number_of_iterations_randomised_search"
        },
        outputs={
            "metrics": "metrics_decision_tree_random_search",
            "continuous_params": "decision_tree_continuous_hyperparameters",
            "discrete_params": "decision_tree_discrete_hyperparameters"
        },
        name="find_optimal_hyperparameter_ranges_for_decision_tree_node"
    )
    
    find_optimal_hyperparameter_ranges_for_svm_node = node(
        train_svm_with_random_search,
        inputs={
            "X_train": "X_train_main",
            "y_train": "y_train_main",
            "X_validate": "X_validate_main",
            "y_validate": "y_validate_main",
            "model_name": "params:svm",
            "number_of_iterations": "params:number_of_iterations_randomised_search_decision_tree"
        },
        outputs={
            "metrics": "metrics_svm_random_search",
            "continuous_params": "svm_continuous_hyperparameters",
            "discrete_params": "svm_discrete_hyperparameters"
        },
        name="find_optimal_hyperparameter_ranges_for_svm_node"
    )

    find_optimal_hyperparameter_ranges_for_ann_node = node(
        train_ann_with_random_search,
        inputs={
            "X_train": "X_train_main",
            "y_train": "y_train_main",
            "X_validate": "X_validate_main",
            "y_validate": "y_validate_main",
            "model_name": "params:ann",
            "number_of_iterations": "params:number_of_iterations_randomised_search_ann"

        },
        outputs={
            "metrics": "metrics_ann_random_search",
            "best_hyperparameters": "ann_best_hyperparameters"
        },
        name="find_optimal_hyperparameter_ranges_for_ann_node"
    )

    find_best_hyperparameters_for_decision_tree_node = node (
        train_decision_tree_with_grid_search,
        inputs={
            "X_train": "X_train_main",
            "y_train": "y_train_main",
            "X_validate": "X_validate_main",
            "y_validate": "y_validate_main",
            "model_name": "params:decision_tree",
        },
        outputs={
            "metrics": "metrics_decision_tree_grid_search",
            "best_hyperparameters": "decision_tree_best_hyperparameters",
            "best_model": 'decision_tree_model_grid_search_best_params'
        },
        name="find_best_hyperparameters_for_decision_tree_node"
    )

    analyse_effect_of_reducing_decision_tree_depth_node = node(
        evaluate_decision_tree_depths,
        inputs={
            "X_train": "X_train_main",
            "y_train": "y_train_main",
            "initial_max_depth": "params:decision_tree_initial_max_depth",
            "min_depth": "params:decision_tree_min_depth"

        },
        outputs="decision_tree_depth_reduction_analysis",
        name="analyse_effect_of_reducing_decision_tree_depth_node"
    )

    # find_best_hyperparameters_for_ann_node = node (
    #     train_ann_with_grid_search,
    #     inputs={
    #         "X_train": "X_train_main",
    #         "y_train": "y_train_main",
    #         "X_validate": "X_validate_main",
    #         "y_validate": "y_validate_main",
    #         "model_name": "params:ann",
    #     },
    #     outputs={
    #         "metrics": "metrics",
    #         "best_hyperparameters": "ann_best_hyperparameters_grid_search",
    #         "best_model": 'ann_model_grid_search_best_params'
    #     },
    #     name="find_best_hyperparameters_for_ann_node"
    # )

    ann_with_optimal_hyperparameters_node = node (
        train_ann_with_grid_search,
        inputs={
            "X_train": "X_train_main",
            "y_train": "y_train_main",
            "X_validate": "X_validate_main",
            "y_validate": "y_validate_main",
            "model_name": "params:ann",
        },
        outputs={
            "metrics": "metrics_ann_grid_search",
            "best_model": 'ann_model_grid_search_best_params'
        },
        name="ann_with_optimal_hyperparameters_node"
    )

    run_final_decision_tree_node = node (
        evaluate_and_return_final_decision_tree_model,
        inputs={
            "X_train": "X_train_main",
            "y_train": "y_train_main",
            "X_test": "X_test_main",
            "y_test": "y_test_main",
            "model_name": "params:decision_tree",
        },
        outputs={
            "metrics": "metrics_final_decision_tree_depth_6",
            "model": "decision_tree_final_model",
        },
        name="run_final_decision_tree_node"
    )

    find_best_hyperparameters_for_svm_node = node (
        train_svm_with_grid_search,
        inputs={
            "X_train": "X_train_main",
            "y_train": "y_train_main",
            "X_validate": "X_validate_main",
            "y_validate": "y_validate_main",
            "model_name": "params:svm",
        },
        outputs={
            "metrics": "metrics_svm_grid_search",
            "best_hyperparameters": "svm_best_hyperparameters_grid_search",
            "best_model": 'svm_model_grid_search_best_params'
        },
        name="find_best_hyperparameters_for_svm_node"
    )
    svm_with_optimal_hyperparameters_node = node (
        evaluate_and_return_final_svm_model,
        inputs={
            "X_train": "X_train_main",
            "y_train": "y_train_main",
            "X_test": "X_test_main",
            "y_test": "y_test_main",
            "model_name": "params:svm",
        },
        outputs={
            "metrics": "metrics_final_svm_search",
            "model": "svm_final_model",
        },
        name="svm_with_optimal_hyperparameters_node"
    )
    # decision_tree_with_optimal_hyperparameters_node = node (
    #     evaluate_and_return_final_decision_tree_model,
    #     inputs={
    #         "X_train": "X_train_main",
    #         "y_train": "y_train_main",
    #         "X_test": "X_test_main",
    #         "y_test": "y_test_main",
    #         "model_name": "params:decision_tree",
    #     },
    #     outputs={
    #         "metrics": "metrics",
    #         "model": "decision_tree_final_model",
    #     },
    #     name="decision_tree_with_optimal_hyperparameters_node"
    # )

   
    return Pipeline(
        [ 
           ## Main pipeline Start
           filter_buyer_payment_practises_on_supply_chain_finance_node,
           create_period_column_node,
           extract_payment_periods_node,
           remove_redundant_columns_node,
           anonymise_data_node,
           encode_column_payments_made_in_the_reporting_period,
           align_columns_node,
           determine_and_assign_risk_levels_buyer_data_node,
           mean_imputation_node,
           train_test_validate_split_node,
   
           find_optimal_hyperparameter_ranges_for_decision_tree_node,
           find_optimal_hyperparameter_ranges_for_svm_node,
           find_optimal_hyperparameter_ranges_for_ann_node,
           find_best_hyperparameters_for_decision_tree_node, 
           analyse_effect_of_reducing_decision_tree_depth_node,
           ann_with_optimal_hyperparameters_node,
           find_best_hyperparameters_for_svm_node,
           svm_with_optimal_hyperparameters_node,
           run_final_decision_tree_node,
           ## Main pipeline End

    

           ## Experimental nodes -- Buyer data only, smote applied and standard scaled various train test splits

        #    mean_imputation_for_experimental_data_node,
        #    experiment_varied_train_test_validate_split_node,
        #    experiment_decision_tree_buyer_data_only_scaled_varied_splits,
        #    experiment_svm_buyer_data_only_scaled_varied_splits,
        #    experiment_ann_buyer_data_only_scaled_varied_splits,


        ## Experimantal nodes -- Buyer data only, smote applied and standard scaled
        #    initial_data_splitting_for_experiments,
        #    initial_data_splitting_for_smote_applied,
        #    experiment_decision_tree_buyer_data_only_scaled
        #    experiment_ann_buyer_data_only_scaled,
        #    experiment_svm_buyer_data_only_scaled
           

           ## Experimental nodes -- Buyer Data Only
        #    determine_and_assign_risk_levels_buyer_data_node,
        #    mean_imputation_for_experimental_data_node,
        #    initial_data_splitting_for_experiments,
        #    experiment_decision_tree_buyer_data_only,
        #    experiment_logistic_regression_buyer_data_only
        #    experiment_svm_buyer_data_only
        #    experiment_ann_buyer_data_only

            ### Probably will remove all noderelated to gdp data
        #    monthly_gdp_headers_removed_node,
        #    calculate_monthly_gdp_averages_node,
        #    combine_datasets_node,
        #    convert_payment_practise_column_data_to_floats_node,
        #    peform_mean_imputation_on_combined_dataset_node,
        # #     robust_scale_percentage_invoices_not_paid_on_agreed_terms_column_node,
        #    determine_and_assign_risk_levels_node,



           ## Experimental nodes -- Combined dataset
        #    combined_data_splitting_for_experiments,
        #    experiments_recursive_feature_elimination_node, 
        #    experiment_decision_tree_combinded_dataset_node
        #    experiment_ann_combinded_dataset_node,



        #    apply_principle_component_analysis_node,
        #    split_data_train_test_validate_node,

        #    execute_decision_tree_node,
        #    smote_oversample_minority_class_node,
        #    execute_logistic_regression_node,
        #    execute_svm_node,
        #    scale_data_for_ann_node,
        #    split_data_train_test_validate_for_ann_node,
        #    split_data_train_test_validate_rfe_node,
        #    recursive_feature_elimination_node,
        #    execute_ann_node,
            
        ]
    )
