from kedro.pipeline import Pipeline, node
from .nodes.data_preprocessing import (filter_data_on_supplychain_finance, extract_payment_periods, create_period_column,
                                       remove_redundant_columns, anonymise_data, encode_column, align_columns
                                       )
def create_pipeline(**kwargs):
    
    filter_buyer_payment_practises_node = node(
                filter_data_on_supplychain_finance,
                inputs="buyer_payment_behaviour_in",
                outputs="buyer_payment_practices_out",
                name="filter_data_on_supplychain_finance_node"
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


    return Pipeline(
        [ 
           filter_buyer_payment_practises_node,
           create_period_column_node,
           extract_payment_periods_node,
           remove_redundant_columns_node,
           anonymise_data_node,
           encode_column_payments_made_in_the_reporting_period,
           align_columns_node
        ]
    )
