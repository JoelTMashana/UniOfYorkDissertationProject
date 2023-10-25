from kedro.pipeline import Pipeline, node
from .nodes.data_preprocessing import (filter_data_on_supplychain_finance, extract_payment_periods, create_period_column,
                                       remove_redundant_columns, anonymise_data
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
    


    return Pipeline(
        [ 
           filter_buyer_payment_practises_node,
           create_period_column_node,
           extract_payment_periods_node,
           remove_redundant_columns_node
        ]
    )
