from kedro.pipeline import Pipeline, node
from .nodes.data_preprocessing import (filter_data_on_supplychain_finance, extract_payment_periods)

def create_pipeline(**kwargs):
    
    filter_buyer_payment_practises_node = node(
                filter_data_on_supplychain_finance,
                inputs="buyer_payment_behaviour_in",
                outputs="buyer_payment_practices_out",
                name="filter_data_on_supplychain_finance_node"
            )
    extract_payment_periods_node = node(
    func=extract_payment_periods,
    inputs="buyer_payment_practices_out",  
    outputs="buyer_payment_practices_payment_periods_out",
    name="extract_payment_periods_node"
)
    
    return Pipeline(
        [ 
           filter_buyer_payment_practises_node,
           extract_payment_periods_node 
        ]
    )
