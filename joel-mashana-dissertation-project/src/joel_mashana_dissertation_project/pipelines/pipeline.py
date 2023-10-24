from kedro.pipeline import Pipeline, node
from .nodes.data_preprocessing import (filter_data_on_supplychain_finance)

def create_pipeline(**kwargs):
    
    filter_node = node(
                filter_data_on_supplychain_finance,
                inputs="buyer_payment_behaviour",
                outputs="buyer_payment_practices_out",
                name="filter_data_on_supplychain_finance_node"
            )
    
    return Pipeline(
        [ 
           filter_node
        ]
    )
