from kedro.pipeline import Pipeline, node
from .nodes import (clean_payment_practise_data, clean_monthly_gdp_data, 
                    clean_inflation_rates_data, combine_datasets, feature_engineering)

def create_pipeline(**kwargs):
    return Pipeline(
        [ # These are just examples but its better for the nodes to be more granular. Allowing for a more 
            # modular process making it easy to slot in a new  operation.
            # so instead of clean monthly gdp data, you would have 'remove outliers', and apply that node 
            # to all datasets. Think about the point here is reusability and modularity.
            node(
                func=clean_payment_practise_data,
                inputs="raw_payment_data",
                outputs="cleaned_payment_data",
                name="clean_payment_data_node"
            ),
            node(
                func=clean_monthly_gdp_data,
                inputs="raw_gdp_data",
                outputs="cleaned_gdp_data",
                name="clean_gdp_data_node"
            ),
            node(
                func=clean_inflation_rates_data,
                inputs="raw_inflation_data",
                outputs="cleaned_inflation_data",
                name="clean_inflation_data_node"
            ),
            node(
                func=combine_datasets,
                inputs=["cleaned_payment_data", "cleaned_gdp_data", "cleaned_inflation_data"],
                outputs="combined_data",
                name="combine_data_node"
            ),
            node(
                func=feature_engineering,
                inputs="combined_data",
                outputs="feature_engineered_data",
                name="feature_engineering_node"
            ),
        ]
    )
