from kedro.pipeline import Pipeline, node
from .nodes import split_data, sum_data

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(split_data, "example_data", ["first_half", "second_half"]),
            node(sum_data, "first_half", "sum_first_half"),
            node(sum_data, "second_half", "sum_second_half"),
        ]
    )