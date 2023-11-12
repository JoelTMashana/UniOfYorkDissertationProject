from kedro.pipeline import Pipeline, node
from .nodes.data_preprocessing import train_decision_tree

def create_pipeline(**kwargs):
    execute_decision_tree_baseline_model_node = node(
        train_decision_tree,
        inputs={
            "data": "combined_data_set_pca_applied",
            "target_column": "params:target",
            "model_name":  "params:decision_tree"
        },
        outputs="decision_tree_baseline_model",
        name="execute_decision_tree_node"
    )

    return Pipeline(
        [ 
            execute_decision_tree_baseline_model_node
        ]
    )