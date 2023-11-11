# from kedro.pipeline import Pipeline, node
# from .nodes.model_training import train_decision_tree

# def create_pipeline(**kwargs):
#     execute_decision_tree_node = node(
#         train_decision_tree,
#         inputs={
#             "data": "combined_data_set_pca_applied",
#             "target_column": "params:target"
#         },
#         outputs="decision_tree_model",
#         name="execute_decision_tree_node"
#     )

#     return Pipeline(
#         [ 
#             # execute_decision_tree_node
#         ]
#     )