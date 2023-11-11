"""Project pipelines."""
from __future__ import annotations

from kedro.pipeline import Pipeline
from joel_mashana_dissertation_project.pipelines import preprocessing_pipeline, training_pipeline, evaluation_pipeline


def register_pipelines():
    return {
        "preprocessing": preprocessing_pipeline.create_pipeline(),
        "training": training_pipeline.create_pipeline(),
        "evaluation": evaluation_pipeline.create_pipeline(),
        "__default__": preprocessing_pipeline.create_pipeline() + 
                       training_pipeline.create_pipeline() + 
                       evaluation_pipeline.create_pipeline()
    }
