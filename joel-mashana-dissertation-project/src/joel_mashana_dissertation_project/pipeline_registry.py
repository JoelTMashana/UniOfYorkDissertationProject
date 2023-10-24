"""Project pipelines."""
from __future__ import annotations

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from joel_mashana_dissertation_project.pipelines.pipeline import create_pipeline as create_data_preprocessing_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    data_preprocessing = create_data_preprocessing_pipeline()
    
    pipelines = {
        "data_preprocessing": data_preprocessing,
        # add other pipelines here as you create them
    }
    
    pipelines["__default__"] = sum(pipelines.values())
    return pipelines
