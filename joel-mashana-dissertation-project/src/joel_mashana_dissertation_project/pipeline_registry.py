from __future__ import annotations

from joel_mashana_dissertation_project.pipelines import preprocessing_pipeline

def register_pipelines():
    return {
        "preprocessing": preprocessing_pipeline.create_pipeline(),
        "__default__": preprocessing_pipeline.create_pipeline()
    }