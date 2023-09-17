# src/context.py

from kedro.pipeline import Pipeline
from pipelines.data_processing.pipeline import create_pipeline as dp_pipeline

class ProjectContext(KedroContext):

    # ... [other methods here]

    def _get_pipelines(self) -> Dict[str, Pipeline]:
        """Returns the project's pipeline."""
        dp = dp_pipeline()
        return {"__default__": dp}
