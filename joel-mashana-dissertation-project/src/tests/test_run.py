"""
This module contains an example test.

Tests should be placed in ``src/tests``, in modules that mirror your
project's structure, and in files named test_*.py. They are simply functions
named ``test_*`` which test a unit of logic.

To run the tests, run ``kedro test`` from the project root directory.
"""

from pathlib import Path

import pytest

from kedro.config import ConfigLoader
from kedro.framework.context import KedroContext
from kedro.framework.hooks import _create_hook_manager
from kedro.framework.project import settings

import numpy as np
import pandas as pd
from  joel_mashana_dissertation_project import mean_imputation 


@pytest.fixture
def config_loader():
    return ConfigLoader(conf_source=str(Path.cwd() / settings.CONF_SOURCE))


@pytest.fixture
def project_context(config_loader):
    return KedroContext(
        package_name="joel_mashana_dissertation_project",
        project_path=Path.cwd(),
        config_loader=config_loader,
        hook_manager=_create_hook_manager(),
    )


# The tests below are here for the demonstration purpose
# and should be replaced with the ones testing the project
# functionality
class TestProjectContext:
    def test_project_path(self, project_context):
        assert project_context.project_path == Path.cwd()

def test_mean_imputation():
    # Create a DataFrame with some missing values
    df = pd.DataFrame({
        'A': [1, 2, np.nan, 4],
        'B': [5, np.nan, 7, 8],
        'Period': ['2021-01', '2021-02', '2021-03', '2021-04']
    })

    # Run the mean imputation
    imputed_df = mean_imputation(df)

    # Check if there are any missing values left
    assert not imputed_df.drop(columns='Period').isnull().values.any(), "There are still missing values after imputation"