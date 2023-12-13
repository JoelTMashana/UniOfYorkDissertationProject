import pytest
import pandas as pd
from joel_mashana_dissertation_project.pipelines.nodes.data_preprocessing import filter_data_on_supplychain_finance

#AAA

def test_filter_data_on_supplychain_finance():
    
    # Arrange
    sample_data = {
        'Start date': ['29/04/2021', '15/06/2021', '20/07/2020', None],
        'End date': ['31/12/2021', '15/12/2021', None, '20/08/2020'],
        'Supply-chain financing offered': [True, False, True, True]
    }

    df = pd.DataFrame(sample_data)
    df['Start date'] = pd.to_datetime(df['Start date'], errors='coerce')
    df['End date'] = pd.to_datetime(df['End date'], errors='coerce')


    # Act
    result = filter_data_on_supplychain_finance(df, 2021)

    # Assert
    assert not result.empty
    assert all(result['Supply-chain financing offered'])
    assert all(pd.notnull(result['Start date']))
    assert all(pd.notnull(result['End date']))
    assert all(result['Start date'].dt.year == 2021)
    assert result.equals(result.sort_values(by='Start date'))