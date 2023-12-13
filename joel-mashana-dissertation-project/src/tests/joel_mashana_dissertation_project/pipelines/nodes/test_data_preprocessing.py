import pytest
import pandas as pd
from joel_mashana_dissertation_project.pipelines.nodes.data_preprocessing import (filter_data_on_supplychain_finance, create_period_column)

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


def test_create_period_column():
    sample_data = {
        'Start date': ['2021-04-01', '2021-06-15'],
        'End date': ['2021-12-31', '2021-07-20']
    }
    df = pd.DataFrame(sample_data)

    result = create_period_column(df)

    assert 'Period' in result.columns
    assert all(result['Period'] == [
        '2021 APR - 2021 DEC', 
        '2021 JUN - 2021 JUL'
    ]) 
    assert result.columns[0] == 'Period'  