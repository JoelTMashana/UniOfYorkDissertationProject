import pytest
import pandas as pd
from joel_mashana_dissertation_project.pipelines.nodes.data_preprocessing import (filter_data_on_supplychain_finance, create_period_column, extract_payment_periods,
                                                                                  remove_redundant_columns
                                                                                  )

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


def test_extract_payment_periods():
    sample_data = {
        'Start date': ['2016-06-01', '2017-04-01', '2018-11-20', '2019-05-15'],
        'End date': ['2016-12-31', '2018-03-30', '2019-10-25', '2020-04-20']
    }
    df = pd.DataFrame(sample_data)

 
    result = extract_payment_periods(df)

    assert 2016 not in result.keys()
    for year in range(2017, 2021):
        assert year in result.keys()
    
    expected_periods = {
        2017: ['(2017 APR, 2018 MAR)'],
        2018: ['(2017 APR, 2018 MAR)', '(2018 NOV, 2019 OCT)'],
        2019: ['(2018 NOV, 2019 OCT)', '(2019 MAY, 2020 APR)'],
        2020: ['(2019 MAY, 2020 APR)']
    }

    assert result == expected_periods


def test_remove_redundant_columns():
    sample_data = {
        'Start date': ['2021-01-01', '2021-02-01'],
        'End date': ['2021-01-31', '2021-02-28'],
        'Filing date': ['2021-02-15', '2021-03-15'],
        'URL': ['https://check-payment-practices.service.gov.uk/report/2', 'https://check-payment-practices.service.gov.uk/report/3'],
        'Shortest (or only) standard payment period': [30, 60]
    }
    df = pd.DataFrame(sample_data)

 
    result = remove_redundant_columns(df)

    assert 'Start date' not in result.columns
    assert 'End date' not in result.columns
    assert 'Filing date' not in result.columns
    assert 'URL' not in result.columns
    assert 'Shortest (or only) standard payment period' in result.columns