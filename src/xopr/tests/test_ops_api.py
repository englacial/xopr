"""
Test OPS API access methods. This file tests that the functions in xopr.ops_api behave
as expected and also verifies that the OPS API has not changed in any breaking ways.
"""

import pytest
import numpy as np
import xarray as xr
import xopr.opr_access as xopr
import xopr.ops_api

test_flights = [
    ('2023_Antarctica_BaslerMKB', '20231229_02', {}),
    ('2022_Antarctica_BaslerMKB', '20230109_01', {}),
    ('2019_Antarctica_GV', '20191105_01', {}),
    ('2018_Antarctica_DC8', '20181107_01', {}),
    ('2017_Antarctica_P3', '20171103_06', {}),
    ('2016_Antarctica_DC8', '20161117_06', {}),
]

invalid_flights = [
    ('2022_Antarctica_BaslerMKB', '20230109_99'),
    ('2016_Antarctica_DC8', '20161014'),
    ('2022', '20230109_01'),
    ('2016_Antarctica_DC8', '20161117_06_b'),
    ('2017_Antarctica_P3', None),
    (None, '20230109_01'),
]


@pytest.mark.parametrize("season,flight_id", [(season, flight_id) for season, flight_id, _ in test_flights])
def test_get_segment_metadata_valid_flights(season, flight_id):
    """
    Test that get_segment_metadata returns expected response structure for valid flights.
    
    This test verifies that valid flight combinations return a successful response
    containing comprehensive segment metadata.
    """
    result = xopr.ops_api.get_segment_metadata(flight_id, season)
    
    # Should return a dict (not None) for valid flights
    assert result is not None, f"Expected dict response for valid flight {season}/{flight_id}, got None"
    assert isinstance(result, dict), f"Expected dict response for {season}/{flight_id}, got {type(result)}"
    
    # Should have successful status
    assert result.get('status') == 1, f"Expected successful status for {season}/{flight_id}, got {result.get('status')}"
    
    # Should contain data with metadata information
    assert 'data' in result, f"Expected 'data' key in response for {season}/{flight_id}"
    
    data = result['data']
    assert data is not None, f"Expected non-null data for {season}/{flight_id}"
    
    if 'metadata' in data:
        # New style response from OPS API
        data = data['metadata']
        expected_keys = ['DOI', 'Funding', 'Organization']
    else:
        expected_keys = ['dois', 'funding_sources', 'rors']

    for key in expected_keys:
        assert key in data, f"Expected key '{key}' in data for {season}/{flight_id}"
        assert isinstance(data[key], list), f"Expected {key} to be a list for {season}/{flight_id}"


@pytest.mark.parametrize("season,flight_id", invalid_flights)
def test_get_segment_metadata_invalid_flights(season, flight_id):
    """
    Test that get_segment_metadata returns status code 0 for invalid flights.
    """
    result = xopr.ops_api.get_segment_metadata(flight_id, season)

    assert result.get('status') == 0, f"Expected status 0 for invalid flight {season}/{flight_id}, got {result.get('status')}"
