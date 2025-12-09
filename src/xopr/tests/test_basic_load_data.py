"""
Spot test basic loading of OPR data
"""

import numpy as np
import pytest
import xarray as xr

import xopr
from xopr.util import equivalent

test_flights = [
    ('2022_Antarctica_BaslerMKB', '20230109_01'),
    ('2016_Antarctica_DC8', '20161117_06'),
]

@pytest.mark.parametrize("collection,segment_path", test_flights)
def test_merge_flights_from_frames(collection, segment_path):
    """
    Test that merge_flights_from_frames correctly merges frames and maintains slow_time monotonicity.

    This test loads two frames from the 2022_Antarctica_BaslerMKB collection
    and verifies that the merge operation works correctly.
    """

    opr = xopr.OPRConnection()

    # Query and load frames
    stac_items = opr.query_frames(collections=[collection], segment_paths=[segment_path])

    # Load only the first two frames for testing
    frames = opr.load_frames(stac_items[:2])

    # Verify we have at least 2 frames
    assert len(frames) >= 2, f"Expected at least 2 frames, got {len(frames)}"

    # Test merge_flights_from_frames
    merged_flight = xopr.merge_frames(frames)

    assert isinstance(merged_flight, xr.Dataset), "Merged flight should be an xarray Dataset"

    # Verify the merged result is an xarray Dataset
    assert isinstance(merged_flight, xr.Dataset), "Merged flight should be an xarray Dataset"

    # Check that slow_time dimension exists
    assert 'slow_time' in merged_flight.dims, "Merged flight should have slow_time dimension"

    # Verify slow_time is monotonic
    slow_time_values = merged_flight.slow_time.values
    assert len(slow_time_values) > 1, "Should have multiple slow_time values"

    # Check that slow_time is monotonically increasing
    time_diffs = np.diff(slow_time_values)
    assert np.all(time_diffs > 0), "slow_time values should be monotonically increasing"

    # Verify that the merged flight has the expected attributes
    assert 'segment_path' in merged_flight.attrs, "Merged flight should have segment_path attribute"
    assert merged_flight.attrs['segment_path'] == segment_path, f"Segment path should be {segment_path}"


@pytest.mark.parametrize("collection,segment_path", test_flights)
def test_param_records_equivalence_qlook_vs_standard(collection, segment_path):
    """
    Test that specific parameters in CSARP_qlook and CSARP_standard data products are equivalent.

    This test loads both CSARP_qlook and CSARP_standard data for a single frame
    and verifies that key param_records attributes are equivalent.
    """
    opr = xopr.OPRConnection()

    # Query frames for the flight
    stac_items = opr.query_frames(collections=[collection], segment_paths=[segment_path])

    # Load the first frame with both data products
    first_item = stac_items.iloc[0]


    # Load CSARP_standard data
    standard_frame = opr.load_frame(first_item, data_product="CSARP_standard")

    # Load CSARP_qlook data
    qlook_frame = opr.load_frame(first_item, data_product="CSARP_qlook")

    # Verify both frames have param_records attributes
    assert 'param_records' in standard_frame.attrs, "CSARP_standard should have param_records attribute"
    assert 'param_records' in qlook_frame.attrs, "CSARP_qlook should have param_records attribute"

    standard_params = standard_frame.attrs['param_records']
    qlook_params = qlook_frame.attrs['param_records']

    # Key paths to check
    key_paths = [
        ['radar', 'prf'],
        ['radar', 'wfs', 'f0'],
        ['radar', 'wfs', 'f1'],
        ['records', 'file', 'base_dir']
    ]

    for path in key_paths:
        # Get values from both param_records
        standard_value = standard_params
        qlook_value = qlook_params

        for key in path:
            if isinstance(key, int):
                standard_value = standard_value[key]
                qlook_value = qlook_value[key]
            else:
                standard_value = standard_value.get(key, None)
                qlook_value = qlook_value.get(key, None)

        # Check equivalence
        assert equivalent(standard_value, qlook_value), f"Values at path {path} are not equivalent: {standard_value} != {qlook_value}"
