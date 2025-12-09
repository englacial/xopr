import pytest

import xopr.geometry

test_regions = [
    pytest.param({'regions': 'East'}, {'projection': 'EPSG:3031', 'area': 10498117e6},
        id='east_region'),
    pytest.param({'regions': 'West'}, {'projection': 'EPSG:3031', 'area': 2974149e6},
        id='west_region'),
    pytest.param({'regions': 'Peninsula'}, {'projection': 'EPSG:3031', 'area': 403987e6},
        id='peninsula_region'),
    pytest.param({}, {'projection': 'EPSG:3031', 'area': 13776253e6},
        id='all_regions'),
    pytest.param({'type': 'FL'}, {'projection': 'EPSG:3031', 'area': 1562401e6},
        id='all_ice_shelves'),
    pytest.param({'type': 'GR'}, {'projection': 'EPSG:3031', 'area': 12213973e6},
        id='all_grounded'),
]

@pytest.mark.parametrize("params,expected_results", test_regions)
def test_get_antarctic_regions(params, expected_results):
    """
    Test the get_antarctic_regions function with various parameters.

    This test checks if the function correctly retrieves regions based on the provided parameters.
    """
    merged = xopr.geometry.get_antarctic_regions(**params, merge_regions=True)

    # Project to requested projection
    merged = xopr.geometry.project_geojson(merged, target_crs=expected_results['projection'])

    # Check if the area of the returned regions matches the expected area
    assert pytest.approx(merged.area, rel=1e-1) == expected_results['area']


def test_get_antarctic_regions_fields():
    """
    Test that get_antarctic_regions with merge_regions=False returns expected fields.
    """
    regions = xopr.geometry.get_antarctic_regions(regions='East', merge_regions=False)

    # Check that we get a GeoDataFrame
    assert hasattr(regions, 'columns'), "Expected GeoDataFrame with columns"

    # Check for expected fields
    expected_fields = ['NAME', 'Regions', 'Subregions', 'TYPE', 'Asso_Shelf', 'geometry']
    for field in expected_fields:
        assert field in regions.columns, f"Missing expected field: {field}"

    # Check that we have at least one region
    assert len(regions) > 0, "Expected at least one region"
