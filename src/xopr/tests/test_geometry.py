import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point

import xopr.geometry
from xopr.geometry import grid_points

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
    expected_fields = ['NAME', 'REGION', 'SUBREGION', 'TYPE', 'geometry']
    for field in expected_fields:
        assert field in regions.columns, f"Missing expected field: {field}"

    # Check that we have at least one region
    assert len(regions) > 0, "Expected at least one region"


test_greenland_regions = [
    pytest.param({'subregion': 'NW'}, {'projection': 'EPSG:3413', 'area': 282740856559},
        id='nw_subregion'),
    pytest.param({'subregion': 'SW'}, {'projection': 'EPSG:3413', 'area': 229293313126},
        id='sw_subregion'),
    pytest.param({'subregion': 'CE'}, {'projection': 'EPSG:3413', 'area': 223896254467},
        id='ce_subregion'),
    pytest.param({}, {'projection': 'EPSG:3413', 'area': 1850585886220},
        id='all_subregions'),
    pytest.param({'type': 'TW'}, {'projection': 'EPSG:3413', 'area': 1617010723720},
        id='all_tidewater'),
    pytest.param({'type': 'LT'}, {'projection': 'EPSG:3413', 'area': 245676109357},
        id='all_land_terminating'),
]

@pytest.mark.parametrize("params,expected_results", test_greenland_regions)
def test_get_greenland_regions(params, expected_results):
    """
    Test the get_greenland_regions function with various parameters.

    This test checks if the function correctly retrieves regions based on the provided parameters.
    """
    merged = xopr.geometry.get_greenland_regions(**params, merge_regions=True)

    # Project to requested projection
    merged = xopr.geometry.project_geojson(merged, target_crs=expected_results['projection'])

    # Check if the area of the returned regions matches the expected area
    assert pytest.approx(merged.area, rel=1e-1) == expected_results['area']


def test_get_greenland_regions_fields():
    """
    Test that get_greenland_regions with merge_regions=False returns expected fields.
    """
    regions = xopr.geometry.get_greenland_regions(subregion='NW', merge_regions=False)

    # Check that we get a GeoDataFrame
    assert hasattr(regions, 'columns'), "Expected GeoDataFrame with columns"

    # Check for expected fields
    expected_fields = ['NAME', 'SUBREGION', 'TYPE', 'geometry']
    for field in expected_fields:
        assert field in regions.columns, f"Missing expected field: {field}"

    # Check that we have at least one region
    assert len(regions) > 0, "Expected at least one region"


# ---------------------------------------------------------------------------
# grid_points
# ---------------------------------------------------------------------------


def _random_points_gdf(n=200, box=10000.0, seed=42, mean=-1000.0, std=50.0):
    """Random projected points with a numeric column for testing."""
    rng = np.random.default_rng(seed)
    xs = rng.uniform(0, box, n)
    ys = rng.uniform(0, box, n)
    vals = rng.normal(loc=mean, scale=std, size=n)
    return gpd.GeoDataFrame(
        {'wgs84': vals},
        geometry=gpd.points_from_xy(xs, ys),
        crs='EPSG:3031',
    )


def test_grid_points_returns_dataset_with_expected_vars():
    gdf = _random_points_gdf()
    grid = grid_points(gdf, column='wgs84', spacing=2000)
    assert set(grid.data_vars) == {'wgs84_median', 'wgs84_std', 'wgs84_count'}
    assert dict(grid.sizes) == {'y': 5, 'x': 5}
    assert int(grid['wgs84_count'].sum()) == len(gdf)


def test_grid_points_custom_aggregations():
    gdf = _random_points_gdf()
    grid = grid_points(gdf, column='wgs84', spacing=2000,
                       aggregations=('mean', 'min', 'max'))
    assert set(grid.data_vars) == {'wgs84_mean', 'wgs84_min', 'wgs84_max'}
    valid = ~np.isnan(grid['wgs84_mean'].values)
    assert np.all(grid['wgs84_min'].values[valid] <= grid['wgs84_mean'].values[valid])
    assert np.all(grid['wgs84_mean'].values[valid] <= grid['wgs84_max'].values[valid])


def test_grid_points_empty_cells_are_nan():
    pts = gpd.GeoDataFrame(
        {'wgs84': [-1000.0, -1010.0]},
        geometry=[Point(500, 500), Point(7500, 7500)],
        crs='EPSG:3031',
    )
    grid = grid_points(pts, column='wgs84', spacing=2000)
    assert np.isnan(grid['wgs84_median'].values).sum() > 0
    assert int(grid['wgs84_count'].sum()) == 2


def test_grid_points_rejects_geographic_crs():
    gdf = gpd.GeoDataFrame(
        {'wgs84': [-1000.0]},
        geometry=[Point(0.0, -75.0)],
        crs='EPSG:4326',
    )
    with pytest.raises(ValueError, match='geographic CRS'):
        grid_points(gdf, column='wgs84', spacing=1000)


def test_grid_points_rejects_missing_column():
    gdf = _random_points_gdf()
    with pytest.raises(KeyError, match="'missing'"):
        grid_points(gdf, column='missing', spacing=2000)


def test_grid_points_explicit_bounds():
    gdf = _random_points_gdf()
    grid = grid_points(gdf, column='wgs84', spacing=2000,
                       bounds=(-2000, -2000, 12000, 12000))
    assert dict(grid.sizes) == {'y': 7, 'x': 7}
