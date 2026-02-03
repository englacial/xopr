"""
Benchmarks for CodSpeed continuous performance testing.

These benchmarks are run via pytest-codspeed in CI and can also be run
locally with pytest-benchmark. Use ``uv run --extra bench pytest
src/xopr/tests/test_benchmarks.py`` to run them.
"""

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_gdf_50k():
    """GeoDataFrame with 50k random Antarctic points for Hilbert sorting."""
    rng = np.random.default_rng(42)
    n = 50_000
    lons = rng.uniform(-180, 180, n)
    lats = rng.uniform(-90, -60, n)
    geometry = gpd.points_from_xy(lons, lats)
    return gpd.GeoDataFrame(
        {"value": rng.standard_normal(n)},
        geometry=geometry,
        crs="EPSG:4326",
    )


@pytest.fixture
def synthetic_gdf_200k():
    """GeoDataFrame with 200k random Antarctic points for Hilbert sorting."""
    rng = np.random.default_rng(42)
    n = 200_000
    lons = rng.uniform(-180, 180, n)
    lats = rng.uniform(-90, -60, n)
    geometry = gpd.points_from_xy(lons, lats)
    return gpd.GeoDataFrame(
        {"value": rng.standard_normal(n)},
        geometry=geometry,
        crs="EPSG:4326",
    )


@pytest.fixture
def antarctic_bbox():
    """Bounding box over West Antarctica for spatial queries."""
    return box(-120, -85, -60, -70)


# ---------------------------------------------------------------------------
# Hilbert sorting benchmarks
# ---------------------------------------------------------------------------

@pytest.mark.benchmark
def test_hilbert_sorting_50k(synthetic_gdf_50k):
    """Benchmark Hilbert curve sorting on 50k points."""
    from xopr.bedmap.converter import _apply_hilbert_sorting

    _apply_hilbert_sorting(synthetic_gdf_50k, verbose=False)


@pytest.mark.benchmark
def test_hilbert_sorting_200k(synthetic_gdf_200k):
    """Benchmark Hilbert curve sorting on 200k points."""
    from xopr.bedmap.converter import _apply_hilbert_sorting

    _apply_hilbert_sorting(synthetic_gdf_200k, verbose=False)


# ---------------------------------------------------------------------------
# STAC catalog query benchmarks (uses public GCS catalogs)
# ---------------------------------------------------------------------------

@pytest.mark.benchmark
def test_query_bedmap_catalog_spatial(antarctic_bbox):
    """Benchmark spatial query against bedmap STAC catalogs."""
    from xopr.bedmap.query import query_bedmap_catalog

    result = query_bedmap_catalog(
        geometry=antarctic_bbox,
        catalog_path="cloud",
    )
    assert len(result) > 0


@pytest.mark.benchmark
def test_query_bedmap_catalog_collection():
    """Benchmark collection-filtered query against bedmap STAC catalogs."""
    from xopr.bedmap.query import query_bedmap_catalog

    result = query_bedmap_catalog(
        collections=["bedmap3"],
        catalog_path="cloud",
    )
    assert len(result) > 0


@pytest.mark.benchmark
def test_query_bedmap_catalog_combined(antarctic_bbox):
    """Benchmark combined spatial + collection query."""
    from xopr.bedmap.query import query_bedmap_catalog

    result = query_bedmap_catalog(
        collections=["bedmap2"],
        geometry=antarctic_bbox,
        catalog_path="cloud",
    )
    assert len(result) >= 0
