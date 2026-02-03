"""
Benchmarks for CodSpeed continuous performance testing.

These benchmarks are run via pytest-codspeed in CI and can also be run
locally with pytest-benchmark. Use ``uv run --extra bench pytest
src/xopr/tests/test_benchmarks.py`` to run them.

Benchmarks are designed to minimize system calls (network I/O, extension
loading) so that CodSpeed's Valgrind-based CPU simulation can accurately
measure the computational work.
"""

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
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


@pytest.fixture(scope="session")
def duckdb_spatial_conn():
    """Pre-loaded DuckDB connection with spatial extension installed."""
    conn = duckdb.connect()
    conn.execute("INSTALL spatial; LOAD spatial;")
    yield conn
    conn.close()


@pytest.fixture(scope="session")
def local_catalog_path():
    """Ensure bedmap catalogs are cached locally before benchmarking."""
    from xopr.stac_cache import ensure_bedmap_catalogs, get_bedmap_catalog_path

    ensure_bedmap_catalogs()
    return get_bedmap_catalog_path()


@pytest.fixture(scope="session")
def opr_connection():
    """Pre-warmed OPRConnection with cached parquet footers."""
    from xopr.opr_access import OPRConnection

    conn = OPRConnection()
    # Warm the DuckDB client and parquet footer cache with a small query
    conn.query_frames(collections=["1993_Greenland_P3"], max_items=1)
    return conn


@pytest.fixture
def antarctic_bbox():
    """Bounding box over West Antarctica for spatial queries."""
    return box(-120, -85, -60, -70)


# ---------------------------------------------------------------------------
# Hilbert sorting benchmarks (pure CPU: DuckDB ST_Hilbert + reorder)
# ---------------------------------------------------------------------------

def _hilbert_sort_cpu(gdf, conn):
    """Hilbert sort using a pre-loaded DuckDB spatial connection."""
    coords_df = pd.DataFrame({
        'lon': gdf.geometry.x,
        'lat': gdf.geometry.y,
    })
    coords_df['orig_idx'] = range(len(coords_df))
    minx, miny, maxx, maxy = gdf.total_bounds
    conn.register('coords', coords_df)
    sorted_order = conn.execute(f"""
        SELECT orig_idx,
               ST_Hilbert(lon, lat,
                   {{'min_x': {minx}, 'min_y': {miny},
                    'max_x': {maxx}, 'max_y': {maxy}}}::BOX_2D
               ) as hilbert_idx
        FROM coords
        ORDER BY hilbert_idx
    """).fetchdf()
    conn.unregister('coords')
    return gdf.iloc[sorted_order['orig_idx'].values].reset_index(drop=True)


@pytest.mark.benchmark
def test_hilbert_sorting_50k(synthetic_gdf_50k, duckdb_spatial_conn):
    """Benchmark Hilbert curve sorting on 50k points."""
    _hilbert_sort_cpu(synthetic_gdf_50k, duckdb_spatial_conn)


@pytest.mark.benchmark
def test_hilbert_sorting_200k(synthetic_gdf_200k, duckdb_spatial_conn):
    """Benchmark Hilbert curve sorting on 200k points."""
    _hilbert_sort_cpu(synthetic_gdf_200k, duckdb_spatial_conn)


# ---------------------------------------------------------------------------
# STAC catalog query benchmarks (local cached catalogs, no network I/O)
# ---------------------------------------------------------------------------

@pytest.mark.benchmark
def test_query_bedmap_catalog_spatial(antarctic_bbox, local_catalog_path):
    """Benchmark spatial query against locally cached bedmap STAC catalogs."""
    from xopr.bedmap.query import query_bedmap_catalog

    result = query_bedmap_catalog(
        geometry=antarctic_bbox,
        catalog_path="local",
    )
    assert len(result) > 0


@pytest.mark.benchmark
def test_query_bedmap_catalog_collection(local_catalog_path):
    """Benchmark collection-filtered query against locally cached catalogs."""
    from xopr.bedmap.query import query_bedmap_catalog

    result = query_bedmap_catalog(
        collections=["bedmap3"],
        catalog_path="local",
    )
    assert len(result) > 0


@pytest.mark.benchmark
def test_query_bedmap_catalog_combined(antarctic_bbox, local_catalog_path):
    """Benchmark combined spatial + collection query on local catalogs."""
    from xopr.bedmap.query import query_bedmap_catalog

    result = query_bedmap_catalog(
        collections=["bedmap2"],
        geometry=antarctic_bbox,
        catalog_path="local",
    )
    assert len(result) >= 0


# ---------------------------------------------------------------------------
# OPR STAC query benchmarks (general radar catalog, not bedmap)
# ---------------------------------------------------------------------------

@pytest.mark.benchmark
def test_query_opr_spatial(antarctic_bbox, opr_connection):
    """Benchmark spatial query against OPR STAC catalog."""
    result = opr_connection.query_frames(geometry=antarctic_bbox, max_items=50)
    assert result is not None


@pytest.mark.benchmark
def test_query_opr_collection(opr_connection):
    """Benchmark collection query against OPR STAC catalog."""
    result = opr_connection.query_frames(
        collections=["1993_Greenland_P3"], max_items=50,
    )
    assert result is not None
