"""
CPU-bound benchmarks for CodSpeed (Valgrind simulation instrument).

These measure pure computational work with minimal system calls:
Hilbert curve sorting via DuckDB's ST_Hilbert with a pre-loaded spatial
extension.
"""

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_gdf_50k():
    """GeoDataFrame with 50k random Antarctic points."""
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
    """GeoDataFrame with 200k random Antarctic points."""
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


# ---------------------------------------------------------------------------
# Hilbert sorting benchmarks
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
