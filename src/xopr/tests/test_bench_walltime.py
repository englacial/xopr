"""
Walltime benchmarks for CodSpeed (wall-clock instrument).

These measure end-to-end query performance including I/O, DuckDB setup,
and parquet reads — operations where system calls are a real part of the
measured work.
"""

import pytest
from shapely.geometry import box

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

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
    conn.query_frames(collections=["1993_Greenland_P3"], max_items=1)
    return conn


@pytest.fixture
def antarctic_bbox():
    """Bounding box over West Antarctica for spatial queries."""
    return box(-120, -85, -60, -70)


# ---------------------------------------------------------------------------
# Bedmap STAC catalog query benchmarks
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
# OPR STAC query benchmarks (general radar catalog)
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
