"""
Integration tests for bedmap query functions using real data.

These tests download real bedmap data and verify that query_bedmap()
returns consistent results regardless of the local_cache flag.

Tests are marked with pytest.mark.integration so they can be skipped
in CI or fast test runs (e.g., pytest -m "not integration").
"""

import pytest
import requests
from shapely.geometry import box

from xopr.bedmap.query import fetch_bedmap, query_bedmap

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def bedmap_cache(tmp_path_factory):
    """Download bedmap1 and a single bedmap3 file to a shared temp directory.

    bedmap1: 1 file, ~59 MB
    bedmap3: NASA_2013_ICEBRIDGE_AIR_BM3, ~12 MB (matches CRESIS/UTIG seasons)
    """
    cache_dir = tmp_path_factory.mktemp("bedmap_cache")
    fetch_bedmap(version='bedmap1', data_dir=cache_dir)

    bm3_url = "https://data.source.coop/englacial/bedmap/data/NASA_2013_ICEBRIDGE_AIR_BM3.parquet"
    bm3_path = cache_dir / "NASA_2013_ICEBRIDGE_AIR_BM3.parquet"
    if not bm3_path.exists():
        resp = requests.get(bm3_url)
        resp.raise_for_status()
        bm3_path.write_bytes(resp.content)

    return cache_dir


class TestQueryBedmapCacheParity:
    """Verify that local_cache=True and local_cache=False return equivalent results."""

    def test_collections_bedmap1_parity(self, bedmap_cache):
        """collections=['bedmap1'] should return BM1 data with and without cache."""
        kwargs = dict(
            collections=['bedmap1'],
            max_rows=50,
            data_dir=bedmap_cache,
        )

        df_cloud = query_bedmap(**kwargs, local_cache=False)
        df_cached = query_bedmap(**kwargs, local_cache=True)

        assert not df_cloud.empty, "Cloud query returned no results"
        assert not df_cached.empty, "Cached query returned no results"

        cloud_sources = set(df_cloud['source_file'].unique())
        cached_sources = set(df_cached['source_file'].unique())

        assert all('BM1' in s for s in cloud_sources), (
            f"Cloud query returned non-BM1 sources: {cloud_sources}"
        )
        assert all('BM1' in s for s in cached_sources), (
            f"Cached query returned non-BM1 sources: {cached_sources}"
        )

    def test_collections_not_ignored_by_cache(self, bedmap_cache):
        """Querying different collections with cache should yield different data."""
        common = dict(max_rows=50, data_dir=bedmap_cache, local_cache=True)

        df_bm1 = query_bedmap(collections=['bedmap1'], **common)
        df_bm3 = query_bedmap(collections=['bedmap3'], **common)

        assert not df_bm1.empty, "bedmap1 cached query returned no results"
        assert not df_bm3.empty, "bedmap3 cached query returned no results"

        bm1_sources = set(df_bm1['source_file'].unique())
        bm3_sources = set(df_bm3['source_file'].unique())

        assert bm1_sources.isdisjoint(bm3_sources), (
            f"bedmap1 and bedmap3 sources overlap: "
            f"BM1={bm1_sources}, BM3={bm3_sources}"
        )
        assert all('BM1' in s for s in bm1_sources)
        assert all('BM3' in s for s in bm3_sources)

    def test_cached_bedmap3_excludes_bedmap1(self, bedmap_cache):
        """collections=['bedmap3'] with cache must not return any BM1 data."""
        df = query_bedmap(
            collections=['bedmap3'],
            max_rows=50,
            data_dir=bedmap_cache,
            local_cache=True,
        )

        assert not df.empty, "bedmap3 cached query returned no results"

        sources = set(df['source_file'].unique())
        assert not any('BM1' in s for s in sources), (
            f"bedmap3 query returned BM1 sources: {sources}"
        )

    def test_spatial_query_respects_collections(self, bedmap_cache):
        """Spatial query with collection filter should only return that collection."""
        # West Antarctica bbox — overlaps both bedmap1 and NASA_2013 BM3 data
        geom = box(-135, -85, -100, -75)

        df_bm1 = query_bedmap(
            collections=['bedmap1'],
            geometry=geom,
            max_rows=50,
            data_dir=bedmap_cache,
            local_cache=True,
        )
        df_bm3 = query_bedmap(
            collections=['bedmap3'],
            geometry=geom,
            max_rows=50,
            data_dir=bedmap_cache,
            local_cache=True,
        )

        assert not df_bm1.empty, "bedmap1 spatial+cached query returned no results"
        assert not df_bm3.empty, "bedmap3 spatial+cached query returned no results"

        bm1_sources = set(df_bm1['source_file'].unique())
        bm3_sources = set(df_bm3['source_file'].unique())

        assert all('BM1' in s for s in bm1_sources), (
            f"bedmap1 spatial query returned non-BM1 sources: {bm1_sources}"
        )
        assert all('BM3' in s for s in bm3_sources), (
            f"bedmap3 spatial query returned non-BM3 sources: {bm3_sources}"
        )
        assert bm1_sources.isdisjoint(bm3_sources), (
            f"Spatial queries with different collections should not overlap: "
            f"BM1={bm1_sources}, BM3={bm3_sources}"
        )
