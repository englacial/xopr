"""
Integration tests for bedmap query functions using real data.

These tests download real bedmap data and verify that query_bedmap()
returns consistent results regardless of the local_cache flag.

Tests are marked with pytest.mark.integration so they can be skipped
in CI or fast test runs (e.g., pytest -m "not integration").
"""

import pytest

from xopr.bedmap.query import fetch_bedmap, query_bedmap

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def bedmap_cache(tmp_path_factory):
    """Download bedmap1 data to a shared temp directory.

    Only fetches bedmap1 (~59 MB, 1 file) to keep downloads small.
    A single bedmap2 file is copied from the main cache if available,
    otherwise downloaded via fetch_bedmap.
    """
    cache_dir = tmp_path_factory.mktemp("bedmap_cache")
    # bedmap1: 1 file, ~59 MB
    fetch_bedmap(version='bedmap1', data_dir=cache_dir)
    # bedmap2: 65 files; we only need one to prove the filter works.
    # fetch_bedmap downloads all files for a version, so instead we
    # grab a single small file directly.
    from pathlib import Path

    import requests

    bm2_url = "https://data.source.coop/englacial/bedmap/data/NIPR_1999_JARE40_GRN_BM2.parquet"
    bm2_path = cache_dir / "NIPR_1999_JARE40_GRN_BM2.parquet"
    if not bm2_path.exists():
        resp = requests.get(bm2_url)
        resp.raise_for_status()
        bm2_path.write_bytes(resp.content)

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

        # Both should contain only BM1 data
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
        df_bm2 = query_bedmap(collections=['bedmap2'], **common)

        assert not df_bm1.empty, "bedmap1 cached query returned no results"
        assert not df_bm2.empty, "bedmap2 cached query returned no results"

        bm1_sources = set(df_bm1['source_file'].unique())
        bm2_sources = set(df_bm2['source_file'].unique())

        # The two collections should have no overlapping source files
        assert bm1_sources.isdisjoint(bm2_sources), (
            f"bedmap1 and bedmap2 sources overlap: "
            f"BM1={bm1_sources}, BM2={bm2_sources}"
        )
        assert all('BM1' in s for s in bm1_sources)
        assert all('BM2' in s for s in bm2_sources)

    def test_cached_bedmap2_excludes_bedmap1(self, bedmap_cache):
        """collections=['bedmap2'] with cache must not return any BM1 data."""
        df = query_bedmap(
            collections=['bedmap2'],
            max_rows=50,
            data_dir=bedmap_cache,
            local_cache=True,
        )

        assert not df.empty, "bedmap2 cached query returned no results"

        sources = set(df['source_file'].unique())
        assert not any('BM1' in s for s in sources), (
            f"bedmap2 query returned BM1 sources: {sources}"
        )
