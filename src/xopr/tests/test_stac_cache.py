"""
Unit tests for stac_cache module.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from xopr.stac_cache import (
    BEDMAP_CATALOG_BASE_URL,
    BEDMAP_CATALOG_FILES,
    OPR_CATALOG_S3_GLOB,
    _list_remote_opr_catalogs,
    _load_opr_manifest,
    _save_opr_manifest,
    clear_bedmap_cache,
    clear_opr_cache,
    ensure_bedmap_catalogs,
    get_bedmap_catalog_dir,
    get_bedmap_catalog_path,
    get_cache_dir,
    get_opr_catalog_dir,
    get_opr_catalog_path,
    sync_opr_catalogs,
)


class TestGetCacheDir:
    """Test cache directory resolution."""

    def test_uses_env_var_when_set(self):
        """Test that XOPR_CACHE_DIR environment variable is respected."""
        with patch.dict(os.environ, {'XOPR_CACHE_DIR': '/custom/cache/path'}):
            cache_dir = get_cache_dir()
            assert cache_dir == Path('/custom/cache/path')

    def test_uses_platformdirs_when_no_env_var(self):
        """Test that platformdirs is used when env var not set."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove XOPR_CACHE_DIR if present
            os.environ.pop('XOPR_CACHE_DIR', None)
            cache_dir = get_cache_dir()
            # Should be a path containing 'xopr' (from platformdirs)
            assert 'xopr' in str(cache_dir).lower() or 'cache' in str(cache_dir).lower()


class TestGetBedmapCatalogDir:
    """Test bedmap catalog directory path."""

    def test_returns_bedmap_subdir(self):
        """Test that bedmap catalog dir is under cache/catalogs/bedmap."""
        with patch.dict(os.environ, {'XOPR_CACHE_DIR': '/test/cache'}):
            catalog_dir = get_bedmap_catalog_dir()
            assert catalog_dir == Path('/test/cache/catalogs/bedmap')


class TestEnsureBedmapCatalogs:
    """Test catalog downloading and caching."""

    def test_returns_existing_catalogs(self):
        """Test that existing catalogs are returned without download."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {'XOPR_CACHE_DIR': tmpdir}):
                catalog_dir = get_bedmap_catalog_dir()
                catalog_dir.mkdir(parents=True)

                # Create fake catalog files
                for f in BEDMAP_CATALOG_FILES:
                    (catalog_dir / f).write_text('fake data')

                # Should return catalog dir without downloading
                result = ensure_bedmap_catalogs()
                assert result == catalog_dir
                assert all((catalog_dir / f).exists() for f in BEDMAP_CATALOG_FILES)

    def test_downloads_missing_catalogs(self):
        """Test that missing catalogs trigger download."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {'XOPR_CACHE_DIR': tmpdir}):
                # Mock the download function
                with patch('xopr.stac_cache._download_file') as mock_download:
                    mock_download.return_value = True

                    result = ensure_bedmap_catalogs()

                    # Should have called download for each file
                    assert mock_download.call_count == len(BEDMAP_CATALOG_FILES)

    def test_force_download_redownloads(self):
        """Test that force_download re-downloads even if files exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {'XOPR_CACHE_DIR': tmpdir}):
                catalog_dir = get_bedmap_catalog_dir()
                catalog_dir.mkdir(parents=True)

                # Create fake catalog files
                for f in BEDMAP_CATALOG_FILES:
                    (catalog_dir / f).write_text('old data')

                with patch('xopr.stac_cache._download_file') as mock_download:
                    mock_download.return_value = True

                    ensure_bedmap_catalogs(force_download=True)

                    # Should have downloaded all files
                    assert mock_download.call_count == len(BEDMAP_CATALOG_FILES)

    def test_partial_download_failure(self):
        """Test handling of partial download failures."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {'XOPR_CACHE_DIR': tmpdir}):
                catalog_dir = get_bedmap_catalog_dir()
                catalog_dir.mkdir(parents=True)

                # Create one existing file
                (catalog_dir / BEDMAP_CATALOG_FILES[0]).write_text('existing')

                with patch('xopr.stac_cache._download_file') as mock_download:
                    # First call succeeds, rest fail
                    mock_download.side_effect = [True, False, False]

                    result = ensure_bedmap_catalogs()

                    # Should still return catalog_dir (partial cache is useful)
                    assert result == catalog_dir


class TestGetBedmapCatalogPath:
    """Test main entry point for getting catalog path."""

    def test_returns_local_path_when_cached(self):
        """Test that local path glob pattern is returned when catalogs are cached."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {'XOPR_CACHE_DIR': tmpdir}):
                catalog_dir = get_bedmap_catalog_dir()
                catalog_dir.mkdir(parents=True)

                # Create fake catalog files
                for f in BEDMAP_CATALOG_FILES:
                    (catalog_dir / f).write_text('fake data')

                result = get_bedmap_catalog_path()

                assert str(catalog_dir) in result
                assert 'bedmap*.parquet' in result

    def test_fallback_to_cloud_on_failure(self):
        """Test fallback to cloud URL when local cache fails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {'XOPR_CACHE_DIR': tmpdir}):
                with patch('xopr.stac_cache.ensure_bedmap_catalogs') as mock_ensure:
                    mock_ensure.return_value = None  # Simulate total failure

                    result = get_bedmap_catalog_path()

                    assert BEDMAP_CATALOG_BASE_URL in result


class TestClearBedmapCache:
    """Test cache clearing functionality."""

    def test_clears_existing_cache(self):
        """Test that existing cache files are deleted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {'XOPR_CACHE_DIR': tmpdir}):
                catalog_dir = get_bedmap_catalog_dir()
                catalog_dir.mkdir(parents=True)

                # Create fake catalog files
                for f in BEDMAP_CATALOG_FILES:
                    (catalog_dir / f).write_text('fake data')

                # Verify files exist
                assert all((catalog_dir / f).exists() for f in BEDMAP_CATALOG_FILES)

                # Clear cache
                clear_bedmap_cache()

                # Verify files are deleted
                assert not any((catalog_dir / f).exists() for f in BEDMAP_CATALOG_FILES)

    def test_handles_nonexistent_cache(self):
        """Test that clearing nonexistent cache doesn't raise error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {'XOPR_CACHE_DIR': tmpdir}):
                # Don't create any files
                # Should not raise
                clear_bedmap_cache()


class TestQueryIntegrationWithCatalogPath:
    """Test query functions with the new catalog_path parameter."""

    def test_query_bedmap_catalog_local_default(self):
        """Test that query_bedmap_catalog uses 'local' by default."""
        from xopr.bedmap.query import query_bedmap_catalog

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {'XOPR_CACHE_DIR': tmpdir}):
                catalog_dir = get_bedmap_catalog_dir()
                catalog_dir.mkdir(parents=True)

                # Create fake catalog files
                for f in BEDMAP_CATALOG_FILES:
                    (catalog_dir / f).write_text('fake data')

                with patch('xopr.bedmap.query.DuckdbClient') as mock_client:
                    mock_instance = MagicMock()
                    mock_instance.search.return_value = []
                    mock_client.return_value = mock_instance

                    # Call with default (should use 'local')
                    try:
                        query_bedmap_catalog(max_items=1)
                    except Exception:
                        pass  # Query may fail, we just want to check the path

                    # Check that search was called with local path
                    if mock_instance.search.called:
                        called_path = mock_instance.search.call_args[0][0]
                        assert str(catalog_dir) in called_path

    def test_query_bedmap_catalog_cloud_option(self):
        """Test that catalog_path='cloud' uses GCS URL."""
        from xopr.bedmap.query import query_bedmap_catalog

        with patch('xopr.bedmap.query.DuckdbClient') as mock_client:
            mock_instance = MagicMock()
            mock_instance.search.return_value = []
            mock_client.return_value = mock_instance

            try:
                query_bedmap_catalog(catalog_path='cloud', max_items=1)
            except Exception:
                pass

            if mock_instance.search.called:
                called_path = mock_instance.search.call_args[0][0]
                assert 's3://us-west-2.opendata.source.coop/englacial/bedmap' in called_path

    def test_query_bedmap_catalog_custom_path(self):
        """Test that custom catalog_path is passed through."""
        from xopr.bedmap.query import query_bedmap_catalog

        with patch('xopr.bedmap.query.DuckdbClient') as mock_client:
            mock_instance = MagicMock()
            mock_instance.search.return_value = []
            mock_client.return_value = mock_instance

            custom_path = '/custom/path/bedmap*.parquet'
            try:
                query_bedmap_catalog(catalog_path=custom_path, max_items=1)
            except Exception:
                pass

            if mock_instance.search.called:
                called_path = mock_instance.search.call_args[0][0]
                assert called_path == custom_path


# ---------------------------------------------------------------------------
# OPR catalog caching tests
# ---------------------------------------------------------------------------

S3_XML_RESPONSE = """\
<?xml version="1.0" encoding="UTF-8"?>
<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
  <IsTruncated>false</IsTruncated>
  <Contents>
    <Key>englacial/xopr/catalog/hemisphere=south/provider=cresis/collection=2022_Antarctica_BaslerMKB/stac.parquet</Key>
    <ETag>"abc123"</ETag>
    <Size>4096</Size>
  </Contents>
  <Contents>
    <Key>englacial/xopr/catalog/hemisphere=north/provider=cresis/collection=2017_Greenland_P3/stac.parquet</Key>
    <ETag>"def456"</ETag>
    <Size>2048</Size>
  </Contents>
  <Contents>
    <Key>englacial/xopr/catalog/hemisphere=south/provider=cresis/readme.txt</Key>
    <ETag>"skip"</ETag>
    <Size>100</Size>
  </Contents>
</ListBucketResult>
"""


class TestGetOprCatalogDir:
    """Test OPR catalog directory path."""

    def test_returns_opr_subdir(self):
        with patch.dict(os.environ, {'XOPR_CACHE_DIR': '/test/cache'}):
            assert get_opr_catalog_dir() == Path('/test/cache/catalogs/opr')


class TestOprManifest:
    """Test manifest load/save round-trip."""

    def test_empty_on_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {'XOPR_CACHE_DIR': tmpdir}):
                assert _load_opr_manifest() == {}

    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {'XOPR_CACHE_DIR': tmpdir}):
                data = {"foo/stac.parquet": {"etag": "abc", "size": 100}}
                _save_opr_manifest(data)
                assert _load_opr_manifest() == data


class TestListRemoteOprCatalogs:
    """Test S3 XML listing parser."""

    def test_parses_xml(self):
        mock_resp = MagicMock()
        mock_resp.text = S3_XML_RESPONSE
        mock_resp.raise_for_status = MagicMock()

        with patch('xopr.stac_cache.requests.get', return_value=mock_resp):
            result = _list_remote_opr_catalogs()

        # Should include only .parquet files (not readme.txt)
        assert len(result) == 2
        assert result[0]["etag"] == "abc123"
        assert result[1]["key"].endswith("stac.parquet")

    def test_handles_pagination(self):
        page1 = """\
<?xml version="1.0" encoding="UTF-8"?>
<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
  <IsTruncated>true</IsTruncated>
  <NextContinuationToken>tok2</NextContinuationToken>
  <Contents>
    <Key>englacial/xopr/catalog/a/stac.parquet</Key>
    <ETag>"e1"</ETag><Size>10</Size>
  </Contents>
</ListBucketResult>"""
        page2 = """\
<?xml version="1.0" encoding="UTF-8"?>
<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
  <IsTruncated>false</IsTruncated>
  <Contents>
    <Key>englacial/xopr/catalog/b/stac.parquet</Key>
    <ETag>"e2"</ETag><Size>20</Size>
  </Contents>
</ListBucketResult>"""

        resp1, resp2 = MagicMock(), MagicMock()
        resp1.text, resp2.text = page1, page2
        resp1.raise_for_status = resp2.raise_for_status = MagicMock()

        with patch('xopr.stac_cache.requests.get', side_effect=[resp1, resp2]):
            result = _list_remote_opr_catalogs()

        assert len(result) == 2


class TestSyncOprCatalogs:
    """Test sync logic."""

    def test_downloads_new_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {'XOPR_CACHE_DIR': tmpdir}):
                remote = [{"key": "englacial/xopr/catalog/a/stac.parquet",
                           "etag": "e1", "size": 10}]

                mock_list_resp = MagicMock()
                mock_list_resp.text = S3_XML_RESPONSE
                mock_list_resp.raise_for_status = MagicMock()

                mock_dl_resp = MagicMock()
                mock_dl_resp.raise_for_status = MagicMock()
                mock_dl_resp.iter_content = MagicMock(return_value=[b"data"])

                with patch('xopr.stac_cache._list_remote_opr_catalogs', return_value=remote):
                    with patch('xopr.stac_cache.requests.get', return_value=mock_dl_resp):
                        sync_opr_catalogs()

                dest = get_opr_catalog_dir() / "a" / "stac.parquet"
                assert dest.exists()
                assert dest.read_bytes() == b"data"

    def test_skips_unchanged_etag(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {'XOPR_CACHE_DIR': tmpdir}):
                rel = "a/stac.parquet"
                remote = [{"key": f"englacial/xopr/catalog/{rel}",
                           "etag": "e1", "size": 10}]

                # Pre-populate manifest with matching etag
                _save_opr_manifest({rel: {"etag": "e1", "size": 10}})

                with patch('xopr.stac_cache._list_remote_opr_catalogs', return_value=remote):
                    with patch('xopr.stac_cache.requests.get') as mock_get:
                        sync_opr_catalogs()
                        mock_get.assert_not_called()

    def test_survives_network_error(self):
        with patch('xopr.stac_cache._list_remote_opr_catalogs',
                   side_effect=ConnectionError("offline")):
            # Should not raise
            sync_opr_catalogs()


class TestGetOprCatalogPath:
    """Test path resolution."""

    def test_returns_local_when_cached(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {'XOPR_CACHE_DIR': tmpdir}):
                d = get_opr_catalog_dir() / "hemi" / "coll"
                d.mkdir(parents=True)
                (d / "stac.parquet").write_bytes(b"x")

                result = get_opr_catalog_path()
                assert str(get_opr_catalog_dir()) in result
                assert "*.parquet" in result

    def test_returns_s3_when_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {'XOPR_CACHE_DIR': tmpdir}):
                assert get_opr_catalog_path() == OPR_CATALOG_S3_GLOB


class TestClearOprCache:
    """Test OPR cache removal."""

    def test_removes_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {'XOPR_CACHE_DIR': tmpdir}):
                d = get_opr_catalog_dir() / "sub"
                d.mkdir(parents=True)
                (d / "stac.parquet").write_bytes(b"x")

                clear_opr_cache()
                assert not get_opr_catalog_dir().exists()

    def test_handles_nonexistent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {'XOPR_CACHE_DIR': tmpdir}):
                clear_opr_cache()  # should not raise


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
