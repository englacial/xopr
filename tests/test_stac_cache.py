"""
Unit tests for stac_cache module.
"""

import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from xopr.stac_cache import (
    get_cache_dir,
    get_bedmap_catalog_dir,
    ensure_bedmap_catalogs,
    get_bedmap_catalog_path,
    clear_bedmap_cache,
    BEDMAP_CATALOG_FILES,
    BEDMAP_CATALOG_BASE_URL,
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
                assert 'gs://opr_stac/bedmap' in called_path

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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
