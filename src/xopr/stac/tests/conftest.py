"""Pytest configuration and fixtures for STAC tests."""

import pytest

from .common import create_mock_dataset, create_mock_metadata


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary output directory for testing."""
    output_dir = tmp_path / "test_output"
    output_dir.mkdir()
    return output_dir


# Legacy fixtures that simply wrap the common functions for backward compatibility
@pytest.fixture
def mock_dataset():
    """Create a mock xarray dataset for testing."""
    return create_mock_dataset()


@pytest.fixture
def mock_dataset_with_science_metadata():
    """Create a mock xarray dataset with scientific metadata."""
    return create_mock_dataset(
        doi='10.1234/test.doi',
        ror='https://ror.org/test',
        funder_text='Test Funding Agency'
    )


@pytest.fixture
def sample_metadata():
    """Create sample metadata as returned by extract_item_metadata."""
    return create_mock_metadata()


@pytest.fixture
def sample_metadata_with_science():
    """Create sample metadata with scientific information."""
    return create_mock_metadata(
        doi='10.1234/test.doi',
        citation='Test Funding Agency'
    )
