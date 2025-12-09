"""Tests for STAC catalog creation functionality."""

from unittest.mock import patch

import numpy as np

from xopr.stac.catalog import create_items_from_flight_data

from .common import (
    SAR_EXT,
    SCI_EXT,
    TEST_CITATION,
    TEST_DOI,
    create_mock_flight_data,
    create_mock_metadata,
    get_test_config,
)


class TestCreateItemsFromFlightData:
    """Test the create_items_from_flight_data function."""

    @patch('xopr.stac.catalog.extract_item_metadata')
    def test_items_without_scientific_metadata(self, mock_extract):
        """Test that SCI_EXT is not added when doi and citation are None."""
        # Setup
        mock_extract.return_value = create_mock_metadata(doi=None, citation=None)
        flight_data = create_mock_flight_data()

        # Test
        items = create_items_from_flight_data(flight_data, get_test_config())

        # Assertions
        assert len(items) == 2  # Two data files in mock flight data
        item = items[0]

        # Should not have scientific properties
        assert 'sci:doi' not in item.properties
        assert 'sci:citation' not in item.properties

        # Should not have scientific extension
        assert SCI_EXT not in item.stac_extensions

        # Should have OPR radar properties (no SAR extension anymore)
        assert 'opr:frequency' in item.properties
        assert 'opr:bandwidth' in item.properties
        # SAR extension should not be present (moved to opr namespace)
        assert SAR_EXT not in item.stac_extensions

    @patch('xopr.stac.catalog.extract_item_metadata')
    def test_items_with_doi_only(self, mock_extract):
        """Test that SCI_EXT is added when doi exists but citation is None."""
        # Setup
        mock_extract.return_value = create_mock_metadata(doi=TEST_DOI, citation=None)
        flight_data = create_mock_flight_data()

        # Test
        items = create_items_from_flight_data(flight_data, get_test_config())

        # Assertions
        assert len(items) == 2
        item = items[0]

        # Should have doi property but not citation
        assert item.properties['sci:doi'] == TEST_DOI
        assert 'sci:citation' not in item.properties

        # Should have scientific extension
        assert SCI_EXT in item.stac_extensions

    @patch('xopr.stac.catalog.extract_item_metadata')
    def test_items_with_citation_only(self, mock_extract):
        """Test that SCI_EXT is added when citation exists but doi is None."""
        # Setup
        mock_extract.return_value = create_mock_metadata(doi=None, citation=TEST_CITATION)
        flight_data = create_mock_flight_data()

        # Test
        items = create_items_from_flight_data(flight_data, get_test_config())

        # Assertions
        assert len(items) == 2
        item = items[0]

        # Should have citation property but not doi
        assert item.properties['sci:citation'] == TEST_CITATION
        assert 'sci:doi' not in item.properties

        # Should have scientific extension
        assert SCI_EXT in item.stac_extensions

    @patch('xopr.stac.catalog.extract_item_metadata')
    def test_items_with_both_doi_and_citation(self, mock_extract):
        """Test that SCI_EXT is added when both doi and citation exist."""
        # Setup
        mock_extract.return_value = create_mock_metadata(doi=TEST_DOI, citation=TEST_CITATION)
        flight_data = create_mock_flight_data()

        # Test
        items = create_items_from_flight_data(flight_data, get_test_config())

        # Assertions
        assert len(items) == 2
        item = items[0]

        # Should have both properties
        assert item.properties['sci:doi'] == TEST_DOI
        assert item.properties['sci:citation'] == TEST_CITATION

        # Should have scientific extension
        assert SCI_EXT in item.stac_extensions

    @patch('xopr.stac.catalog.extract_item_metadata')
    def test_metadata_extraction_failure(self, mock_extract):
        """Test that items with failed metadata extraction are skipped."""
        # Setup - make extract_item_metadata raise an exception
        mock_extract.side_effect = Exception("Metadata extraction failed")
        flight_data = create_mock_flight_data()

        # Test
        with patch('builtins.print'):  # Suppress warning print
            items = create_items_from_flight_data(flight_data, get_test_config())

        # Assertions
        assert len(items) == 0  # No items should be created

    @patch('xopr.stac.catalog.extract_item_metadata')
    def test_opr_properties_as_python_types(self, mock_extract):
        """Test that OPR radar properties are stored as Python float types."""
        # Setup
        mock_extract.return_value = create_mock_metadata()
        flight_data = create_mock_flight_data()

        # Test
        items = create_items_from_flight_data(flight_data, get_test_config())

        # Assertions
        assert len(items) == 2
        item = items[0]

        # Check types are Python float, not numpy
        assert isinstance(item.properties['opr:frequency'], float)
        assert isinstance(item.properties['opr:bandwidth'], float)
        assert not isinstance(item.properties['opr:frequency'], np.floating)
        assert not isinstance(item.properties['opr:bandwidth'], np.floating)
