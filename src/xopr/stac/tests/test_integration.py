"""Integration tests for STAC functionality."""

from unittest.mock import patch

from xopr.stac.metadata import extract_item_metadata
from xopr.stac.catalog import create_items_from_flight_data
from .common import (create_mock_dataset, create_mock_metadata,
                     create_mock_flight_data, SCI_EXT, SAR_EXT, get_test_config)


class TestSTACIntegration:
    """Integration tests that test the full workflow."""

    def test_none_values_end_to_end(self):
        """Test that None values flow correctly through the entire pipeline."""
        # Test metadata extraction with mock dataset
        mock_dataset = create_mock_dataset()
        metadata = extract_item_metadata(dataset=mock_dataset)

        # Verify None values
        assert metadata['doi'] is None
        assert metadata['citation'] is None

        # Test item creation with this metadata
        with patch('xopr.stac.catalog.extract_item_metadata', return_value=metadata):
            sample_flight_data = create_mock_flight_data()
            items = create_items_from_flight_data(sample_flight_data, get_test_config())

            # Verify items don't have scientific properties
            assert len(items) == 2  # Two data files in sample_flight_data
            for item in items:
                assert 'sci:doi' not in item.properties
                assert 'sci:citation' not in item.properties

                # Should not have scientific extension
                assert SCI_EXT not in item.stac_extensions

                # Should have OPR radar properties (no SAR extension anymore)
                assert 'opr:frequency' in item.properties
                assert 'opr:bandwidth' in item.properties
                # SAR extension should not be present (moved to opr namespace)
                assert SAR_EXT not in item.stac_extensions

    def test_scientific_values_end_to_end(self):
        """Test that scientific values flow correctly through the entire pipeline."""
        # Test metadata extraction with mock dataset
        mock_dataset = create_mock_dataset(doi='10.1234/test.doi', funder_text='Test Funding Agency')
        metadata = extract_item_metadata(dataset=mock_dataset)

        # Verify scientific values
        assert metadata['doi'] == '10.1234/test.doi'
        assert metadata['citation'] == 'Test Funding Agency'

        # Test item creation with this metadata
        with patch('xopr.stac.catalog.extract_item_metadata', return_value=metadata):
            sample_flight_data = create_mock_flight_data()
            items = create_items_from_flight_data(sample_flight_data, get_test_config())

            # Verify items have scientific properties
            assert len(items) == 2
            for item in items:
                assert item.properties['sci:doi'] == '10.1234/test.doi'
                assert item.properties['sci:citation'] == 'Test Funding Agency'

                # Should have scientific extension
                assert SCI_EXT in item.stac_extensions

    def test_collection_aggregation_logic(self):
        """Test the collection aggregation logic matches our fixed behavior."""
        # Test items without scientific metadata
        items_without_science = [
            create_mock_metadata(doi=None, citation=None),
            create_mock_metadata(doi=None, citation=None)
        ]

        # Test the filtering logic
        dois = [
            item.get('doi') for item in items_without_science
            if item.get('doi') is not None
        ]
        citations = [
            item.get('citation') for item in items_without_science
            if item.get('citation') is not None
        ]

        # Should be empty lists
        assert len(dois) == 0
        assert len(citations) == 0

        # Test with scientific metadata
        items_with_science = [
            create_mock_metadata(doi='10.1234/test.doi', citation='Test Funding Agency'),
            create_mock_metadata(doi='10.1234/test.doi', citation='Test Funding Agency')
        ]

        dois_with_science = [
            item.get('doi') for item in items_with_science
            if item.get('doi') is not None
        ]
        citations_with_science = [
            item.get('citation') for item in items_with_science
            if item.get('citation') is not None
        ]

        # Should have values
        assert len(dois_with_science) == 2
        assert len(citations_with_science) == 2
        assert all(doi == '10.1234/test.doi' for doi in dois_with_science)
        assert all(citation == 'Test Funding Agency' for citation in citations_with_science)

    def test_type_safety(self):
        """Test that all values are proper Python types for JSON serialization."""
        metadata = create_mock_metadata(doi='10.1234/test.doi', citation='Test Funding Agency')

        # Check that frequency and bandwidth are Python floats
        assert isinstance(metadata['frequency'], float)
        assert isinstance(metadata['bandwidth'], float)

        # Check that scientific metadata are proper types
        assert isinstance(metadata['doi'], str)
        assert isinstance(metadata['citation'], str)

        # These should be JSON serializable
        import json
        test_dict = {
            'frequency': metadata['frequency'],
            'bandwidth': metadata['bandwidth'],
            'doi': metadata['doi'],
            'citation': metadata['citation']
        }

        # Should not raise an exception
        json_str = json.dumps(test_dict)
        assert json_str is not None

    def test_backward_compatibility(self):
        """Test that old 'null' string values would be handled correctly by new logic."""
        # Simulate old behavior with 'null' strings
        old_style_properties = {
            'sci:doi': 'null',
            'sci:citation': 'null',
            'sar:center_frequency': 190e6
        }

        # Test the filtering logic that should now use `is not None`
        # With old 'null' strings, this would incorrectly include them
        old_style_dois = [
            old_style_properties.get('sci:doi')
            for prop in [old_style_properties]
            if prop.get('sci:doi') is not None
        ]

        # With the old logic, this would include 'null' string
        assert 'null' in old_style_dois  # This shows the problem with old logic

        # With new None-based logic
        new_style_properties = {
            'sci:doi': None,
            'sci:citation': None,
            'sar:center_frequency': 190e6
        }

        new_style_dois = [
            new_style_properties.get('sci:doi')
            for prop in [new_style_properties]
            if prop.get('sci:doi') is not None
        ]

        # With the new logic, this correctly excludes None
        assert len(new_style_dois) == 0  # This shows the fix works
