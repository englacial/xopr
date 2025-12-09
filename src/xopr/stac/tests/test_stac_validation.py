"""Tests for STAC validation using stac-validator for our library functions."""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from stac_validator.validate import StacValidate

from xopr.stac.catalog import (
    create_collection,
    create_item,
    create_items_from_flight_data,
)

from .common import (
    TEST_CITATION,
    TEST_DOI,
    create_mock_flight_data,
    create_mock_metadata,
    get_test_config,
)


class TestSTACValidation:
    """Test STAC validation for objects created by our library functions."""

    def _validate_stac_via_file(self, stac_object):
        """Helper method to validate STAC objects via temporary file.

        Parameters
        ----------
        stac_object : pystac object
            STAC object (catalog, collection, or item) to validate

        Returns
        -------
        bool
            True if valid, False if invalid
        """
        # Write to temporary file and validate via file path
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(stac_object.to_dict(), f, indent=2, default=str)
            temp_file = f.name

        try:
            validator = StacValidate(stac_file=temp_file)
            result = validator.run()
            return result
        finally:
            Path(temp_file).unlink()

    def test_validate_create_collection(self):
        """Test that collections created by create_collection() are valid."""
        from datetime import datetime

        import pystac

        # Create extent as required by create_collection
        spatial_extent = pystac.SpatialExtent(bboxes=[[-69.86, -71.37, -69.84, -71.35]])
        temporal_extent = pystac.TemporalExtent(intervals=[[datetime(2016, 10, 14), datetime(2016, 10, 14)]])
        extent = pystac.Extent(spatial=spatial_extent, temporal=temporal_extent)

        collection = create_collection(
            collection_id="test-collection",
            description="Test collection for validation",
            extent=extent,
            license="various"
        )

        collection_dict = collection.to_dict()
        validator = StacValidate()
        result = validator.validate_dict(collection_dict)
        assert result is True, f"create_collection() produced invalid collection"

    def test_validate_create_item(self):
        """Test that items created by create_item() are valid."""
        from datetime import datetime

        from shapely.geometry import Point, mapping

        # Create geometry and bbox for create_item
        point = Point(-69.85, -71.36)
        geometry = mapping(point)
        bbox = [-69.86, -71.37, -69.84, -71.35]

        item = create_item(
            item_id="test-item",
            geometry=geometry,
            bbox=bbox,
            datetime=datetime(2016, 10, 14, 16, 12, 44),
            properties={"test": "value"}
        )

        result = self._validate_stac_via_file(item)
        assert result is True, f"create_item() produced invalid item"

    @patch('xopr.stac.catalog.extract_item_metadata')
    def test_validate_create_items_from_flight_data(self, mock_extract):
        """Test that items created by create_items_from_flight_data() are valid."""
        # Setup mock with both scientific and SAR metadata
        mock_extract.return_value = create_mock_metadata(
            doi=TEST_DOI,
            citation=TEST_CITATION,
            frequency=190e6,
            bandwidth=50e6
        )
        flight_data = create_mock_flight_data()

        # Create items using our library function
        config = get_test_config()
        items = create_items_from_flight_data(
            flight_data,
            config,
            campaign_name="test_campaign"
        )

        # Validate each item created by our function
        assert len(items) > 0, "create_items_from_flight_data() should create items"

        for i, item in enumerate(items):
            result = self._validate_stac_via_file(item)
            assert result is True, f"create_items_from_flight_data() produced invalid item {i}"


    @patch('xopr.stac.catalog.extract_item_metadata')
    def test_validate_items_with_extensions(self, mock_extract):
        """Test that items with extensions created by create_items_from_flight_data() are valid."""
        # Test with no extensions (minimal case)
        mock_extract.return_value = create_mock_metadata(
            doi=None, citation=None, frequency=None, bandwidth=None
        )
        flight_data = create_mock_flight_data()

        config = get_test_config()
        items = create_items_from_flight_data(flight_data, config)
        for item in items:
            result = self._validate_stac_via_file(item)
            assert result is True, f"create_items_from_flight_data() with no extensions produced invalid item"

        # Test with scientific extension only
        mock_extract.return_value = create_mock_metadata(
            doi=TEST_DOI, citation=None, frequency=None, bandwidth=None
        )

        items = create_items_from_flight_data(flight_data, config)
        for item in items:
            result = self._validate_stac_via_file(item)
            assert result is True, f"create_items_from_flight_data() with scientific extension produced invalid item"

        # Test with SAR extension only
        mock_extract.return_value = create_mock_metadata(
            doi=None, citation=None, frequency=190e6, bandwidth=50e6
        )

        items = create_items_from_flight_data(flight_data, config)
        for item in items:
            result = self._validate_stac_via_file(item)
            assert result is True, f"create_items_from_flight_data() with SAR extension produced invalid item"

        # Test with both extensions
        mock_extract.return_value = create_mock_metadata(
            doi=TEST_DOI, citation=TEST_CITATION, frequency=190e6, bandwidth=50e6
        )

        items = create_items_from_flight_data(flight_data, config)
        for item in items:
            result = self._validate_stac_via_file(item)
            assert result is True, f"create_items_from_flight_data() with both extensions produced invalid item"

    def test_validate_catalog_with_metadata_aggregation(self):
        """Test that catalogs using collect_uniform_metadata produce valid STAC with proper metadata."""
        from datetime import datetime

        import pystac

        from xopr.stac.catalog import create_collection
        from xopr.stac.metadata import collect_uniform_metadata

        # Create mock items with scientific metadata that should be aggregated
        from .common import create_mock_stac_item
        items = []
        for i in range(3):
            item = create_mock_stac_item(
                doi=TEST_DOI,
                citation=TEST_CITATION,
                sar_freq=190e6,
                sar_bandwidth=50e6
            )
            # Fix href methods to return proper strings
            item.id = f"item_{i}"
            item.get_self_href = Mock(return_value=f"https://test.example.com/items/item_{i}.json")
            item.self_href = f"https://test.example.com/items/item_{i}.json"
            items.append(item)

        # Test collect_uniform_metadata to get extensions and extra fields
        extensions, extra_fields = collect_uniform_metadata(items,
            ['sci:doi', 'sci:citation', 'opr:frequency', 'opr:bandwidth'])

        # Create extent for collection
        spatial_extent = pystac.SpatialExtent(bboxes=[[-69.86, -71.37, -69.84, -71.35]])
        temporal_extent = pystac.TemporalExtent(intervals=[[datetime(2016, 10, 14), datetime(2016, 10, 14)]])
        extent = pystac.Extent(spatial=spatial_extent, temporal=temporal_extent)

        # Create collection with metadata collected by our active function
        collection = create_collection(
            collection_id="test-metadata-collection",
            description="Test collection with aggregated metadata",
            extent=extent,
            license="various",
            stac_extensions=extensions
        )

        # Apply the extra fields collected by our function
        for key, value in extra_fields.items():
            collection.extra_fields[key] = value

        # Fix collection href for validation
        collection.set_self_href("https://test.example.com/collections/test-metadata-collection.json")

        # Add items to collection
        collection.add_items(items)

        # Validate the collection
        collection_dict = collection.to_dict()
        validator = StacValidate()
        result = validator.validate_dict(collection_dict)
        assert result is True, f"Collection with collect_uniform_metadata produced invalid STAC"

        # Verify that metadata was properly aggregated
        sci_ext = 'https://stac-extensions.github.io/scientific/v1.0.0/schema.json'
        sar_ext = 'https://stac-extensions.github.io/sar/v1.3.0/schema.json'

        assert sci_ext in collection_dict['stac_extensions'], "Scientific extension should be present"
        # SAR extension should NOT be present anymore (moved to opr namespace)
        assert sar_ext not in collection_dict['stac_extensions'], "SAR extension should not be present (moved to opr namespace)"
        assert collection_dict['sci:doi'] == TEST_DOI, "DOI should be aggregated"
        assert collection_dict['sci:citation'] == TEST_CITATION, "Citation should be aggregated"
        # Check OPR properties instead of SAR properties
        assert collection_dict['opr:frequency'] == 190e6, "Center frequency should be aggregated as opr:frequency"
        assert collection_dict['opr:bandwidth'] == 50e6, "Bandwidth should be aggregated as opr:bandwidth"

    def test_invalid_stac_objects_fail_validation(self):
        """Test that invalid STAC objects are correctly rejected by the validator."""
        validator = StacValidate()

        # Test invalid catalog (missing required fields)
        invalid_catalog = {
            "type": "Catalog",
            "stac_version": "1.1.0",
            # Missing required 'id' and 'description' fields
        }
        result = validator.validate_dict(invalid_catalog)
        assert result is False, "Invalid catalog should fail validation"

        # Test invalid collection (missing required fields)
        invalid_collection = {
            "type": "Collection",
            "stac_version": "1.1.0",
            "id": "test-collection",
            # Missing required 'description', 'license', and 'extent' fields
        }
        result = validator.validate_dict(invalid_collection)
        assert result is False, "Invalid collection should fail validation"

        # Test invalid item (missing required fields)
        invalid_item = {
            "type": "Feature",
            "stac_version": "1.1.0",
            "id": "test-item",
            # Missing required 'geometry', 'bbox', 'properties', and 'datetime'
        }
        result = validator.validate_dict(invalid_item)
        assert result is False, "Invalid item should fail validation"

        # Test item with invalid geometry
        invalid_geometry_item = {
            "type": "Feature",
            "stac_version": "1.1.0",
            "id": "test-item",
            "geometry": "not-a-geometry",  # Invalid geometry
            "bbox": [-69.86, -71.37, -69.84, -71.35],
            "properties": {"datetime": "2016-10-14T16:12:44Z"},
            "links": [],
            "assets": {}
        }
        result = validator.validate_dict(invalid_geometry_item)
        assert result is False, "Item with invalid geometry should fail validation"

        # Test item with invalid bbox (wrong number of coordinates)
        invalid_bbox_item = {
            "type": "Feature",
            "stac_version": "1.1.0",
            "id": "test-item",
            "geometry": {"type": "Point", "coordinates": [-69.85, -71.36]},
            "bbox": [-69.86, -71.37],  # Invalid bbox (should have 4 coordinates)
            "properties": {"datetime": "2016-10-14T16:12:44Z"},
            "links": [],
            "assets": {}
        }
        result = validator.validate_dict(invalid_bbox_item)
        assert result is False, "Item with invalid bbox should fail validation"

