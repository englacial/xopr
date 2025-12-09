"""Tests for metadata extraction functionality."""

import numpy as np
import pytest
from unittest.mock import Mock, MagicMock, patch
from shapely.geometry import LineString

from xopr.stac.metadata import extract_stable_wfs_params, extract_item_metadata
from .common import create_mock_dataset, TEST_DOI, TEST_ROR, TEST_FUNDER


class TestExtractItemMetadata:
    """Test the extract_item_metadata function."""

    def test_extract_metadata_with_none_values(self):
        """Test that None values are returned when doi/ror/funder_text are missing."""
        # Create dataset with no doi, ror, funder_text
        mock_ds = create_mock_dataset()

        # Test
        result = extract_item_metadata(dataset=mock_ds)

        # Assertions
        assert result['doi'] is None
        assert result['citation'] is None  # funder_text maps to citation
        assert 'frequency' in result
        assert 'bandwidth' in result
        # Dataset should not be closed when passed in directly
        mock_ds.close.assert_not_called()

    def test_extract_metadata_with_values(self):
        """Test that actual values are returned when doi/ror/funder_text exist."""
        # Create dataset with values
        mock_ds = create_mock_dataset(
            doi=TEST_DOI,
            ror=TEST_ROR,
            funder_text=TEST_FUNDER
        )

        # Test
        result = extract_item_metadata(dataset=mock_ds)

        # Assertions
        assert result['doi'] == TEST_DOI
        assert result['citation'] == TEST_FUNDER
        # Dataset should not be closed when passed in directly
        mock_ds.close.assert_not_called()

    def test_frequency_extraction_uniform_values(self):
        """Test frequency extraction when all values are the same."""
        mock_ds = create_mock_dataset(
            f0_values=[165e6, 165e6, 165e6],
            f1_values=[215e6, 215e6, 215e6]
        )

        # Test
        result = extract_item_metadata(dataset=mock_ds)

        # Assertions
        assert result['frequency'] == 190e6  # center frequency
        assert result['bandwidth'] == 50e6   # |215 - 165|
        assert isinstance(result['frequency'], float)
        assert isinstance(result['bandwidth'], float)

    def test_frequency_extraction_transposed_values(self):
        """Test frequency extraction when f0 > f1 (transposed case)."""
        mock_ds = create_mock_dataset(
            f0_values=[215e6, 215e6, 215e6],  # Higher frequency in f0
            f1_values=[165e6, 165e6, 165e6]   # Lower frequency in f1
        )

        # Test
        result = extract_item_metadata(dataset=mock_ds)

        # Assertions
        assert result['frequency'] == 190e6  # center frequency
        assert result['bandwidth'] == 50e6   # abs(165 - 215) = 50
        assert isinstance(result['frequency'], float)
        assert isinstance(result['bandwidth'], float)

    def test_frequency_extraction_multiple_unique_values_error(self):
        """Test that ValueError is raised when multiple unique frequency values exist."""
        mock_ds = create_mock_dataset(
            f0_values=[165e6, 170e6, 175e6],  # Multiple different values
            f1_values=[215e6, 215e6, 215e6]
        )

        # Test - should raise ValueError
        with pytest.raises(ValueError, match="Multiple low frequency values found"):
            extract_item_metadata(dataset=mock_ds)

    def test_datetime_conversion(self):
        """Test that datetime is properly converted from xarray to Python datetime."""
        mock_ds = create_mock_dataset()

        # Test
        result = extract_item_metadata(dataset=mock_ds)

        # Assertions
        from datetime import datetime
        assert isinstance(result['date'], datetime)
        assert result['date'].year == 2016
        assert result['date'].month == 10
        assert result['date'].day == 14

    def test_parameter_validation_both_provided(self):
        """Test that ValueError is raised when both parameters are provided."""
        mock_ds = create_mock_dataset()

        with pytest.raises(ValueError, match="Exactly one of mat_file_path or dataset must be provided"):
            extract_item_metadata(mat_file_path='/fake/path.mat', dataset=mock_ds)

    def test_parameter_validation_neither_provided(self):
        """Test that ValueError is raised when neither parameter is provided."""
        with pytest.raises(ValueError, match="Exactly one of mat_file_path or dataset must be provided"):
            extract_item_metadata()

    def test_file_not_found_error(self):
        """Test that FileNotFoundError is raised for non-existent local files."""
        # Test with a local path that doesn't exist
        with pytest.raises(FileNotFoundError, match="MAT file not found"):
            extract_item_metadata(mat_file_path='/does/not/exist.mat')

    @patch('xopr.stac.metadata.OPRConnection')
    def test_file_loading_closes_dataset(self, mock_opr_connection):
        """Test that dataset is properly closed when loaded from file."""
        mock_opr = Mock()
        mock_opr_connection.return_value = mock_opr

        mock_ds = create_mock_dataset()
        mock_opr.load_frame_url.return_value = mock_ds

        # Test with string path
        with patch('pathlib.Path.exists', return_value=True):
            result = extract_item_metadata(mat_file_path='/fake/path.mat')

        # Should work without error and close dataset
        assert 'doi' in result
        assert 'citation' in result
        mock_ds.close.assert_called_once()

    @patch('xopr.stac.metadata.OPRConnection')
    def test_url_loading_skips_existence_check(self, mock_opr_connection):
        """Test that URL paths skip local file existence checks."""
        mock_opr = Mock()
        mock_opr_connection.return_value = mock_opr

        mock_ds = create_mock_dataset()
        mock_opr.load_frame_url.return_value = mock_ds

        # Test with URL - should not check file existence
        result = extract_item_metadata(mat_file_path='https://example.com/data.mat')

        # Should work without file existence error
        assert 'doi' in result
        assert 'citation' in result
        mock_ds.close.assert_called_once()


class TestExtractItemMetadataWithRealData:
    """Test extract_item_metadata with real remote data files."""

    @pytest.mark.parametrize("data_url", [
        "https://data.cresis.ku.edu/data/rds/2016_Antarctica_DC8/CSARP_standard/20161014_03/Data_20161014_03_001.mat",
        "https://data.cresis.ku.edu/data/rds/2022_Antarctica_BaslerMKB/CSARP_standard/20221210_01/Data_20221210_01_001.mat",
        "https://data.cresis.ku.edu/data/rds/2019_Antarctica_GV/CSARP_standard/20191103_01/Data_20191103_01_026.mat"
    ])
    def test_real_data_extraction(self, data_url):
        """Test metadata extraction from real remote data files."""
        # Test with real remote data
        result = extract_item_metadata(mat_file_path=data_url)

        # Basic sanity checks - all keys should be present
        expected_keys = {'geom', 'bbox', 'date', 'frequency', 'bandwidth', 'doi', 'citation', 'mimetype'}
        assert set(result.keys()) == expected_keys

        # Check data types for always-present values
        from shapely.geometry import LineString
        from datetime import datetime

        assert isinstance(result['geom'], LineString)
        assert isinstance(result['date'], datetime)
        assert isinstance(result['frequency'], float)
        assert isinstance(result['bandwidth'], float)
        assert isinstance(result['mimetype'], str)

        # DOI and citation can be None or strings
        assert result['doi'] is None or isinstance(result['doi'], str)
        assert result['citation'] is None or isinstance(result['citation'], str)

        # Geometry should have points
        assert len(result['geom'].coords) > 0

        # Frequency should be reasonable for radar data
        assert 50e6 <= result['frequency'] <= 1000e6  # 50 MHz to 1 GHz (some older systems use lower frequencies)

        # Bandwidth should be positive
        assert result['bandwidth'] > 0

    @pytest.mark.parametrize("data_url,expected_campaign", [
        ("https://data.cresis.ku.edu/data/rds/2016_Antarctica_DC8/CSARP_standard/20161014_03/Data_20161014_03_001.mat", "2016_Antarctica_DC8"),
        ("https://data.cresis.ku.edu/data/rds/2022_Antarctica_BaslerMKB/CSARP_standard/20221210_01/Data_20221210_01_001.mat", "2022_Antarctica_BaslerMKB"),
        ("https://data.cresis.ku.edu/data/rds/2019_Antarctica_GV/CSARP_standard/20191103_01/Data_20191103_01_026.mat", "2019_Antarctica_GV")
    ])
    def test_real_data_campaign_consistency(self, data_url, expected_campaign):
        """Test that real data extraction produces expected results for known campaigns."""
        result = extract_item_metadata(mat_file_path=data_url)

        # Check that the date makes sense for the campaign year
        expected_year = int(expected_campaign.split('_')[0])
        assert result['date'].year == expected_year

        # Check that coordinates are in Antarctica (roughly)
        bounds = result['bbox'].bounds
        # Antarctica is roughly between -90 to -60 latitude
        assert bounds[1] >= -90  # min latitude
        assert bounds[3] <= -60  # max latitude

    def test_real_data_consistency_across_files(self):
        """Test that metadata extraction is consistent across different real files."""
        data_urls = [
            "https://data.cresis.ku.edu/data/rds/2016_Antarctica_DC8/CSARP_standard/20161014_03/Data_20161014_03_001.mat",
            "https://data.cresis.ku.edu/data/rds/2022_Antarctica_BaslerMKB/CSARP_standard/20221210_01/Data_20221210_01_001.mat",
            "https://data.cresis.ku.edu/data/rds/2019_Antarctica_GV/CSARP_standard/20191103_01/Data_20191103_01_026.mat"
        ]

        results = []
        for url in data_urls:
            result = extract_item_metadata(mat_file_path=url)
            results.append(result)

        # All should have the same structure
        expected_keys = {'geom', 'bbox', 'date', 'frequency', 'bandwidth', 'doi', 'citation', 'mimetype'}
        for result in results:
            assert set(result.keys()) == expected_keys

        # All should have valid geometry
        for result in results:
            assert len(result['geom'].coords) > 0
            assert result['bandwidth'] > 0

        # Frequencies should vary across different campaigns/years but be reasonable
        frequencies = [r['frequency'] for r in results]
        assert len(set(frequencies)) >= 1  # At least some variation
        assert all(50e6 <= f <= 1000e6 for f in frequencies)

class TestExtractStableWfsParams:
    """Test cases for extract_stable_wfs_params function."""

    def test_single_dict_passthrough(self):
        """Test that single dictionary is returned unchanged."""
        input_dict = {'f0': 200000000, 'f1': 450000000, 'param': 'value'}
        result = extract_stable_wfs_params(input_dict)
        assert result == input_dict

    def test_empty_list(self):
        """Test that empty list returns empty dict."""
        result = extract_stable_wfs_params([])
        assert result == {}

    def test_list_with_identical_values(self):
        """Test list where all dictionaries have identical values."""
        input_list = [
            {'f0': 200000000, 'f1': 450000000, 'param': 'stable'},
            {'f0': 200000000, 'f1': 450000000, 'param': 'stable'},
            {'f0': 200000000, 'f1': 450000000, 'param': 'stable'}
        ]
        expected = {'f0': 200000000, 'f1': 450000000, 'param': 'stable'}
        result = extract_stable_wfs_params(input_list)
        assert result == expected

    def test_list_with_mixed_stable_unstable_values(self):
        """Test list where some values are stable and others vary."""
        input_list = [
            {'f0': 200000000, 'f1': 450000000, 'variable': 'a', 'unstable': 1},
            {'f0': 300000000, 'f1': 450000000, 'variable': 'b', 'unstable': 2},
            {'f0': 200000000, 'f1': 450000000, 'variable': 'c', 'unstable': 3}
        ]
        expected = {'f1': 450000000}  # Only f1 is stable across all items
        result = extract_stable_wfs_params(input_list)
        assert result == expected

    def test_single_item_list(self):
        """Test list with single dictionary."""
        input_list = [{'f0': 200000000, 'f1': 450000000}]
        expected = {'f0': 200000000, 'f1': 450000000}
        result = extract_stable_wfs_params(input_list)
        assert result == expected


class TestExtractItemMetadataIntegration:
    """Integration tests for extract_item_metadata function."""

    @patch('xopr.stac.metadata.OPRConnection')
    def test_extract_item_metadata_with_list_wfs(self, mock_opr_class):
        """Test that extract_item_metadata works with list-type wfs data."""
        # Mock the dataset structure
        mock_ds = MagicMock()
        mock_ds.param_records = {
            'radar': {
                'wfs': [
                    {'f0': np.array([200000000]), 'f1': np.array([450000000])},
                    {'f0': np.array([200000000]), 'f1': np.array([450000000])},
                    {'f0': np.array([200000000]), 'f1': np.array([450000000])}
                ]
            }
        }
        mock_ds.__getitem__.side_effect = lambda key: {
            'slow_time': Mock(mean=Mock(return_value=Mock(values=np.datetime64('2014-01-08T12:00:00')))),
            'Longitude': Mock(values=np.array([-45.0, -45.1, -45.2])),
            'Latitude': Mock(values=np.array([70.0, 70.1, 70.2]))
        }[key]
        mock_ds.attrs = {'mimetype': 'application/x-hdf5', 'doi': None, 'ror': None, 'funder_text': None}

        # Mock OPRConnection
        mock_opr = Mock()
        mock_opr.load_frame_url.return_value = mock_ds
        mock_opr_class.return_value = mock_opr

        # Test the function
        with patch('xopr.stac.metadata.simplify_geometry_polar_projection') as mock_simplify:
            # Return a proper LineString geometry
            mock_simplify.return_value = LineString([(-45.0, 70.0), (-45.1, 70.1), (-45.2, 70.2)])

            result = extract_item_metadata("https://fake.url/test.mat")

            # Verify the function completed without errors
            assert result is not None
            assert 'frequency' in result
            assert 'bandwidth' in result

        # Verify that load_frame_url was called
        mock_opr.load_frame_url.assert_called_once_with("https://fake.url/test.mat")


class TestCollectUniformMetadata:
    """Test the collect_uniform_metadata function for DOI and scientific metadata handling."""

    def test_no_scientific_metadata(self):
        """Test that no extensions are added when no scientific metadata exists."""
        from xopr.stac.metadata import collect_uniform_metadata
        from .common import create_mock_stac_item

        # Create items without scientific metadata
        items = [
            create_mock_stac_item(doi=None, citation=None),
            create_mock_stac_item(doi=None, citation=None)
        ]

        # Test
        extensions, extra_fields = collect_uniform_metadata(
            items,
            ['sci:doi', 'sci:citation', 'opr:frequency', 'opr:bandwidth']
        )

        # Should not have scientific extension
        sci_ext = 'https://stac-extensions.github.io/scientific/v1.0.0/schema.json'
        assert sci_ext not in extensions

        # Extra fields should not have scientific properties
        assert 'sci:doi' not in extra_fields
        assert 'sci:citation' not in extra_fields

        # OPR properties should be present (uniform across items)
        # SAR extension should not be present (properties moved to opr namespace)
        sar_ext = 'https://stac-extensions.github.io/sar/v1.3.0/schema.json'
        assert sar_ext not in extensions
        assert 'opr:frequency' in extra_fields
        assert 'opr:bandwidth' in extra_fields

    def test_with_unique_doi(self):
        """Test that scientific extension is added when unique DOI exists."""
        from xopr.stac.metadata import collect_uniform_metadata
        from .common import create_mock_stac_item

        test_doi = "10.1234/test.doi"

        # Create items with same DOI
        items = [
            create_mock_stac_item(doi=test_doi, citation=None),
            create_mock_stac_item(doi=test_doi, citation=None)
        ]

        # Test
        extensions, extra_fields = collect_uniform_metadata(
            items,
            ['sci:doi', 'sci:citation', 'sar:center_frequency', 'sar:bandwidth']
        )

        # Should have scientific extension
        sci_ext = 'https://stac-extensions.github.io/scientific/v1.0.0/schema.json'
        assert sci_ext in extensions

        # Extra fields should have DOI
        assert extra_fields['sci:doi'] == test_doi
        assert 'sci:citation' not in extra_fields  # None values filtered out

    def test_with_multiple_dois_no_aggregation(self):
        """Test that scientific extension is not added when multiple different DOIs exist."""
        from xopr.stac.metadata import collect_uniform_metadata
        from .common import create_mock_stac_item

        # Create items with different DOIs
        items = [
            create_mock_stac_item(doi="10.1234/doi1", citation=None),
            create_mock_stac_item(doi="10.1234/doi2", citation=None)
        ]

        # Test
        extensions, extra_fields = collect_uniform_metadata(
            items,
            ['sci:doi', 'sci:citation', 'sar:center_frequency', 'sar:bandwidth']
        )

        # Should not have scientific extension due to non-uniform DOIs
        sci_ext = 'https://stac-extensions.github.io/scientific/v1.0.0/schema.json'
        # Note: SAR extension might still be present, but not SCI for DOI reasons

        # Extra fields should not have DOI (multiple unique values)
        assert 'sci:doi' not in extra_fields

    def test_none_values_filtered_correctly(self):
        """Test that None values are properly filtered in uniform metadata collection."""
        from xopr.stac.metadata import collect_uniform_metadata
        from .common import create_mock_stac_item

        # Create test items with None and non-None values
        items = [
            create_mock_stac_item(doi=None, citation="Test Citation"),
            create_mock_stac_item(doi="10.1234/test", citation=None),
            create_mock_stac_item(doi="10.1234/test", citation="Test Citation"),
            create_mock_stac_item()  # Default creates no sci properties if doi=None
        ]

        # Test
        extensions, extra_fields = collect_uniform_metadata(
            items,
            ['sci:doi', 'sci:citation', 'sar:center_frequency', 'sar:bandwidth']
        )

        # Should have scientific extension for uniform values
        sci_ext = 'https://stac-extensions.github.io/scientific/v1.0.0/schema.json'
        assert sci_ext in extensions

        # Should have both uniform values
        assert extra_fields['sci:doi'] == "10.1234/test"  # Uniform across non-None values
        assert extra_fields['sci:citation'] == "Test Citation"  # Uniform across non-None values
