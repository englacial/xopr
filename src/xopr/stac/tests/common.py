"""Common utilities and mock creation functions for STAC tests."""

from datetime import datetime
from unittest.mock import Mock

import numpy as np
import pystac
from omegaconf import OmegaConf
from shapely.geometry import LineString, box


def create_mock_dataset(doi=None, ror=None, funder_text=None,
                       f0_values=None, f1_values=None):
    """Create a mock xarray dataset for testing.

    Parameters
    ----------
    doi : str, optional
        DOI value to include in attrs, by default None
    ror : str, optional
        ROR value to include in attrs, by default None
    funder_text : str, optional
        Funder text to include in attrs, by default None
    f0_values : array-like, optional
        Low frequency values, by default [165e6, 165e6, 165e6]
    f1_values : array-like, optional
        High frequency values, by default [215e6, 215e6, 215e6]

    Returns
    -------
    Mock
        Mock dataset object with proper structure
    """
    mock_ds = Mock()

    # Mock attributes
    mock_ds.attrs = {}
    if doi is not None:
        mock_ds.attrs['doi'] = doi
    if ror is not None:
        mock_ds.attrs['ror'] = ror
    if funder_text is not None:
        mock_ds.attrs['funder_text'] = funder_text
    mock_ds.attrs['mimetype'] = 'application/x-hdf5'

    # Mock coordinate data access
    longitude_mock = Mock()
    longitude_mock.values = np.array([-69.86, -69.85, -69.84])

    latitude_mock = Mock()
    latitude_mock.values = np.array([-71.35, -71.36, -71.37])

    slow_time_mean_mock = Mock()
    slow_time_mean_mock.values = np.datetime64('2016-10-14T16:12:44')

    slow_time_mock = Mock()
    slow_time_mock.mean.return_value = slow_time_mean_mock

    def getitem_side_effect(key):
        if key == 'Longitude':
            return longitude_mock
        elif key == 'Latitude':
            return latitude_mock
        elif key == 'slow_time':
            return slow_time_mock
        else:
            raise KeyError(key)

    mock_ds.__getitem__ = Mock(side_effect=getitem_side_effect)

    # Mock param_records
    mock_ds.param_records = {
        'radar': {
            'wfs': {
                'f0': np.array(f0_values or [165e6, 165e6, 165e6]),
                'f1': np.array(f1_values or [215e6, 215e6, 215e6])
            }
        }
    }

    # Mock close method
    mock_ds.close = Mock()

    return mock_ds


def create_mock_metadata(doi=None, citation=None, frequency=190e6, bandwidth=50e6):
    """Create mock metadata as returned by extract_item_metadata.

    Parameters
    ----------
    doi : str, optional
        DOI value, by default None
    citation : str, optional
        Citation value, by default None
    frequency : float, optional
        Center frequency, by default 190e6
    bandwidth : float, optional
        Bandwidth, by default 50e6

    Returns
    -------
    dict
        Mock metadata dictionary
    """
    return {
        'geom': LineString([(-69.86, -71.35), (-69.85, -71.36), (-69.84, -71.37)]),
        'bbox': box(-69.86, -71.37, -69.84, -71.35),
        'date': datetime(2016, 10, 14, 16, 12, 44),
        'frequency': frequency,
        'bandwidth': bandwidth,
        'doi': doi,
        'citation': citation,
        'mimetype': 'application/x-hdf5'
    }


def create_mock_flight_data():
    """Create mock flight data for testing.

    Returns
    -------
    dict
        Mock flight data dictionary
    """
    return {
        'flight_id': '20161014_03',
        'date': '20161014',
        'flight_num': '03',
        'data_files': {
            'CSARP_standard': {
                'Data_20161014_03_001.mat': '/path/to/Data_20161014_03_001.mat',
                'Data_20161014_03_002.mat': '/path/to/Data_20161014_03_002.mat'
            },
            'CSARP_layer': {
                'Data_20161014_03_001.mat': '/path/to/layer/Data_20161014_03_001.mat',
                'Data_20161014_03_002.mat': '/path/to/layer/Data_20161014_03_002.mat'
            }
        }
    }


def create_mock_campaign_data():
    """Create mock campaign data for testing.

    Returns
    -------
    dict
        Mock campaign data dictionary
    """
    return {
        'name': '2016_Antarctica_DC8',
        'year': '2016',
        'location': 'Antarctica',
        'aircraft': 'DC8',
        'path': '/test/path/2016_Antarctica_DC8'
    }


def create_mock_stac_item(doi=None, citation=None, sar_freq=190e6, sar_bandwidth=50e6):
    """Create a mock STAC item for testing.

    Parameters
    ----------
    doi : str, optional
        DOI to include in properties, by default None
    citation : str, optional
        Citation to include in properties, by default None
    sar_freq : float, optional
        OPR frequency (formerly SAR center frequency), by default 190e6
    sar_bandwidth : float, optional
        OPR bandwidth (formerly SAR bandwidth), by default 50e6

    Returns
    -------
    Mock
        Mock STAC item object
    """
    item = Mock(spec=pystac.Item)
    item.properties = {
        'opr:date': '20161014',
        'opr:segment': 3,  # Changed from opr:flight
        'opr:frame': 1  # Changed from opr:segment
    }

    # Add scientific properties if provided
    if doi is not None:
        item.properties['sci:doi'] = doi
    if citation is not None:
        item.properties['sci:citation'] = citation

    # Add OPR properties (formerly SAR properties)
    if sar_freq is not None:
        item.properties['opr:frequency'] = sar_freq
    if sar_bandwidth is not None:
        item.properties['opr:bandwidth'] = sar_bandwidth

    # Mock bbox, datetime, and geometry for extent calculation
    item.bbox = [-69.86, -71.37, -69.84, -71.35]
    item.datetime = datetime(2016, 10, 14, 16, 12, 44)

    # Add geometry as GeoJSON (required by merge_item_geometries)
    from shapely.geometry import LineString, mapping
    line_geom = LineString([(-69.86, -71.35), (-69.85, -71.36), (-69.84, -71.37)])
    item.geometry = mapping(line_geom)

    # Default extensions (SAR extension removed - properties moved to opr namespace)
    item.stac_extensions = [
        'https://stac-extensions.github.io/file/v2.1.0/schema.json'
    ]

    # Add scientific extension if scientific properties exist
    if doi is not None or citation is not None:
        sci_ext = 'https://stac-extensions.github.io/scientific/v1.0.0/schema.json'
        if sci_ext not in item.stac_extensions:
            item.stac_extensions.append(sci_ext)

    return item


# Constants for commonly used values
TEST_DOI = "10.1234/test.doi"
TEST_ROR = "https://ror.org/test"
TEST_FUNDER = "Test Funding Agency"
TEST_CITATION = "Test Citation"

# STAC extension URLs
SCI_EXT = 'https://stac-extensions.github.io/scientific/v1.0.0/schema.json'
SAR_EXT = 'https://stac-extensions.github.io/sar/v1.3.0/schema.json'
FILE_EXT = 'https://stac-extensions.github.io/file/v2.1.0/schema.json'

# Base test configuration for all STAC tests
BASE_CONFIG = OmegaConf.create({
    'assets': {'base_url': "https://test.example.com/"},
    'data': {'primary_product': 'CSARP_standard'},
    'geometry': {'tolerance': 50},
})

def get_test_config(**overrides):
    """Get a test configuration with optional overrides.

    Parameters
    ----------
    **overrides
        Any configuration overrides to apply

    Returns
    -------
    OmegaConf
        Test configuration with overrides applied

    Examples
    --------
    >>> config = get_test_config()  # Use base config
    >>> config = get_test_config(geometry={'tolerance': 100})  # Override tolerance
    >>> config = get_test_config(assets={'base_url': 'https://custom.com/'})  # Override base URL
    """
    config = BASE_CONFIG.copy()
    if overrides:
        config = OmegaConf.merge(config, OmegaConf.create(overrides))
    return config
