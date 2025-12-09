"""
Simplified metadata extraction utilities using OmegaConf configuration.

This module has been refactored to accept DictConfig objects directly,
removing the need for multiple parameters.
"""

import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from omegaconf import DictConfig
from shapely.geometry import LineString, Point, box

from xopr.opr_access import OPRConnection

from .geometry import simplify_geometry_polar_projection


def discover_flight_lines(campaign_path: Union[str, Path], conf: DictConfig) -> List[Dict[str, Any]]:
    """
    Discover flight lines for a campaign using configuration.

    Parameters
    ----------
    campaign_path : Union[str, Path]
        Path to campaign directory
    conf : DictConfig
        Configuration object with data.primary_product and data.extra_products

    Returns
    -------
    List[Dict[str, Any]]
        List of flight line metadata dictionaries
    """
    campaign_path = Path(campaign_path)

    # Get products from config
    primary_product = conf.data.primary_product
    extra_products = conf.data.get('extra_products', []) or []

    product_path = campaign_path / primary_product

    if not product_path.exists():
        raise FileNotFoundError(f"Data product directory not found: {product_path}")

    flight_pattern = re.compile(r'^(\d{8}_\d+)$')
    flights = []

    for flight_dir in product_path.iterdir():
        if flight_dir.is_dir():
            match = flight_pattern.match(flight_dir.name)
            if match:
                flight_id = match.group(1)
                parts = flight_id.split('_')
                date_part = parts[0]
                flight_num = parts[1]

                # Collect data files for primary product
                data_files = {
                    primary_product: {
                        f.name: str(f) for f in flight_dir.glob("*.mat")
                        if "_img" not in f.name
                    }
                }

                # Include extra data products if they exist
                for extra_product in extra_products:
                    extra_product_path = campaign_path / extra_product / flight_dir.name
                    if extra_product_path.exists():
                        data_files[extra_product] = {
                            f.name: str(f) for f in extra_product_path.glob("*.mat")
                        }

                if data_files:
                    flights.append({
                        'flight_id': flight_id,
                        'date': date_part,
                        'flight_num': flight_num,
                        'data_files': data_files
                    })

    return sorted(flights, key=lambda x: x['flight_id'])


def extract_item_metadata(
    mat_file_path: Union[str, Path] = None,
    dataset=None,
    conf: Optional[DictConfig] = None
) -> Dict[str, Any]:
    """
    Extract metadata from MAT/HDF5 file with optional configuration.

    Parameters
    ----------
    mat_file_path : Union[str, Path], optional
        Path or URL to MAT/HDF5 file
    dataset : xarray.Dataset, optional
        Pre-loaded dataset
    conf : DictConfig, optional
        Configuration for geometry simplification

    Returns
    -------
    Dict[str, Any]
        Extracted metadata including geometry, bbox, date, etc.
    """
    # Validate input
    if (mat_file_path is None) == (dataset is None):
        raise ValueError("Exactly one of mat_file_path or dataset must be provided")

    should_close_dataset = False

    if mat_file_path is not None:
        if isinstance(mat_file_path, str):
            file_path = Path(mat_file_path)
        else:
            file_path = mat_file_path

        # Check existence for local files
        if not str(mat_file_path).startswith(('http://', 'https://')):
            if not file_path.exists():
                raise FileNotFoundError(f"MAT file not found: {file_path}")

        opr = OPRConnection(cache_dir="radar_cache")
        ds = opr.load_frame_url(str(mat_file_path))
        should_close_dataset = True
    else:
        ds = dataset

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        date = pd.to_datetime(ds['slow_time'].mean().values).to_pydatetime()

    # Create geometry
    geom_series = gpd.GeoSeries(map(Point, zip(ds['Longitude'].values, ds['Latitude'].values)))
    line = LineString(geom_series.tolist())

    # Apply simplification based on config
    if conf and conf.get('geometry', {}).get('simplify', True):
        tolerance = conf.geometry.get('tolerance', 100.0)
        line = simplify_geometry_polar_projection(line, simplify_tolerance=tolerance)

    bounds = shapely.bounds(line)
    boundingbox = box(bounds[0], bounds[1], bounds[2], bounds[3])

    # Extract radar parameters
    stable_wfs = extract_stable_wfs_params(find_radar_wfs_params(ds))

    low_freq_array = stable_wfs['f0']
    high_freq_array = stable_wfs['f1']

    unique_low_freq = np.unique(low_freq_array)
    if len(unique_low_freq) != 1:
        raise ValueError(f"Multiple low frequency values found: {unique_low_freq}")
    low_freq = float(unique_low_freq[0])

    unique_high_freq = np.unique(high_freq_array)
    if len(unique_high_freq) != 1:
        raise ValueError(f"Multiple high frequency values found: {unique_high_freq}")
    high_freq = float(unique_high_freq[0])

    bandwidth = float(np.abs(high_freq - low_freq))
    center_freq = float((low_freq + high_freq) / 2)

    # Extract science metadata
    doi = ds.attrs.get('doi', None)
    cite = ds.attrs.get('funder_text', None)
    mime = ds.attrs['mimetype']

    if should_close_dataset:
        ds.close()

    return {
        'geom': line,
        'bbox': boundingbox,
        'date': date,
        'frequency': center_freq,
        'bandwidth': bandwidth,
        'doi': doi,
        'citation': cite,
        'mimetype': mime
    }


def discover_campaigns(data_root: Union[str, Path], conf: Optional[DictConfig] = None) -> List[Dict[str, str]]:
    """
    Discover all campaigns in the data directory.

    Parameters
    ----------
    data_root : Union[str, Path]
        Root directory containing campaign subdirectories
    conf : DictConfig, optional
        Configuration with optional filters

    Returns
    -------
    List[Dict[str, str]]
        List of campaign metadata dictionaries
    """
    campaign_pattern = re.compile(r'^(\d{4})_([^_]+)_([^_]+)$')
    campaigns = []

    data_root = Path(data_root)

    if not data_root.exists():
        raise FileNotFoundError(f"Data root directory not found: {data_root}")

    for item in data_root.iterdir():
        if item.is_dir():
            match = campaign_pattern.match(item.name)
            if match:
                year, location, aircraft = match.groups()

                # Apply filters if config provided
                if conf and 'campaigns' in conf.data:
                    include = conf.data.campaigns.get('include', [])
                    exclude = conf.data.campaigns.get('exclude', [])

                    if include and item.name not in include:
                        continue
                    if exclude and item.name in exclude:
                        continue

                campaigns.append({
                    'name': item.name,
                    'year': year,
                    'location': location,
                    'aircraft': aircraft,
                    'path': str(item)
                })

    return sorted(campaigns, key=lambda x: (x['year'], x['name']))


# Helper functions that were in the original but are still needed

def find_radar_wfs_params(ds):
    """Find radar WFS parameters from dataset."""
    search_paths = [
        lambda: ds.param_records['radar']['wfs'],
        lambda: ds.param_csarp['radar']['wfs'],
        lambda: ds.param_radar['wfs'],
        lambda: ds.radar_params['wfs'],
        lambda: ds.params['radar']['wfs'],
    ]

    for get_params in search_paths:
        try:
            return get_params()
        except (KeyError, AttributeError):
            continue

    available = [attr for attr in dir(ds) if 'param' in attr.lower()]
    raise KeyError(f"Radar WFS parameters not found. Available param attributes: {available}")


def extract_stable_wfs_params(wfs_data: Union[Dict, List[Dict]]) -> Dict:
    """Extract stable parameters from wfs data structure."""
    if isinstance(wfs_data, dict):
        return wfs_data

    if not wfs_data:
        return {}

    common_keys = set(wfs_data[0].keys())
    for item in wfs_data[1:]:
        common_keys &= set(item.keys())

    stable_params = {}
    for key in common_keys:
        values = [item[key] for item in wfs_data]
        if len(set(map(str, values))) == 1:
            stable_params[key] = values[0]

    return stable_params


def collect_uniform_metadata(items: List, property_keys: List[str]) -> tuple[List[str], dict]:
    """
    Collect metadata properties that have uniform values across items.

    Parameters
    ----------
    items : List[pystac.Item]
        List of STAC items to extract metadata from
    property_keys : List[str]
        List of property keys to check

    Returns
    -------
    tuple
        (extensions_needed, extra_fields_dict)
    """
    SCI_EXT = 'https://stac-extensions.github.io/scientific/v1.0.0/schema.json'
    SAR_EXT = 'https://stac-extensions.github.io/sar/v1.3.0/schema.json'

    extensions = []
    extra_fields = {}

    property_mappings = {
        'sci:doi': SCI_EXT,
        'sci:citation': SCI_EXT,
        'sar:center_frequency': SAR_EXT,
        'sar:bandwidth': SAR_EXT
    }

    for key in property_keys:
        values = [
            item.properties.get(key)
            for item in items
            if item.properties.get(key) is not None
        ]

        if values and len(np.unique(values)) == 1:
            ext = property_mappings.get(key)
            if ext and ext not in extensions:
                extensions.append(ext)
            extra_fields[key] = values[0]

    return extensions, extra_fields
