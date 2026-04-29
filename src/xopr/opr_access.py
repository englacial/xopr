"""
Core data access interface for Open Polar Radar (OPR) datasets.

This module provides the primary interface for discovering, querying, and loading
polar ice-penetrating radar data from the Open Polar Radar archive. It implements
a STAC (SpatioTemporal Asset Catalog) based workflow for efficient data discovery
and retrieval.

Primary Class
-------------
OPRConnection : Main interface for accessing OPR data
    Provides methods for querying STAC catalogs, loading radar frames,
    and retrieving layer picks from both file-based and database sources.

Typical Workflow
----------------
1. Create an OPRConnection object, optionally with local caching
2. Query the STAC catalog to find radar data matching your criteria
3. Load radar frames as xarray Datasets
4. Optionally load associated layer picks
5. Process and visualize the data

See this notebook for an end-to-end example:
- https://docs.englacial.org/xopr/demo-notebook/

Examples
--------
>>> import xopr
>>> # Connect to OPR with local caching
>>> opr = xopr.OPRConnection(cache_dir='radar_cache')
>>>
>>> # Query for data in a specific collection and segment
>>> stac_items = opr.query_frames(
...     collections=['2022_Antarctica_BaslerMKB'],
...     segment_paths=['20230109_01']
... )
>>>
>>> # Load the radar frames
>>> frames = opr.load_frames(stac_items)
>>>
>>> # Get layer picks (surface/bed)
>>> layers = opr.get_layers(frames[0])

See Also
--------
xopr.geometry : Geographic utilities for region selection and projection
xopr.radar_util : Processing functions for radar data and layers
ops_api : Interface to the OPS database API
"""

import re
import warnings
from typing import Union

import antimeridian
import fsspec
import geopandas as gpd
import h5py
import hdf5storage
import numpy as np
import pandas as pd
import scipy.io
import shapely
import xarray as xr
from rustac import DuckdbClient

from . import opr_tools, ops_api
from .cf_units import apply_cf_compliant_attrs
from .matlab_attribute_utils import (
    decode_hdf5_matlab_variable,
    extract_legacy_mat_attributes,
)
from .radar_util import layer_twtt_to_range
from .stac_cache import get_opr_catalog_path, sync_opr_catalogs


class OPRConnection:
    def __init__(self,
                 collection_url: str = "https://data.cresis.ku.edu/data/",
                 cache_dir: str = None,
                 stac_parquet_href: str = None,
                 sync_catalogs: bool = True):
        """
        Initialize connection to OPR data archive.

        Parameters
        ----------
        collection_url : str, optional
            Base URL for OPR data collection.
        cache_dir : str, optional
            Directory to cache downloaded data for faster repeated access.
        stac_parquet_href : str, optional
            Path or URL pattern to STAC catalog parquet files.  When *None*
            (default), the local cache is used if available, falling back to
            the S3 catalog.
        sync_catalogs : bool, optional
            If True (default), sync OPR STAC catalogs to a local cache
            before resolving the catalog path.  Uses ETag-based change
            detection so repeated calls are cheap (single HTTP request).
        """
        self.collection_url = collection_url
        self.cache_dir = cache_dir
        self._user_set_href = stac_parquet_href is not None
        if sync_catalogs and not self._user_set_href:
            sync_opr_catalogs()
        self.stac_parquet_href = stac_parquet_href or get_opr_catalog_path()

        self.fsspec_cache_kwargs = {}
        self.fsspec_url_prefix = ''
        if cache_dir:
            self.fsspec_cache_kwargs['cache_storage'] = cache_dir
            self.fsspec_cache_kwargs['check_files'] = True
            self.fsspec_url_prefix = 'filecache::'

        self.db_layers_metadata = None # Cache for OPS database layers metadata

        # Lazy-initialized DuckDB client for STAC queries — reusing a single
        # session caches remote file metadata (parquet footers) across calls
        self._duckdb_client = None

    @property
    def duckdb_client(self):
        """Lazy-initialized DuckDB client, recreated after pickling."""
        if self._duckdb_client is None:
            self._duckdb_client = DuckdbClient()
        return self._duckdb_client

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_duckdb_client'] = None  # DuckdbClient can't be pickled
        return state

    def _open_file(self, url):
        """Helper method to open files with appropriate caching.

        Local filesystem paths are returned as-is (no fsspec wrapping).
        Remote URLs (http, https, s3, gs) use filecache or simplecache.
        """
        if not url.startswith(('http://', 'https://', 's3://', 'gs://')):
            return url
        if self.fsspec_url_prefix:
            return fsspec.open_local(f"{self.fsspec_url_prefix}{url}", filecache=self.fsspec_cache_kwargs)
        else:
            return fsspec.open_local(f"simplecache::{url}", simplecache=self.fsspec_cache_kwargs)

    def _extract_segment_info(self, segment):
        """Extract collection, segment_path, and frame from Dataset or dict."""
        if isinstance(segment, xr.Dataset):
            return (segment.attrs.get('collection'),
                    segment.attrs.get('segment_path'),
                    segment.attrs.get('frame'))
        else:
            return (segment['collection'],
                    f"{segment['properties'].get('opr:date')}_{segment['properties'].get('opr:segment'):02d}",
                    segment['properties'].get('opr:frame'))

    def query_frames(self, collections: list[str] = None, segment_paths: list[str] = None,
                     geometry = None, date_range: tuple = None, properties: dict = {},
                     max_items: int = None, exclude_geometry: bool = False,
                     search_kwargs: dict = {}) -> gpd.GeoDataFrame:
        """
        Query STAC catalog for radar frames matching search criteria.

        Multiple parameters are combined with AND logic. Lists of values passed to a
        single parameter are treated with OR logic (any value matches).

        Parameters
        ----------
        collections : list[str] or str, optional
            Collection name(s) (e.g., "2022_Antarctica_BaslerMKB").
        segment_paths : list[str] or str, optional
            Segment path(s) in format "YYYYMMDD_NN" (e.g., "20230126_01").
        geometry : shapely geometry or dict, optional
            Geospatial geometry to filter by intersection.
        date_range : str, optional
            Date range in ISO format (e.g., "2021-01-01/2025-06-01").
        properties : dict, optional
            Additional STAC properties to filter by.
        max_items : int, optional
            Maximum number of items to return.
        exclude_geometry : bool, optional
            If True, exclude geometry field to reduce response size.
        search_kwargs : dict, optional
            Additional keyword arguments for STAC search.

        Returns
        -------
        gpd.GeoDataFrame or None
            GeoDataFrame of matching STAC items with columns for collection,
            geometry, properties, and assets. Returns None if no matches found.
        """

        search_params = {}

        # Exclude geometry -- do not return the geometry field to reduce response size
        if exclude_geometry:
            search_params['exclude'] = ['geometry']

        # Handle collections (seasons)
        if collections is not None:
            search_params['collections'] = [collections] if isinstance(collections, str) else collections

        # Handle geometry filtering
        if geometry is not None:
            if hasattr(geometry, '__geo_interface__'):
                geometry = geometry.__geo_interface__

            # Fix geometries that cross the antimeridian
            geometry = antimeridian.fix_geojson(geometry, reverse=True)

            search_params['intersects'] = geometry

        # Handle date range filtering
        if date_range is not None:
            search_params['datetime'] = date_range

        # Handle max_items
        if max_items is not None:
            search_params['limit'] = max_items

        # Handle segment_paths filtering using CQL2
        filter_conditions = []

        if segment_paths is not None:
            segment_paths = [segment_paths] if isinstance(segment_paths, str) else segment_paths

            # Create OR conditions for segment paths
            segment_conditions = []
            for segment_path in segment_paths:
                try:
                    date_str, segment_num_str = segment_path.split('_')
                    segment_num = int(segment_num_str)

                    # Create AND condition for this specific segment
                    segment_condition = {
                        "op": "and",
                        "args": [
                            {
                                "op": "=",
                                "args": [{"property": "opr:date"}, date_str]
                            },
                            {
                                "op": "=",
                                "args": [{"property": "opr:segment"}, segment_num]
                            }
                        ]
                    }
                    segment_conditions.append(segment_condition)
                except ValueError:
                    print(f"Warning: Invalid segment_path format '{segment_path}'. Expected format: YYYYMMDD_NN")
                    continue

            if segment_conditions:
                if len(segment_conditions) == 1:
                    filter_conditions.append(segment_conditions[0])
                else:
                    # Multiple segments - combine with OR
                    filter_conditions.append({
                        "op": "or",
                        "args": segment_conditions
                    })

        # Add any additional property filters
        for key, value in properties.items():
            filter_conditions.append({
                "op": "=",
                "args": [{"property": key}, value]
            })

        # Combine all filter conditions with AND
        if filter_conditions:
            if len(filter_conditions) == 1:
                filter_expr = filter_conditions[0]
            else:
                filter_expr = {
                    "op": "and",
                    "args": filter_conditions
                }

            search_params['filter'] = filter_expr

        # Add any extra kwargs to search
        search_params.update(search_kwargs)

        #print(search_params) # TODO: Remove

        # Perform the search
        # from rustac import DuckdbClient
        items = self.duckdb_client.search(self.stac_parquet_href, **search_params)
        if isinstance(items, dict):
            items = items['features']

        if not items or len(items) == 0:
            warnings.warn("No items found matching the query criteria", UserWarning)
            return None

        # Convert to GeoDataFrame
        items_df = gpd.GeoDataFrame(items)
        # Set index
        items_df = items_df.set_index(items_df['id'])
        items_df.index.name = 'stac_item_id'
        # Set the geometry column
        if 'geometry' in items_df.columns and not exclude_geometry:
            items_df = items_df.set_geometry(items_df['geometry'].apply(shapely.geometry.shape))
            items_df.crs = "EPSG:4326"

        # Reorder the columns, leaving any extra columns at the end
        desired_order = ['collection', 'geometry', 'properties', 'assets']
        items_df = items_df[[col for col in desired_order if col in items_df.columns] + list(items_df.columns.difference(desired_order))]

        return items_df

    def load_frames(self, stac_items: gpd.GeoDataFrame,
                    data_product: str = "CSARP_standard",
                    image: Union[int, None] = None,
                    merge_flights: bool = False,
                    skip_errors: bool = False,
                    allow_unlisted_products: bool = False
                    ) -> Union[list[xr.Dataset], xr.Dataset]:
        """
        Load multiple radar frames from STAC items.

        Parameters
        ----------
        stac_items : gpd.GeoDataFrame
            STAC items returned from query_frames.
        data_product : str, optional
            Data product to load (e.g., "CSARP_standard", "CSARP_qlook").
        image : int or None, optional
            The image number to load for each frame. If None (default), loads the
            combined image. If specified, `allow_unlisted_products` must be True.
        merge_flights : bool, optional
            If True, merge frames from the same segment into single Datasets.
        skip_errors : bool, optional
            If True, skip failed frames and continue loading others.
        allow_unlisted_products : bool, optional
            If True, attempt to load the specified data product even if it's not
            listed in the item's assets. See `load_frame` for details.

        Returns
        -------
        list[xr.Dataset] or xr.Dataset
            List of radar frame Datasets, or list of merged Datasets if
            merge_flights=True.
        """
        frames = []

        for idx, item in stac_items.iterrows():
            try:
                frame = self.load_frame(item, data_product, image=image, allow_unlisted_products=allow_unlisted_products)
                frames.append(frame)
            except Exception as e:
                print(f"Error loading frame for item {item.get('id', 'unknown')}: {e}")
                if skip_errors:
                    continue
                else:
                    raise e

        if merge_flights:
            return opr_tools.merge_frames(frames)
        else:
            return frames

    def load_frame(self, stac_item, data_product: str = "CSARP_standard",
                   image: Union[int, None] = None,
                   allow_unlisted_products: bool = False) -> xr.Dataset:
        """
        Load a single radar frame from a STAC item.

        Parameters
        ----------
        stac_item : dict or pd.Series
            STAC item containing asset URLs.
        data_product : str, optional
            Data product to load (e.g., "CSARP_standard", "CSARP_qlook").
        image : int or None, optional
            The image number to load for this frame. If None (default), loads the
            combined image. If specified, `allow_unlisted_products` must be True.
        allow_unlisted_products : bool, optional
            If True, attempt to load the specified data product even if it's not
            listed in the item's assets.  This can be useful for loading non-standard
            products. If set to True and the data product is not found in the STAC
            item assets, the method will attempt to construct the URL based on any
            available CSARP_* asset. (If the frame is entirely unlisted, you can use
            `load_frame_url` instead.)
        Returns
        -------
        xr.Dataset
            Radar frame with coordinates slow_time and twtt, and data variables
            including Data, Latitude, Longitude, Elevation, Surface, etc.
        """

        assets = stac_item['assets']

        # Get the data asset
        data_asset = assets.get(data_product)
        if not data_asset:
            if not allow_unlisted_products:
                available_assets = list(assets.keys())
                raise ValueError(f"{data_product} asset not found in STAC item. Available assets: {available_assets}")
            else:
                # Find any available CSARP_* asset to use as a template for constructing the URL
                template_asset_name = None
                template_asset_url = None
                for asset_name, asset_info in assets.items():
                    if asset_name.startswith('CSARP_'):
                        template_asset_name = asset_name
                        template_asset_url = asset_info.get('href')
                        break

                if not template_asset_url:
                    available_assets = list(assets.keys())
                    raise ValueError(f"No CSARP_* asset found in STAC item to use as a template for constructing URL for {data_product}. Available assets: {available_assets}")

                url = template_asset_url.replace(template_asset_name, data_product)
        else:
            # The asset does exist in the STAC item, so just get the URL from the asset
            url = data_asset.get('href')

            if not url:
                raise ValueError(f"No href found in {data_product} asset")

        # If a specific image is requested, modify the URL to point to that image
        if image is not None:
            if not allow_unlisted_products:
                raise ValueError("Specifying an image number requires allow_unlisted_products=True to construct the URL")
            url = url.replace("Data_", f"Data_img_{image:02d}_")

        # Load the frame using the existing method
        return self.load_frame_url(url)

    def load_frame_url(self, url: str) -> xr.Dataset:
        """
        Load a radar frame directly from a URL.

        Automatically detects and handles both HDF5 and legacy MATLAB formats.

        Parameters
        ----------
        url : str
            URL to radar frame data file (.mat).

        Returns
        -------
        xr.Dataset
            Radar frame with CF-compliant metadata and citation information.
        """

        file = self._open_file(url)

        filetype = None
        try:
            ds = self._load_frame_hdf5(file)
            filetype = 'hdf5'
        except OSError:
            ds = self._load_frame_matlab(file)
            filetype = 'matlab'

        # Add the source URL as an attribute
        ds.attrs['source_url'] = url

        # Apply CF-compliant attributes
        ds = apply_cf_compliant_attrs(ds)

        # Get the season and segment from the URL
        match = re.search(r'(\d{4}_\w+_[A-Za-z0-9]+)\/([\w_]+)\/[\d_]+\/[\w]+(\d{8}_\d{2}_\d{3})', url)
        if match:
            collection, data_product, granule = match.groups()
            date, segment_id, frame_id = granule.split('_')
            ds.attrs['collection'] = collection
            ds.attrs['data_product'] = data_product
            ds.attrs['granule'] = granule
            ds.attrs['segment_path'] = f"{date}_{segment_id}"
            ds.attrs['date_str'] = date
            ds.attrs['segment'] = int(segment_id)
            ds.attrs['frame'] = int(frame_id)

            # Load citation information
            result = ops_api.get_segment_metadata(segment_name=ds.attrs['segment_path'], season_name=collection)
            if result:
                if isinstance(result['data'], str):
                    warnings.warn(f"Warning: Unexpected result from ops_api: {result['data']}", UserWarning)
                else:
                    result_data = {}
                    for key, value in result['data'].items():
                        if len(value) == 1:
                            result_data[key] = value[0]
                        elif len(value) > 1:
                            result_data[key] = set(value)

                    if 'dois' in result_data:
                        ds.attrs['doi'] = result_data['dois']
                    if 'rors' in result_data:
                        ds.attrs['ror'] = result_data['rors']
                    if 'funding_sources' in result_data:
                        ds.attrs['funder_text'] = result_data['funding_sources']

        # Add the rest of the Matlab parameters
        if filetype == 'hdf5':
            ds.attrs['mimetype'] = 'application/x-hdf5'
            ds.attrs.update(decode_hdf5_matlab_variable(h5py.File(file, 'r'),
                                                        skip_variables=True,
                                                        skip_errors=True))
        elif filetype == 'matlab':
            ds.attrs['mimetype'] = 'application/x-matlab-data'
            ds.attrs.update(extract_legacy_mat_attributes(file,
                                                          skip_keys=ds.keys(),
                                                          skip_errors=True))

        return ds

    def _load_frame_hdf5(self, file) -> xr.Dataset:
        """
        Load a radar frame from an HDF5 file.

        Parameters
        ----------
        file :
            The path to the HDF5 file containing radar frame data.

        Returns
        -------
        xr.Dataset
            The loaded radar frame as an xarray Dataset.
        """

        ds = xr.open_dataset(file, engine='h5netcdf', phony_dims='sort')

        # Re-arrange variables to provide useful dimensions and coordinates

        ds = ds.squeeze() # Drop the singleton dimensions matlab adds

        ds = ds.rename({ # Label the dimensions with more useful names
            ds.Data.dims[0]: 'slow_time_idx',
            ds.Data.dims[1]: 'twtt_idx',
        })

        # Make variables with no dimensions into scalar attributes
        for var in ds.data_vars:
            if ds[var].ndim == 0:
                ds.attrs[var] = ds[var].item()
                ds = ds.drop_vars(var)

        # Make the file_type an attribute
        if 'file_type' in ds.data_vars:
            ds.attrs['file_type'] = ds['file_type'].to_numpy()
            ds = ds.drop_vars('file_type')

        # Name the two time coordinates
        ds = ds.rename({'Time': 'twtt', 'GPS_time': 'slow_time'})
        ds = ds.set_coords(['slow_time', 'twtt'])

        slow_time_1d = pd.to_datetime(ds['slow_time'].values, unit='s')
        ds = ds.assign_coords(slow_time=('slow_time_idx', slow_time_1d))

        # Make twtt and slow_time the indexing coordinates
        ds = ds.swap_dims({'twtt_idx': 'twtt'})
        ds = ds.swap_dims({'slow_time_idx': 'slow_time'})

        return ds

    def _load_frame_matlab(self, file) -> xr.Dataset:
        """
        Load a radar frame from a MATLAB file.

        Parameters
        ----------
        file :
            The path to the MATLAB file containing radar frame data.

        Returns
        -------
        xr.Dataset
            The loaded radar frame as an xarray Dataset.
        """

        m = scipy.io.loadmat(file, mat_dtype=False)

        key_dims = {
            'Time': ('twtt',),
            'GPS_time': ('slow_time',),
            'Latitude': ('slow_time',),
            'Longitude': ('slow_time',),
            'Elevation': ('slow_time',),
            'Roll': ('slow_time',),
            'Pitch': ('slow_time',),
            'Heading': ('slow_time',),
            'Surface': ('slow_time',),
            'Data': ('twtt', 'slow_time')
        }

        ds = xr.Dataset(
            {
                key: (dims, np.squeeze(m[key])) for key, dims in key_dims.items() if key in m
            },
            coords={
                'twtt': ('twtt', np.squeeze(m['Time'])),
                'slow_time': ('slow_time', pd.to_datetime(np.squeeze(m['GPS_time']), unit='s')),
            }
        )

        return ds

    def get_collections(self) -> list:
        """
        Get all available collections (seasons/campaigns) in STAC catalog.

        Returns
        -------
        list[dict]
            List of collection dictionaries with id, description, and extent.
        """

        return self.duckdb_client.get_collections(self.stac_parquet_href)

    def get_segments(self, collection_id: str) -> list:
        """
        Get all segments (flights) within a collection.

        Parameters
        ----------
        collection_id : str
            STAC collection ID (e.g., "2022_Antarctica_BaslerMKB").

        Returns
        -------
        list[dict]
            List of segment dictionaries with segment_path, date, flight_number,
            and frame count.
        """
        # Query STAC API for all items in collection (exclude geometry for better performance)
        items = self.query_frames(collections=[collection_id], exclude_geometry=True)

        if items is None or len(items) == 0:
            print(f"No items found in collection '{collection_id}'")
            return []

        # Group items by segment (opr:date + opr:segment)
        segments = {}
        for idx, item in items.iterrows():
            properties = item['properties']
            date = properties['opr:date']
            flight_num = properties['opr:segment']

            if date and flight_num is not None:
                segment_path = f"{date}_{flight_num:02d}"

                if segment_path not in segments:
                    segments[segment_path] = {
                        'segment_path': segment_path,
                        'date': date,
                        'flight_number': flight_num,
                        'collection': collection_id,
                        'frames': [],
                        'item_count': 0
                    }

                segments[segment_path]['frames'].append(properties.get('opr:frame'))
                segments[segment_path]['item_count'] += 1

        # Sort segments by date and flight number
        segment_list = list(segments.values())
        segment_list.sort(key=lambda x: (x['date'], x['flight_number']))

        return segment_list

    def _get_layers_files(self, segment: Union[xr.Dataset, dict], raise_errors=True) -> dict:
        """
        Fetch layers from the CSARP_layers files

        See https://gitlab.com/openpolarradar/opr/-/wikis/Layer-File-Guide for file formats

        Parameters
        ----------
        segment : xr.Dataset or dict
            Radar frame Dataset or STAC item.
        raise_errors : bool, optional
            If True, raise errors when layers cannot be found.

        Returns
        -------
        dict
            Dictionary mapping layer names (e.g., "standard:surface",
            "standard:bottom") to layer Datasets.
        """
        collection, segment_path, frame = self._extract_segment_info(segment)

        # If we already have a STAC item, just use it. Otherwise, query to find the matching STAC items
        if isinstance(segment, xr.Dataset):
            properties = {}
            if frame:
                properties['opr:frame'] = frame

            # Query STAC collection for CSARP_layer files matching this specific segment

            # Get items from this specific segment
            stac_items = self.query_frames(collections=[collection], segment_paths=[segment_path], properties=properties)

            # Filter for items that have CSARP_layer assets
            layer_items = []
            for idx, item in stac_items.iterrows():
                if 'CSARP_layer' in item['assets']:
                    layer_items.append(item)
        else:
            layer_items = [segment] if 'CSARP_layer' in segment.get('assets', {}) else []

        if not layer_items:
            if raise_errors:
                raise ValueError(f"No CSARP_layer files found for segment path {segment_path} in collection {collection}")
            else:
                return {}

        # Load each layer file and combine them
        layer_frames = []
        for item in layer_items:
            layer_asset = item['assets']['CSARP_layer']
            if layer_asset and 'href' in layer_asset:
                url = layer_asset['href']
                try:
                    layer_ds = self.load_layers_file(url)
                    layer_frames.append(layer_ds)
                except Exception as e:
                    raise e # TODO
                    print(f"Warning: Failed to load layer file {url}: {e}")
                    continue

        if not layer_frames:
            if raise_errors:
                raise ValueError(f"No valid CSARP_layer files could be loaded for segment {segment_path} in collection {collection}")
            else:
                return {}

        # Concatenate all layer frames along slow_time dimension
        layers_segment = xr.concat(layer_frames, dim='slow_time', combine_attrs='drop_conflicts', data_vars='minimal')
        layers_segment = layers_segment.sortby('slow_time')

        # Trim to bounds of the original dataset
        layers_segment = self._trim_to_bounds(layers_segment, segment)

        # Find the layer organization file to map layer IDs to names
        # Layer organization files are one per directory. For example, the layer file:
        # https://data.cresis.ku.edu/data/rds/2017_Antarctica_BaslerJKB/CSARP_layer/20180105_03/Data_20180105_03_003.mat
        # would have a layer organization file at:
        # https://data.cresis.ku.edu/data/rds/2017_Antarctica_BaslerJKB/CSARP_layer/20180105_03/layer_20180105_03.mat

        layer_organization_file_url = re.sub(r'Data_(\d{8}_\d{2})_\d{3}.mat', r'layer_\1.mat', url)
        layer_organization_ds = self._load_layer_organization(self._open_file(layer_organization_file_url))

        # Split into separate layers by ID
        layers = {}

        layer_ids = np.unique(layers_segment['layer'])

        for i, layer_id in enumerate(layer_ids):
            layer_id_int = int(layer_id)
            layer_name = str(layer_organization_ds['lyr_name'].sel(lyr_id=layer_id_int).values.item().squeeze())
            layer_group = str(layer_organization_ds['lyr_group_name'].sel(lyr_id=layer_id_int).values.item().squeeze())
            if layer_group == '[]':
                layer_group = ''
            layer_display_name = f"{layer_group}:{layer_name}"

            layer_ds = layers_segment.sel(layer=layer_id).drop_vars('layer')
            # Only add non-empty layers with at least some non-NaN data
            if layer_ds.sizes.get('slow_time', 0) > 0:
                if not layer_ds.to_array().isnull().all():
                    layers[layer_display_name] = layer_ds

        return layers

    def _trim_to_bounds(self, ds: xr.Dataset, ref: Union[xr.Dataset, dict]) -> xr.Dataset:
        start_time, end_time = None, None
        if isinstance(ref, xr.Dataset) and 'slow_time' in ref.coords:
            start_time = ref['slow_time'].min()
            end_time = ref['slow_time'].max()
        else:
            properties = ref.get('properties', {})
            if 'start_datetime' in properties and 'end_datetime' in properties:
                start_time = pd.to_datetime(properties['start_datetime'])
                end_time = pd.to_datetime(properties['end_datetime'])

        if start_time:
            return ds.sel(slow_time=slice(start_time, end_time))
        else:
            return ds

    def load_layers_file(self, url: str) -> xr.Dataset:
        """
        Load layer data from CSARP_layer file.

        Parameters
        ----------
        url : str
            URL or path to layer file.

        Returns
        -------
        xr.Dataset
            Layer data with coordinates slow_time and layer, and data variables
            twtt, quality, type, lat, lon, and elev.
            Layer data in a standardized format with coordinates:
            - slow_time: GPS time converted to datetime
            And data variables:
            - twtt: Two-way travel time for each layer
            - quality: Quality values for each layer
            - type: Type values for each layer
            - lat, lon, elev: Geographic coordinates
            - id: Layer IDs
        """

        file = self._open_file(url)

        ds = self._load_layers(file)

        # Add the source URL as an attribute
        ds.attrs['source_url'] = url

        # Apply common manipulations to match the expected structure
        # Convert GPS time to datetime coordinate
        if 'gps_time' in ds.variables:
            slow_time_dt = pd.to_datetime(ds['gps_time'].values, unit='s')
            ds = ds.assign_coords(slow_time=('slow_time', slow_time_dt))

            # Set slow_time as the main coordinate and remove gps_time from data_vars
            if 'slow_time' not in ds.dims:
                ds = ds.swap_dims({'gps_time': 'slow_time'})

            # Remove gps_time from data_vars if it exists there to avoid conflicts
            if ('gps_time' in ds.data_vars) or ('gps_time' in ds.coords):
                ds = ds.drop_vars('gps_time')

        # Sort by slow_time if it exists
        if 'slow_time' in ds.coords:
            ds = ds.sortby('slow_time')

        return ds

    def _load_layers(self, file) -> xr.Dataset:
        """
        Load layer data file using hdf5storage

        Parameters:
        file : str
            Path to the layer file
        Returns: xr.Dataset
        """

        d = hdf5storage.loadmat(file, appendmat=False)

        # Ensure 'id' is at least 1D for the layer dimension
        id_data = np.atleast_1d(d['id'].squeeze())

        # For 2D arrays with (layer, slow_time) dimensions, ensure they remain 2D
        # even when there's only one layer
        quality_data = np.atleast_2d(d['quality'].squeeze())
        if quality_data.shape[0] == 1 or quality_data.ndim == 1:
            quality_data = quality_data.reshape(len(id_data), -1)

        twtt_data = np.atleast_2d(d['twtt'].squeeze())
        if twtt_data.shape[0] == 1 or twtt_data.ndim == 1:
            twtt_data = twtt_data.reshape(len(id_data), -1)

        type_data = np.atleast_2d(d['type'].squeeze())
        if type_data.shape[0] == 1 or type_data.ndim == 1:
            type_data = type_data.reshape(len(id_data), -1)

        ds = xr.Dataset({
            'file_type': ((), d['file_type'].squeeze()),
            'file_version': ((), d['file_version'].squeeze()),
            'elev': (('slow_time',), d['elev'].squeeze()),
            #'gps_source': ((), d['gps_source'].squeeze()),
            'gps_time': (('slow_time',), d['gps_time'].squeeze()),
            'id': (('layer',), id_data),
            'lat': (('slow_time',), d['lat'].squeeze()),
            'lon': (('slow_time',), d['lon'].squeeze()),
            'quality': (('layer', 'slow_time'), quality_data),
            'twtt': (('layer', 'slow_time'), twtt_data),
            'type': (('layer', 'slow_time'), type_data),
        })

        ds = ds.assign_coords({'layer': ds['id'], 'slow_time': ds['gps_time']})

        return ds

    def _load_layer_organization(self, file) -> xr.Dataset:
        """
        Load a layer organization file

        Parameters
        ----------
        file : str
            Path to the HDF5 layer organization file

        Returns
        -------
        xr.Dataset
            Raw layer data from HDF5 file
        """
        d = hdf5storage.loadmat(file, appendmat=False)
        ds = xr.Dataset({
            'file_type': ((), d['file_type'].squeeze()),
            'file_version': ((), d['file_version'].squeeze()),
            'lyr_age': (('lyr_id',), np.atleast_1d(d['lyr_age'].squeeze())),
            'lyr_age_source': (('lyr_id',), np.atleast_1d(d['lyr_age_source'].squeeze())),
            'lyr_desc': (('lyr_id',), np.atleast_1d(d['lyr_desc'].squeeze())),
            'lyr_group_name': (('lyr_id',), np.atleast_1d(d['lyr_group_name'].squeeze())),
            'lyr_id': (('lyr_id',), np.atleast_1d(d['lyr_id'].squeeze())),
            'lyr_name': (('lyr_id',), np.atleast_1d(d['lyr_name'].squeeze())),
            'lyr_order': (('lyr_id',), np.atleast_1d(d['lyr_order'].squeeze())),
            }, attrs={'param': d['param']})

        ds = ds.set_index(lyr_id='lyr_id')

        return ds

    def _get_layers_db(self, flight: Union[xr.Dataset, dict], include_geometry=True, raise_errors=True) -> dict:
        """
        Load layer picks from OPS database API.

        Parameters
        ----------
        flight : xr.Dataset or dict
            Radar frame Dataset or STAC item.
        include_geometry : bool, optional
            If True, include geometry information in layers.
        raise_errors : bool, optional
            If True, raise errors when layers cannot be found.

        Returns
        -------
        dict
            Dictionary mapping layer names to layer Datasets.
        """

        collection, segment_path, _ = self._extract_segment_info(flight)

        if 'Antarctica' in collection:
            location = 'antarctic'
        elif 'Greenland' in collection:
            location = 'arctic'
        else:
            raise ValueError("Dataset does not belong to a recognized location (Antarctica or Greenland).")

        layer_points = ops_api.get_layer_points(
            segment_name=segment_path,
            season_name=collection,
            location=location,
            include_geometry=include_geometry
        )

        if layer_points['status'] != 1:
            if raise_errors:
                raise ValueError(f"Failed to fetch layer points. Received response with status {layer_points['status']}.")
            else:
                return {}

        layer_ds_raw = xr.Dataset(
            {k: (['gps_time'], v) for k, v in layer_points['data'].items() if k != 'gps_time'},
            coords={'gps_time': layer_points['data']['gps_time']}
        )
        # Split into a dictionary of layers based on lyr_id
        layer_ids = set(layer_ds_raw['lyr_id'].to_numpy())
        layer_ids = [int(layer_id) for layer_id in layer_ids if not np.isnan(layer_id)]

        # Get the mapping of layer IDs to names
        if self.db_layers_metadata is None:
            layer_metadata = ops_api.get_layer_metadata()
            if layer_metadata['status'] == 1:
                df = pd.DataFrame(layer_metadata['data'])
                df['display_name'] = df['lyr_group_name'] + ":" + df['lyr_name']
                self.db_layers_metadata = df
            else:
                raise ValueError("Failed to fetch layer metadata from OPS API.")

        layers = {}
        for layer_id in layer_ids:
            layer_name = self.db_layers_metadata.loc[self.db_layers_metadata['lyr_id'] == layer_id, 'display_name'].values[0]

            layer_ds = layer_ds_raw.where(layer_ds_raw['lyr_id'] == layer_id, drop=True)

            layer_ds = layer_ds.sortby('gps_time')

            layer_ds = layer_ds.rename({'gps_time': 'slow_time'})
            layer_ds = layer_ds.set_coords(['slow_time'])

            layer_ds['slow_time'] = pd.to_datetime(layer_ds['slow_time'].values, unit='s')

            # Filter to the same time range as flight
            layer_ds = self._trim_to_bounds(layer_ds, flight)

            # Only add non-empty layers with at least some non-NaN data
            if layer_ds.sizes.get('slow_time', 0) > 0:
                if not layer_ds.to_array().isnull().all():
                    layers[layer_name] = layer_ds

        return layers

    def get_layers(self, ds : Union[xr.Dataset, dict], source: str = 'auto', include_geometry=True, errors='warn') -> dict:
        """
        Load layer picks from files or database.

        Tries files first, falls back to database if source='auto'.

        Parameters
        ----------
        ds : Union[xr.Dataset, dict]
            Radar frame Dataset or STAC item.
        source : {'auto', 'files', 'db'}, optional
            Source for layers. 'auto' tries files then falls back to database.
        include_geometry : bool, optional
            If True, include geometry information when fetching from the API.
        errors : str, optional
            How to handle missing layer data: 'warn' (default) returns None with a warning,
            'error' raises an exception.

        Returns
        -------
        dict or None
            A dictionary mapping layer IDs to their corresponding data, or None if no layers
            found and errors='warn'.
        """
        collection, segment_path, _ = self._extract_segment_info(ds)

        if source == 'auto':
            # Try to get layers from files first
            try:
                layers = self._get_layers_files(ds, raise_errors=True)
                return layers
            except Exception:
                # Fallback to API if no layers found in files
                try:
                    return self._get_layers_db(ds, include_geometry=include_geometry, raise_errors=True)
                except Exception as e:
                    if errors == 'error':
                        raise
                    else:
                        warnings.warn(f"No layer data found for {collection}/{segment_path}: {e}")
                        return None
        elif source == 'files':
            try:
                return self._get_layers_files(ds, raise_errors=True)
            except Exception as e:
                if errors == 'error':
                    raise
                else:
                    warnings.warn(f"No layer files found for {collection}/{segment_path}: {e}")
                    return None
        elif source == 'db':
            try:
                return self._get_layers_db(ds, include_geometry=include_geometry, raise_errors=True)
            except Exception as e:
                if errors == 'error':
                    raise
                else:
                    warnings.warn(f"No layer data in DB for {collection}/{segment_path}: {e}")
                    return None
        else:
            raise ValueError("Invalid source specified. Must be one of: 'auto', 'files', 'db'.")

    def _row_to_stac_dict(self, row):
        """Convert a frames-gdf row (flat or nested) to a STAC-item-style dict.

        The :meth:`query_frames` form keeps STAC properties as a nested
        ``properties`` column. Flattened catalogs (e.g. source.coop parquets)
        instead have ``opr:date``/``opr:segment``/``opr:frame`` as top-level
        columns. Both forms are normalized here.
        """
        if 'properties' in row.index and isinstance(row.get('properties'), dict):
            properties = row['properties']
        else:
            properties = {}
            for col in ('opr:date', 'opr:segment', 'opr:frame', 'opr:mbox'):
                if col in row.index:
                    properties[col] = row[col]
        return {
            'id': row.get('id'),
            'collection': row.get('collection'),
            'geometry': row.get('geometry'),
            'properties': properties,
            'assets': row.get('assets', {}) if 'assets' in row.index else {},
        }

    def _normalize_frames_input(self, frames):
        """Normalize a frames input (gdf, Series, dict, pystac.Item) to a list of STAC dicts."""
        if isinstance(frames, gpd.GeoDataFrame) or isinstance(frames, pd.DataFrame):
            return [self._row_to_stac_dict(row) for _, row in frames.iterrows()]
        if isinstance(frames, pd.Series):
            return [self._row_to_stac_dict(frames)]
        if isinstance(frames, dict):
            return [frames]
        if hasattr(frames, 'to_dict'):  # pystac.Item or compatible
            return [frames.to_dict()]
        raise TypeError(
            f"frames must be GeoDataFrame, Series, dict, or pystac.Item; got {type(frames).__name__}"
        )

    def load_bed_picks(
        self,
        frames,
        layer: str = 'standard:bottom',
        vertical: str = 'wgs84',
        target_crs: str = None,
        keep_mbox: bool = False,
        show_progress: bool = True,
    ) -> gpd.GeoDataFrame:
        """
        Load layer picks for one or many frames into a flat GeoDataFrame.

        For each input frame this fetches the requested layer plus the
        ``standard:surface`` layer (needed to convert TWTT to elevation),
        drops samples with missing layer data, and tags every pick with
        the source frame's STAC id and frame identifiers.

        The output schema is a strict superset of the layer-parquet
        schema, so it is a drop-in ``layer_gdf`` for
        :func:`xopr.bedmap.morton_match.disambiguate_matches`.

        Parameters
        ----------
        frames : gpd.GeoDataFrame, pd.Series, dict, or pystac.Item
            One or many frames to load picks for. Accepts:

            - GeoDataFrame from :meth:`query_frames` (one row per frame)
            - A single row of that GeoDataFrame (``pd.Series``)
            - A STAC item as nested dict (e.g. ``item.to_dict()``)
            - A ``pystac.Item`` instance
        layer : str, default 'standard:bottom'
            Layer name in the OPR layer file.
        vertical : {'wgs84', 'range'}, default 'wgs84'
            Vertical coordinate to compute. ``'wgs84'`` requires the
            surface layer's ``elev`` field.
        target_crs : str, optional
            If given, project pick coordinates to this CRS (e.g.
            ``'EPSG:3031'``). If None, picks stay in EPSG:4326.
        keep_mbox : bool, default False
            If True, attach the source frame's ``opr:mbox`` to every pick
            row (handy for downstream morton filtering without re-loading
            the catalog).
        show_progress : bool, default True
            Show a tqdm progress bar when loading multiple frames.

        Returns
        -------
        gpd.GeoDataFrame
            One row per pick. Columns:

            - ``geometry`` — Point in ``target_crs`` (or EPSG:4326)
            - ``<vertical>`` — the requested vertical coordinate
            - ``twtt`` — two-way travel time
            - ``slow_time`` — pick timestamp
            - ``id`` — STAC item id of the source frame
            - ``collection`` — STAC collection name
            - ``opr:date``, ``opr:segment``, ``opr:frame`` — frame identifiers
            - ``segment_path`` — ``f"{opr:date}_{opr:segment:02d}"``
            - ``opr:mbox`` — only if ``keep_mbox=True``

            Frames whose layer is unavailable are silently skipped.

        Examples
        --------
        >>> opr = xopr.OPRConnection()
        >>> frames = opr.query_frames(collections=['2009_Antarctica_DC8'])
        >>> picks = opr.load_bed_picks(frames, target_crs='EPSG:3031')
        >>> picks[['id', 'wgs84', 'segment_path']].head()
        """
        frames_list = self._normalize_frames_input(frames)

        iterator = frames_list
        if show_progress and len(frames_list) > 1:
            try:
                from tqdm.auto import tqdm
                iterator = tqdm(frames_list, desc='Loading bed picks')
            except ImportError:
                pass

        pick_dfs = []
        for frame_dict in iterator:
            try:
                layers = self.get_layers(frame_dict, errors='warn')
            except Exception:
                continue
            if layers is None or layer not in layers or 'standard:surface' not in layers:
                continue

            ds = layer_twtt_to_range(
                layers[layer], layers['standard:surface'],
                vertical_coordinate=vertical,
            )
            ds = ds.dropna('slow_time', subset=[vertical])
            n = ds.sizes.get('slow_time', 0)
            if n == 0:
                continue

            props = frame_dict.get('properties', {}) or {}
            opr_date = props.get('opr:date')
            opr_segment = int(props['opr:segment']) if props.get('opr:segment') is not None else None
            opr_frame = int(props['opr:frame']) if props.get('opr:frame') is not None else None
            segment_path = (
                f"{opr_date}_{opr_segment:02d}"
                if opr_date is not None and opr_segment is not None
                else None
            )

            df = pd.DataFrame({
                vertical: ds[vertical].values,
                'twtt': ds['twtt'].values if 'twtt' in ds else np.full(n, np.nan),
                'slow_time': ds['slow_time'].values,
                'id': frame_dict.get('id'),
                'collection': frame_dict.get('collection'),
                'opr:date': opr_date,
                'opr:segment': opr_segment,
                'opr:frame': opr_frame,
                'segment_path': segment_path,
                '_lon': ds['lon'].values,
                '_lat': ds['lat'].values,
            })
            if keep_mbox and 'opr:mbox' in props:
                df['opr:mbox'] = [props['opr:mbox']] * n
            pick_dfs.append(df)

        if not pick_dfs:
            cols = [vertical, 'twtt', 'slow_time', 'id', 'collection',
                    'opr:date', 'opr:segment', 'opr:frame', 'segment_path']
            if keep_mbox:
                cols.append('opr:mbox')
            empty = pd.DataFrame({c: pd.Series(dtype='object') for c in cols})
            out = gpd.GeoDataFrame(empty, geometry=gpd.GeoSeries([], crs='EPSG:4326'))
            return out.to_crs(target_crs) if target_crs else out

        combined = pd.concat(pick_dfs, ignore_index=True)
        geom = gpd.points_from_xy(combined['_lon'], combined['_lat'])
        combined = combined.drop(columns=['_lon', '_lat'])
        out = gpd.GeoDataFrame(combined, geometry=geom, crs='EPSG:4326')
        if target_crs is not None:
            out = out.to_crs(target_crs)
        return out
